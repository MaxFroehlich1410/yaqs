# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compressed pure and locally purified comb objectives using Ballarin's kernels.

The state is propagated on a comb tree. Closed bonds and local purification
legs are compressed using the author's normalized, regularized SVD. No global
simulated vector, density matrix, or unitary is formed. Dense target conversion
is a separate, explicitly bounded convenience for the small benchmark fixtures.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from experiments.state_preparation_competitors.evaluation import pauli_distribution
from experiments.state_preparation_competitors.iqsp_tn_backend import CircuitTensorGates, tensor_angle
from experiments.state_preparation_competitors.vendor.ballarin_comb.combdevice import CombTNDevice
from experiments.state_preparation_competitors.vendor.ballarin_comb.noisy_combdevice import NoisyCombTNDevice
from experiments.state_preparation_competitors.vendor.ballarin_comb.svd import truncated_svd_gesdd

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike, NDArray
    from qiskit import QuantumCircuit


@dataclass(frozen=True)
class CombLayout:
    """Equal-length teeth with their first sites connected along a backbone.

    Site numbering is ascending Qiskit qubit order, grouped by tooth. Root
    tensors have axes (backbone left, physical, tooth, backbone right); other
    tensors have axes (toward root, physical, toward tip).
    """

    sites_per_tooth: int
    teeth: int = 1

    def __post_init__(self) -> None:
        """Reject unsupported layouts, including one-site teeth."""
        if type(self.sites_per_tooth) is not int or self.sites_per_tooth < 2:
            msg = "sites_per_tooth must be an integer of at least two."
            raise ValueError(msg)
        if type(self.teeth) is not int or self.teeth < 1:
            msg = "teeth must be a positive integer."
            raise ValueError(msg)

    @property
    def num_qubits(self) -> int:
        """Total register size."""
        return self.sites_per_tooth * self.teeth

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Tooth edges followed by backbone edges."""
        width = self.sites_per_tooth
        tooth = [(root + j, root + j + 1) for root in range(0, self.num_qubits, width) for j in range(width - 1)]
        spine = [(root, root + width) for root in range(0, self.num_qubits - width, width)]
        return tuple(tooth + spine)

    def bond_axis(self, first: int, second: int) -> int:
        """Return the first tensor's axis connecting a neighboring second site."""
        low, high = sorted((first, second))
        if (low, high) not in self.edges:
            msg = "Two-qubit operations must follow an edge of the declared comb."
            raise ValueError(msg)
        return 0 if first == high else (3 if high - low == self.sites_per_tooth else 2)


@dataclass(frozen=True)
class CombOptions:
    """Author backend defaults, plus an explicit local tensor allocation limit.

    Tolerances bound the relative discarded squared singular values at each
    split, before the author's norm-restoring rescaling; they are not a bound
    on the final circuit fidelity error. Caps can force larger local errors.
    """

    closed_bond: int = 64
    environment_bond: int = 16
    closed_tolerance: float = 1e-9
    environment_tolerance: float = 1e-12
    max_tensor_elements: int = 2**24
    checkpoint_interval: int = 0
    target_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Require finite, nonnegative tolerances and positive integer caps."""
        for name in ("closed_bond", "environment_bond", "max_tensor_elements"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)
        for name in ("closed_tolerance", "environment_tolerance", "target_tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                msg = f"{name} must be finite and in [0,1)."
                raise ValueError(msg)
        if type(self.checkpoint_interval) is not int or self.checkpoint_interval < 0:
            msg = "checkpoint_interval must be a nonnegative integer; zero retains the full autograd tape."
            raise ValueError(msg)


@dataclass(frozen=True)
class CombTarget:
    """Normalized pure comb target; tensors are privately copied on construction."""

    layout: CombLayout
    tensors: tuple[torch.Tensor, ...]
    source: str = "supplied_comb_tensors"

    def __post_init__(self) -> None:
        """Validate physical dimensions, matching bonds, boundaries, and norm."""
        if len(self.tensors) != self.layout.num_qubits:
            msg = "Target tensor count must match the comb layout."
            raise ValueError(msg)
        private = tuple(t.detach().clone().to(dtype=torch.complex128, device="cpu") for t in self.tensors)
        for site, tensor in enumerate(private):
            root = site % self.layout.sites_per_tooth == 0
            if tensor.ndim != (4 if root else 3) or tensor.shape[1] != 2 or not bool(torch.isfinite(tensor).all()):
                msg = "Target comb tensors require finite entries and physical dimension two."
                raise ValueError(msg)
            used = {1}
            for a, b in self.layout.edges:
                if site in {a, b}:
                    other = b if site == a else a
                    axis = self.layout.bond_axis(site, other)
                    used.add(axis)
                    if tensor.shape[axis] != private[other].shape[self.layout.bond_axis(other, site)]:
                        msg = "Target comb bond dimensions do not match."
                        raise ValueError(msg)
            if any(tensor.shape[axis] != 1 for axis in range(tensor.ndim) if axis not in used):
                msg = "Target comb boundary bonds must have dimension one."
                raise ValueError(msg)
        object.__setattr__(self, "tensors", private)
        state = CombTNDevice(self.layout.sites_per_tooth, self.layout.teeth)
        state.states = list(private)
        norm = complex(state.overlap(private).item())
        if abs(norm - 1) > 1e-9:
            msg = "Target comb must be normalized."
            raise ValueError(msg)


def comb_target_from_vector(target: ArrayLike, layout: CombLayout, *, relative_tolerance: float = 0.0) -> CombTarget:
    """Factor a q2-q12 fixture by successive leaf SVDs, exact by default.

    This is a dense-input convenience, not a replacement for the publication's
    TCI construction at large sizes. Supply ``CombTarget`` directly there.

    Args:
        target: Normalized little-endian amplitudes, at most twelve qubits.
        layout: Requested comb topology.
        relative_tolerance: Optional discarded squared-weight tolerance per split.
            Zero preserves every direction, including exact zeros. The campaign
            may explicitly remove numerical null directions using 1e-28.

    Returns:
        A normalized target; zero tolerance retains every singular direction.
    """
    vector = np.asarray(target, dtype=np.complex128)
    CombOptions(target_tolerance=relative_tolerance)
    n = layout.num_qubits
    if n > 12 or vector.shape != (2**n,) or not np.isfinite(vector).all() or abs(np.vdot(vector, vector) - 1) > 1e-10:
        msg = "Expected a finite normalized q2-q12 target matching the comb layout."
        raise ValueError(msg)
    rest = vector.reshape((2,) * n).transpose(tuple(reversed(range(n))))
    labels = [("p", j) for j in range(n)]
    tensors: list[torch.Tensor | None] = [None] * n
    width = layout.sites_per_tooth
    for root in range(0, n, width):
        for site in range(root + width - 1, root, -1):
            group = [("p", site)] + ([("b", site)] if site < root + width - 1 else [])
            left, rest, labels = _target_split(rest, labels, group, ("b", site - 1), relative_tolerance)
            right = left.shape[1] if left.ndim == 3 else 1
            tensors[site] = torch.tensor(left.reshape(2, right, -1).transpose(2, 0, 1))
        group = ([("s", root - width)] if root else []) + [("p", root), ("b", root)]
        if root + width < n:
            left, rest, labels = _target_split(rest, labels, group, ("s", root), relative_tolerance)
            dims = (1, *left.shape) if root == 0 else left.shape
            tensors[root] = torch.tensor(left.reshape(dims))
        else:
            array = rest.transpose([labels.index(label) for label in group])
            if root == 0:
                array = array[None, ...]
            tensors[root] = torch.tensor(array[..., None])
    source = (
        "exact_dense_fixture_leaf_svd"
        if relative_tolerance == 0
        else f"dense_fixture_leaf_svd_rtol_{relative_tolerance}"
    )
    if relative_tolerance:
        # The leaf factors are isometries; normalize the final center only.
        center = tensors[-width]
        assert center is not None
        tensors[-width] = center / torch.linalg.vector_norm(center)
    return CombTarget(layout, tuple(t for t in tensors if t is not None), source)


def _target_split(
    tensor: NDArray[np.complex128],
    labels: list[tuple[str, int]],
    group: list[tuple[str, int]],
    bond: tuple[str, int],
    tolerance: float = 0.0,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], list[tuple[str, int]]]:
    """Peel one leaf group from a dense target, retaining numerical rank."""
    remaining = [label for label in labels if label not in group]
    axes = [labels.index(label) for label in group + remaining]
    grouped = tensor.transpose(axes)
    shape = grouped.shape[: len(group)]
    other = grouped.shape[len(group) :]
    u, singular, vh = np.linalg.svd(grouped.reshape(math.prod(shape), -1), full_matrices=False)
    # Preserve every singular direction unless a target tolerance is explicitly requested.
    rank = singular.size
    if tolerance:
        tail = np.concatenate((np.cumsum(singular[::-1] ** 2)[::-1], [0.0]))
        rank = int(np.flatnonzero(tail[1:] <= tolerance * tail[0])[0]) + 1
        u, singular, vh = u[:, :rank], singular[:rank], vh[:rank]
    return u.reshape(*shape, rank), (singular[:, None] * vh).reshape(rank, *other), [bond, *remaining]


class CombTensorNetworkObjective(CircuitTensorGates):
    """Ballarin comb propagation with explicit pure, H2, or logical noise.

    ``hardware='H2'`` implements the paper's Eq. (8)-(9): noiseless one-qubit
    gates and angle-dependent local depolarization after RZZ gates. ``IBM`` and
    ``Google`` are explicitly shared-benchmark channels including correlated
    two-site errors. They are never substituted for H2 silently.
    """

    def __init__(
        self,
        template: QuantumCircuit,
        target: CombTarget,
        hardware: str | None = None,
        *,
        options: CombOptions | None = None,
    ) -> None:
        """Validate topology and cache local gates, without allocating a global state."""
        self.comb_target = target
        self.layout = target.layout
        self.options = options or CombOptions()
        self.noise_profile = hardware
        super().__init__(template, None if hardware == "H2" else hardware)
        if template.num_qubits != self.layout.num_qubits:
            msg = "Circuit and target comb must have matching register sizes."
            raise ValueError(msg)
        for item in template.data:
            sites = [template.find_bit(q).index for q in item.qubits]
            if len(sites) == 2:
                self.layout.bond_axis(*sites)
                if hardware == "H2" and item.operation.name != "rzz":
                    msg = "H2 noisy training requires native RZZ two-qubit gates."
                    raise ValueError(msg)
        self.last_diagnostics: dict[str, Any] = {}

    def _split(
        self, matrix: torch.Tensor, chi: int, *, rel_tol: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the author's normalized SVD and record actual discarded weight."""
        self._allocation(matrix.numel())
        u, s, v, discarded = truncated_svd_gesdd(matrix, chi, rel_tol=rel_tol, verbosity=1)
        total = float(matrix.detach().abs().square().sum())
        self.last_diagnostics["svd_splits"] += 1
        self.last_diagnostics["sum_local_relative_discarded_weight"] += float(discarded.detach()) / max(total, 1e-300)
        return u, s, v

    def _allocation(self, elements: int) -> None:
        """Guard and record local tensor allocations before forming merged tensors."""
        if elements > self.options.max_tensor_elements:
            msg = "Comb local tensor exceeds max_tensor_elements; no dense fallback is permitted."
            raise ValueError(msg)
        self.last_diagnostics["largest_local_tensor_elements"] = max(
            self.last_diagnostics["largest_local_tensor_elements"], elements
        )

    def _new_state(self, *, reset_diagnostics: bool = True) -> CombTNDevice | NoisyCombTNDevice:
        """Create a fresh zero state and install per-instance compression accounting."""
        if reset_diagnostics:
            self.last_diagnostics = {
                "svd_splits": 0,
                "sum_local_relative_discarded_weight": 0.0,
                "largest_local_tensor_elements": 0,
            }
        state: CombTNDevice | NoisyCombTNDevice
        if self.noise_profile is None:
            state = CombTNDevice(
                self.layout.sites_per_tooth,
                self.layout.teeth,
                ctol=self.options.closed_tolerance,
                cmbd=self.options.closed_bond,
            )
        else:
            state = NoisyCombTNDevice(
                self.layout.sites_per_tooth,
                self.layout.teeth,
                ctol=self.options.closed_tolerance,
                cmbd=self.options.closed_bond,
                otol=self.options.environment_tolerance,
                ombd=self.options.environment_bond,
            )
        # Extracted kernels use only (matrix, chi, rel_tol), unlike the full upstream signature.
        state.svd_decomposition = cast("Any", self._split)
        return state

    def _environment_compress(self, state: NoisyCombTNDevice, site: int, expanded: torch.Tensor) -> None:
        """Compress the local purification leg with the center located at this site."""
        # Explicit axes: left, physical, environment, remaining virtual bonds.
        order = [0, 1, *range(3, expanded.ndim), 2]
        arranged = expanded.permute(order)
        self._allocation(arranged.numel())
        u, singular, _ = self._split(
            arranged.reshape(-1, arranged.shape[-1]),
            self.options.environment_bond,
            rel_tol=self.options.environment_tolerance,
        )
        factor = (u * singular).reshape(*arranged.shape[:-1], -1)
        result = factor.permute(0, 1, factor.ndim - 1, *range(2, factor.ndim - 1))
        state.osys_dim[site] = singular.numel()
        state[site] = result.reshape(result.shape[0], 2 * singular.numel(), *result.shape[3:])

    def _one_site_channel(self, state: NoisyCombTNDevice, site: int, kraus: torch.Tensor) -> None:
        """Apply local Kraus matrices and compress their environment leg."""
        state.iso_towards(site)
        tensor = state[site]
        tensor = tensor.reshape(tensor.shape[0], 2, state.osys_dim[site], *tensor.shape[2:])
        self._allocation(tensor.numel() * kraus.shape[0])
        expanded = torch.tensordot(tensor, kraus, dims=([1], [2]))
        # left, old environment, virtual..., Kraus, output physical
        expanded = expanded.permute(0, expanded.ndim - 1, 1, expanded.ndim - 2, *range(2, expanded.ndim - 2))
        expanded = expanded.reshape(expanded.shape[0], 2, -1, *tensor.shape[3:])
        self._environment_compress(state, site, expanded)

    def _two_site_channel(self, state: NoisyCombTNDevice, sites: list[int], kraus: torch.Tensor) -> None:
        """Apply a joint channel with one shared Kraus index, preserving correlations."""
        first, second = sorted(sites)
        state.iso_towards(first)
        a, b = state[first], state[second]
        bond_a = self.layout.bond_axis(first, second)
        bond_b = self.layout.bond_axis(second, first)
        shape_a, shape_b = list(a.shape), list(b.shape)
        a = a.reshape(a.shape[0], 2, state.osys_dim[first], *a.shape[2:])
        b = b.reshape(b.shape[0], 2, state.osys_dim[second], *b.shape[2:])
        axis_a, axis_b = bond_a + 1, bond_b
        labels_a = list(range(a.ndim))
        labels_b = list(range(a.ndim, a.ndim + b.ndim))
        labels_b[axis_b] = labels_a[axis_a]
        fresh = a.ndim + b.ndim
        out_a, out_b, noise = fresh, fresh + 1, fresh + 2
        ka = kraus.reshape(-1, 2, 2, 2, 2)
        if sites[0] == first:  # Convert Qiskit's reversed local tensor order.
            ka = ka.permute(0, 2, 1, 4, 3)
        left = [label for axis, label in enumerate(labels_a) if axis != axis_a]
        right = [label for axis, label in enumerate(labels_b) if axis != axis_b]
        left[left.index(labels_a[1])] = out_a
        left.insert(left.index(labels_a[2]) + 1, noise)
        right[right.index(labels_b[1])] = out_b
        self._allocation(a.numel() * b.numel() // (a.shape[axis_a] ** 2) * kraus.shape[0])
        joined = torch.einsum(
            a, labels_a, b, labels_b, ka, [noise, out_a, out_b, labels_a[1], labels_b[1]], left + right
        )
        left_size = math.prod([dim for axis, dim in enumerate(a.shape) if axis != axis_a]) * kraus.shape[0]
        u, singular, v = self._split(
            joined.reshape(left_size, -1), self.options.closed_bond, rel_tol=self.options.closed_tolerance
        )
        # Keep the center on the first site for the subsequent environment SVD.
        left_shape = [dim for axis, dim in enumerate(a.shape) if axis != axis_a]
        left_shape[2] *= kraus.shape[0]
        left_tensor = (u * singular).reshape(*left_shape, -1)
        left_tensor = left_tensor.movedim(-1, axis_a)
        r = v.mH.reshape(singular.numel(), *[dim for axis, dim in enumerate(b.shape) if axis != axis_b])
        r = r.movedim(0, axis_b)
        shape_a[bond_a] = shape_b[bond_b] = singular.numel()
        shape_a[1] *= kraus.shape[0]
        state.osys_dim[first] *= kraus.shape[0]
        state[first], state[second] = left_tensor.reshape(shape_a), r.reshape(shape_b)
        state.iso_center = first
        self._environment_compress(state, first, left_tensor)

    def _h2_channel(self, state: NoisyCombTNDevice, index: int, theta: torch.Tensor, sites: list[int]) -> None:
        """Apply Eq. (8)-(9), including the angle derivative and constant error."""
        # The gate generator has angle theta/2; theta is in radians, canonicalized.

        angle = tensor_angle(self.template.data[index].operation.params[0], theta, self.parameters)
        canonical = torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
        epsilon = 2.1e-4 + 1.43e-3 * canonical.abs()
        r = (1 - torch.sqrt(1 - 1.25 * epsilon)) / 3
        paulis = torch.tensor(
            [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=torch.complex128
        )
        probabilities = torch.stack([1 - 3 * r, r, r, r])
        kraus = probabilities.sqrt()[:, None, None] * paulis
        for site in sites:
            self._one_site_channel(state, site, kraus)

    def _apply_interval(
        self, state: CombTNDevice | NoisyCombTNDevice, theta: torch.Tensor, start: int, end: int
    ) -> None:
        """Apply an unchanged gate/channel interval to the supplied private state."""
        for index in range(start, end):
            item = self.template.data[index]
            sites = [self.template.find_bit(q).index for q in item.qubits]
            matrix = self._matrix(index, theta)
            if len(sites) == 1:
                state.apply_one_site_operator(sites[0], matrix)
            else:
                first, second = sites
                axis = self.layout.bond_axis(first, second)
                self._allocation(state[first].numel() * state[second].numel() // state[first].shape[axis] ** 2)
                state.apply_two_sites_operator(second, first, matrix)
            if self.noise_profile == "H2" and len(sites) == 2:
                assert isinstance(state, NoisyCombTNDevice)
                self._h2_channel(state, index, theta, sites)
            elif self.noise_profile in {"IBM", "Google"}:
                assert isinstance(state, NoisyCombTNDevice)
                probabilities, matrices = pauli_distribution(self.noise_profile, len(sites))
                kraus = torch.tensor(np.sqrt(probabilities)[:, None, None] * np.stack(matrices), dtype=torch.complex128)
                if len(sites) == 1:
                    self._one_site_channel(state, sites[0], kraus)
                else:
                    self._two_site_channel(state, sites, kraus)

    @contextmanager
    def _recompute_diagnostics(self) -> Iterator[None]:
        """Keep backward recomputation from double counting physical truncations."""
        saved = self.last_diagnostics
        self.last_diagnostics = dict(saved)
        try:
            yield
        finally:
            self.last_diagnostics = saved

    def _checkpoint_segment(
        self,
        start: int,
        end: int,
        center: int,
        environment: tuple[int, ...],
        theta: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Restore an independent state so recomputation never mutates saved inputs."""
        state = self._new_state(reset_diagnostics=False)
        state.states = list(tensors)
        state.iso_center = center
        if isinstance(state, NoisyCombTNDevice):
            state.osys_dim = list(environment)
        self._apply_interval(state, theta, start, end)
        dims = state.osys_dim if isinstance(state, NoisyCombTNDevice) else [1] * self.layout.num_qubits
        return (*state.states, torch.tensor(state.iso_center), torch.tensor(dims))

    def _propagate(self, theta: torch.Tensor) -> CombTNDevice | NoisyCombTNDevice:
        """Execute the same contractions with optional exact autograd recomputation."""
        state = self._new_state()
        interval = self.options.checkpoint_interval
        if not interval or not torch.is_grad_enabled() or not theta.requires_grad:
            self._apply_interval(state, theta, 0, len(self.template.data))
            return state
        for start in range(0, len(self.template.data), interval):
            environment = tuple(state.osys_dim) if isinstance(state, NoisyCombTNDevice) else ()
            segment = partial(
                self._checkpoint_segment,
                start,
                min(start + interval, len(self.template.data)),
                state.iso_center,
                environment,
            )
            output = cast(
                "tuple[torch.Tensor, ...]",
                checkpoint(
                    segment,
                    theta,
                    *state.states,
                    use_reentrant=False,
                    context_fn=lambda: (nullcontext(), self._recompute_diagnostics()),
                ),
            )
            state.states = list(output[:-2])
            state.iso_center = int(output[-2])
            if isinstance(state, NoisyCombTNDevice):
                state.osys_dim = output[-1].tolist()
        return state

    def _fidelity_tensor(self, theta: torch.Tensor) -> torch.Tensor:
        """Contract compressed output with the pure target without dense conversion."""
        state = self._propagate(theta)
        overlap = state.overlap(self.comb_target.tensors).squeeze()
        result = overlap.abs().square() if self.noise_profile is None else overlap.real
        self.last_diagnostics.update(
            stored_tensor_elements=sum(t.numel() for t in state.states),
            maximum_closed_bond=max(
                t.shape[self.layout.bond_axis(a, b)] for a, b in self.layout.edges for t in [state[a]]
            ),
            maximum_environment_bond=max(state.osys_dim) if isinstance(state, NoisyCombTNDevice) else 1,
            trace=float(CombTNDevice.overlap(cast("Any", state), state.states).detach().real.squeeze()),
        )
        return result

    def loss_gradient(self, theta: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Return compressed infidelity and the author's regularized SVD gradient."""
        values = self._theta(theta, gradients=True)
        loss = 1 - self._fidelity_tensor(values)
        gradient = torch.autograd.grad(loss, values)[0].detach().numpy() if self.parameters else np.empty(0)
        result = float(loss.detach())
        if not math.isfinite(result) or not np.isfinite(gradient).all():
            msg = "Nonfinite compressed-comb objective or gradient."
            raise FloatingPointError(msg)
        return result, gradient

    def fidelity(self, theta: ArrayLike) -> float:
        """Evaluate the same deterministic compressed objective without autograd."""
        with torch.no_grad():
            return float(self._fidelity_tensor(self._theta(theta)))

    def diagnostics(self, theta: ArrayLike) -> dict[str, Any]:
        """Report measured ranks, storage, discarded weights and implementation scope."""
        self.fidelity(theta)
        return {
            **self.last_diagnostics,
            "backend": "ballarin_compressed_comb" if self.noise_profile is None else "ballarin_locally_purified_comb",
            "layout": asdict(self.layout),
            "compression": asdict(self.options),
            "noise_profile": self.noise_profile,
            "global_state_or_density_materialized": False,
            "positive_semidefinite_by_construction": True,
            "gradient": "author regularized SVD backward; approximate at degeneracies/rank changes",
            "target_source": self.comb_target.source,
            "target_tensor_elements": sum(t.numel() for t in self.comb_target.tensors),
            "upstream_commit": "5bbd7353f63dfc868c3b0a0d6f29f5c86506f207",
        }
