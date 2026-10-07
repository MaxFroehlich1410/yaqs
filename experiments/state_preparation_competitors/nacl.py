# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Paper-based NACL reconstruction with the upstream OrthoMADS optimizer.

The author implementation is not available to this benchmark. Structural moves,
annealing, and native circuit simplification follow Section II.E of Cincio et al.;
choices not specified there are explicit controls and provenance. This is neither
an author-code reproduction nor a gradient-optimizer substitution.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import itertools
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from experiments.state_preparation_competitors.common import Candidate, validate_target

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Protocol

    from numpy.typing import ArrayLike, NDArray

    class NomadPoint(Protocol):
        """Structural typing for the upstream extension's blackbox point."""

        def get_coord(self, index: int) -> float:
            """Return one optimization coordinate."""
            ...

        def setBBO(self, value: bytes) -> None:  # noqa: N802  # Match the upstream public Cython API.
            """Assign the ASCII blackbox output."""
            ...


NOMAD_VERSION = "4.6.0"
INNER_SEED_RANGE = 2**16
ALGORITHM = "nacl_paper_based_orthomads_2021"
PAPER_DOI = "10.1103/PRXQuantum.2.010324"


@dataclass(frozen=True)
class NativeGate:
    """An RZ/SX/CX operation with an angle only for RZ, in execution order."""

    name: str
    sites: tuple[int, ...]
    angle: float = 0.0

    def __post_init__(self) -> None:
        """Reject unsupported, nonfinite, or nonlocal native operations.

        Raises:
            ValueError: If the native gate or its support is invalid.
        """
        if (
            self.name not in {"rz", "sx", "cx"}
            or not isinstance(self.sites, tuple)
            or len(self.sites) != (2 if self.name == "cx" else 1)
            or any(type(site) is not int or site < 0 for site in self.sites)
            or (self.name == "cx" and abs(self.sites[0] - self.sites[1]) != 1)
            or not isinstance(self.angle, (float, int))
            or isinstance(self.angle, bool)
            or not math.isfinite(self.angle)
            or (self.name != "rz" and self.angle != 0)
        ):
            msg = "Expected finite RZ/SX or nearest-neighbor directed CX operations."
            raise ValueError(msg)


@dataclass(frozen=True)
class MadsResult:
    """The actual upstream optimizer's best evaluated point and scalar trace."""

    theta: NDArray[np.float64]
    loss: float
    receipt: dict[str, Any]


def load_nomad() -> ModuleType:
    """Import the pinned upstream implementation without a fallback optimizer.

    Returns:
        The installed official PyNomadBBO extension module.

    Raises:
        ImportError: If the optional pinned package is absent or has a wrong version.
    """
    try:
        version = importlib.metadata.version("PyNomadBBO")
    except importlib.metadata.PackageNotFoundError as error:
        msg = "NACL requires the optional official PyNomadBBO==4.6.0 package; see NACL_PROVENANCE.md."
        raise ImportError(msg) from error
    if version != NOMAD_VERSION:
        msg = f"NACL requires PyNomadBBO=={NOMAD_VERSION}, found {version}."
        raise ImportError(msg)
    return importlib.import_module("PyNomad")


def native_template(gates: tuple[NativeGate, ...], num_qubits: int) -> tuple[QuantumCircuit, NDArray[np.float64]]:
    """Parameterize every RZ independently, preserving all native operations.

    Args:
        gates: Native gates in execution order.
        num_qubits: Register size from one through six.

    Returns:
        A native template and angles in Qiskit's lexicographic parameter order.

    Raises:
        ValueError: If gates fall outside the register.
    """
    if type(num_qubits) is not int or not 1 <= num_qubits <= 6 or any(max(gate.sites) >= num_qubits for gate in gates):
        msg = "Native gates must fit a one- to six-qubit register."
        raise ValueError(msg)
    circuit = QuantumCircuit(num_qubits)
    angles = []
    for gate in gates:
        if gate.name == "rz":
            circuit.rz(Parameter(f"rz_{len(angles):06d}"), gate.sites[0])
            angles.append(gate.angle)
        elif gate.name == "sx":
            circuit.sx(gate.sites[0])
        else:
            circuit.cx(gate.sites[0], gate.sites[1])
    return circuit, np.asarray(angles, dtype=np.float64)


def resolve_angles(gates: tuple[NativeGate, ...], theta: ArrayLike) -> tuple[NativeGate, ...]:
    """Replace RZ angles without modifying gate structure or periodic wrapping.

    Args:
        gates: Original native sequence.
        theta: Finite real RZ parameters in execution order.

    Returns:
        Resolved native gate sequence.

    Raises:
        ValueError: If parameters are nonfinite or have the wrong shape.
    """
    values = np.asarray(theta)
    if (
        values.shape != (sum(gate.name == "rz" for gate in gates),)
        or not np.isrealobj(values)
        or values.dtype.kind not in "fiu"
        or not np.all(np.isfinite(values))
    ):
        msg = "Expected one finite real parameter per RZ gate."
        raise ValueError(msg)
    iterator = iter(values)
    return tuple(NativeGate("rz", gate.sites, float(next(iterator))) if gate.name == "rz" else gate for gate in gates)


def commute(first: NativeGate, second: NativeGate) -> bool:
    """Recognize a conservative set of exact native-gate commutation rules.

    Args:
        first: First native operation.
        second: Second native operation.

    Returns:
        Whether an explicitly checked rule permits exchanging the operations.
    """
    if set(first.sites).isdisjoint(second.sites):
        return True
    if first.name == second.name:
        if first.name != "cx":
            return first.sites == second.sites
        return first.sites[0] == second.sites[0] or first.sites[1] == second.sites[1]
    if first.name == "cx":
        first, second = second, first
    return second.name == "cx" and (
        (first.name == "rz" and first.sites[0] == second.sites[0])
        or (first.name == "sx" and first.sites[0] == second.sites[1])
    )


def simplify_gates(gates: tuple[NativeGate, ...]) -> tuple[NativeGate, ...]:
    """Apply only exact identities/commutation, never sensitivity-based pruning.

    Combine commuting RZs, cancel matching commuting CX pairs, and cancel groups
    of four commuting SXs. RZ angles equal to zero modulo 2 pi are removed exactly.
    Global phases are immaterial for preparation; noisy fidelity is reevaluated.

    Args:
        gates: Resolved native operations.

    Returns:
        An ideally equivalent native sequence, up to a global phase.
    """
    result: list[NativeGate] = []
    for original in gates:
        angle = math.remainder(original.angle, 2 * math.pi) if original.name == "rz" else 0.0
        gate = NativeGate(original.name, original.sites, angle)
        if gate.name == "rz" and gate.angle == 0:
            continue
        matches = []
        for index in range(len(result) - 1, -1, -1):
            previous = result[index]
            if previous.name == gate.name and previous.sites == gate.sites:
                matches.append(index)
                if gate.name != "sx" or len(matches) == 3:
                    break
            if not commute(gate, previous):
                break
        if matches and gate.name == "rz":
            index = matches[0]
            merged = math.remainder(result[index].angle + gate.angle, 2 * math.pi)
            result.pop(index)
            if merged != 0:
                result.insert(index, NativeGate("rz", gate.sites, merged))
        elif matches and gate.name == "cx":
            result.pop(matches[0])
        elif len(matches) == 3 and gate.name == "sx":
            for index in matches:
                result.pop(index)
        else:
            result.append(gate)
    return tuple(result)


def identity_block(num_qubits: int, rng: np.random.Generator, *, two_qubit: bool) -> tuple[NativeGate, ...]:
    """Sample a variationally extensible native resolution of the identity.

    One-qubit blocks are RZ(a), SX, RZ(pi), SX, RZ(pi-a), equal to -i I.
    Two-qubit blocks are CX, RZ(0) on its target, CX, initially I. All RZ angles
    become independent variables during OrthoMADS. The particular blocks are a
    disclosed reproduction choice because the NACL paper does not specify them.

    Args:
        num_qubits: Register size.
        rng: Caller-owned seeded generator.
        two_qubit: Whether to use a nearest-neighbor two-qubit block.

    Returns:
        Native operations whose product is identity up to global phase.

    Raises:
        ValueError: If the requested support does not fit the register.
    """
    if type(num_qubits) is not int or not 1 <= num_qubits <= 6 or (two_qubit and num_qubits < 2):
        msg = "Identity block support must fit a one- to six-qubit register."
        raise ValueError(msg)
    if two_qubit:
        low = int(rng.integers(num_qubits - 1))
        pair = (low, low + 1) if rng.integers(2) else (low + 1, low)
        return (NativeGate("cx", pair), NativeGate("rz", (pair[1],)), NativeGate("cx", pair))
    qubit = int(rng.integers(num_qubits))
    angle = float(rng.uniform(-math.pi, math.pi))
    return (
        NativeGate("rz", (qubit,), angle),
        NativeGate("sx", (qubit,)),
        NativeGate("rz", (qubit,), math.pi),
        NativeGate("sx", (qubit,)),
        NativeGate("rz", (qubit,), math.pi - angle),
    )


def propose_structure(
    gates: tuple[NativeGate, ...], num_qubits: int, rng: np.random.Generator, *, max_native_gates: int
) -> tuple[tuple[NativeGate, ...], dict[str, Any]]:
    """Uniformly select an available insertion, removal, or adjacent shuffle.

    Args:
        gates: Current resolved native sequence.
        num_qubits: Register size.
        rng: Caller-owned generator.
        max_native_gates: Explicit search resource cap, not a fidelity cutoff.

    Returns:
        Proposed sequence and the sampled structural move receipt.

    Raises:
        ValueError: If the cap is invalid or smaller than the existing sequence.
    """
    if type(max_native_gates) is not int or max_native_gates < len(gates):
        msg = "max_native_gates must be an integer at least the current gate count."
        raise ValueError(msg)
    actions = []
    if len(gates) + 5 <= max_native_gates:
        actions.append("insert_1q_identity")
    if num_qubits > 1 and len(gates) + 3 <= max_native_gates:
        actions.append("insert_2q_identity")
    if gates:
        actions.append("remove_gate")
    if len(gates) > 1:
        actions.append("shuffle_adjacent")
    if not actions:
        return gates, {"move": "no_available_move"}
    action = str(rng.choice(actions))
    proposal = list(gates)
    positions = len(gates) + 1 if action.startswith("insert") else len(gates)
    index = int(rng.integers(positions - (1 if action == "shuffle_adjacent" else 0)))
    if action.startswith("insert"):
        proposal[index:index] = identity_block(num_qubits, rng, two_qubit=action == "insert_2q_identity")
    elif action == "remove_gate":
        proposal.pop(index)
    else:
        proposal[index], proposal[index + 1] = proposal[index + 1], proposal[index]
    return tuple(proposal), {"move": action, "index": index, "available_moves": actions}


def anneal_accept(current_loss: float, proposed_loss: float, temperature: float, uniform: float) -> bool:
    """Apply the published exponential acceptance rule without underflow.

    Args:
        current_loss: Current finite infidelity.
        proposed_loss: Proposed finite infidelity.
        temperature: Nonnegative annealing temperature; zero is greedy.
        uniform: Caller-sampled value in [0, 1).

    Returns:
        Whether to accept the proposed circuit.

    Raises:
        ValueError: If arguments are nonfinite or outside their allowed range.
    """
    if (
        not all(math.isfinite(value) for value in (current_loss, proposed_loss, temperature, uniform))
        or temperature < 0
        or not 0 <= uniform < 1
    ):
        msg = "Annealing requires finite losses, nonnegative temperature and a uniform sample in [0, 1)."
        raise ValueError(msg)
    change = proposed_loss - current_loss
    return change <= 0 or (temperature > 0 and (uniform == 0 or math.log(uniform) < -change / temperature))


def ortho_mads(
    objective: Callable[[NDArray[np.float64]], float],
    initial: ArrayLike,
    *,
    evaluations: int,
    seed: int,
    initial_frame_size: float = 0.5,
    min_frame_size: float = 1e-6,
) -> MadsResult:
    """Optimize with official NOMAD's original OrthoMADS poll directions.

    No gradient, bounds, surrogate, quadratic-model, Nelder-Mead, or speculative
    search is substituted. The objective's actual evaluations are recorded and
    its best evaluated parameters are returned, including under budget stopping.

    Args:
        objective: Deterministic finite scalar loss.
        initial: Nonempty finite real initial parameters.
        evaluations: Positive blackbox evaluation budget including the start.
        seed: Nonnegative signed 32-bit seed passed to NOMAD's native parser.
            Initialization costs O(seed); the NACL driver samples 16-bit seeds.
        initial_frame_size: Initial mesh frame in radians.
        min_frame_size: Positive stopping frame, smaller than the initial frame.

    Returns:
        Best parameters, loss, upstream stop information and evaluation trace.

    Raises:
        ValueError: If controls or objective values are invalid.
        RuntimeError: If NOMAD reports a failed run or does not evaluate a point.
        ImportError: If the optional pinned optimizer is unavailable.
    """
    theta = np.asarray(initial)
    if (
        theta.ndim != 1
        or theta.size == 0
        or not np.isrealobj(theta)
        or theta.dtype.kind not in "fiu"
        or not np.all(np.isfinite(theta))
        or type(evaluations) is not int
        or evaluations < 1
        or type(seed) is not int
        or not 0 <= seed <= 2**31 - 1
        or not 0 < min_frame_size < initial_frame_size < math.inf
    ):
        msg = "OrthoMADS needs finite parameters, positive budget/frame sizes and a nonnegative signed 32-bit seed."
        raise ValueError(msg)
    nomad = load_nomad()
    parameters = [
        "BB_OUTPUT_TYPE OBJ",
        f"MAX_BB_EVAL {evaluations}",
        "DISPLAY_DEGREE 0",
        "DIRECTION_TYPE ORTHO 2N",
        "DIRECTION_TYPE_SECONDARY_POLL ORTHO 2N",
        "QUAD_MODEL_SEARCH false",
        "SGTELIB_MODEL_SEARCH false",
        "NM_SEARCH false",
        "SPECULATIVE_SEARCH false",
        "LH_SEARCH 0 0",
        "EVAL_QUEUE_SORT DIR_LAST_SUCCESS",
        f"INITIAL_FRAME_SIZE * {initial_frame_size:.17g}",
        f"MIN_FRAME_SIZE * {min_frame_size:.17g}",
        f"SEED {seed}",
    ]
    records: list[dict[str, float | int]] = []
    best = theta.astype(np.float64, copy=True)
    best_loss = math.inf
    errors: list[Exception] = []

    def evaluate(point: NomadPoint) -> int:
        """Evaluate a NOMAD callback and retain failures outside its C++ boundary.

        Args:
            point: Upstream point exposing get_coord and setBBO.

        Returns:
            One for a valid blackbox evaluation, zero for a captured exception.
        """
        nonlocal best, best_loss
        try:
            values = np.asarray([point.get_coord(index) for index in range(theta.size)], dtype=np.float64)
            loss = float(objective(values))
        except Exception as error:  # The native extension otherwise suppresses callback exceptions.
            errors.append(error)
            return 0
        if not math.isfinite(loss):
            errors.append(ValueError("OrthoMADS objective returned a nonfinite loss."))
            return 0
        point.setBBO(f"{loss:.17g}".encode("ascii"))
        records.append({"evaluation": len(records) + 1, "loss": loss})
        if loss < best_loss:
            best, best_loss = values.copy(), loss
        return 1

    # NOMAD reuses native global RNG state; a repeated SEED parameter alone does
    # not reset it. Its public reset API is required for deterministic replay.
    nomad.setSeed(seed)
    nomad.resetRandomNumberGenerator()
    result = nomad.optimize(evaluate, theta.astype(float).tolist(), [], [], parameters)
    if errors:
        raise errors[0]
    # PyNomad 4.6.0 can report negative flags on mesh convergence despite returning feasible
    # incumbents for this unconstrained objective. Require those incumbents and
    # retain the unmodified flag rather than pretending to infer convergence.
    frame_converged = "Min frame size reached" in str(result.get("stop_reason", ""))
    if not records or not result.get("x_best_feas") or (result["run_flag"] < 0 and not frame_converged):
        msg = f"NOMAD failed: {result.get('stop_reason', 'no successful evaluations')}"
        raise RuntimeError(msg)
    return MadsResult(
        best,
        best_loss,
        {
            "optimizer": "official_PyNomadBBO_OrthoMADS",
            "version": NOMAD_VERSION,
            "parameters": parameters,
            "seed": seed,
            "requested_evaluations": evaluations,
            "evaluations": len(records),
            "upstream_evaluations": int(result["nb_evals"]),
            "run_flag": int(result["run_flag"]),
            "feasible_incumbent_with_negative_flag": result["run_flag"] < 0,
            "stop_reason": str(result["stop_reason"]),
            "records": records,
        },
    )


def optimize_structure(
    gates: tuple[NativeGate, ...],
    target: NDArray[np.complex128],
    hardware: str | None,
    *,
    evaluations: int,
    seed: int,
    initial_frame_size: float,
    min_frame_size: float,
) -> tuple[tuple[NativeGate, ...], float, dict[str, Any]]:
    """Evaluate exact noisy infidelity during all continuous NACL updates.

    Args:
        gates: Structure, initialized from the previous accepted circuit.
        target: Normalized complex target.
        hardware: IBM/Google native noise model, or None for an ideal oracle.
        evaluations: OrthoMADS evaluation budget.
        seed: Optimizer seed.
        initial_frame_size: Initial mesh frame in radians.
        min_frame_size: Minimum mesh frame in radians.

    Returns:
        Optimized gates, best infidelity and scalar optimization history.
    """
    from experiments.state_preparation_competitors.noisy_core import DenseObjective  # noqa: PLC0415

    template, initial = native_template(gates, target.size.bit_length() - 1)
    objective = DenseObjective(template, target, hardware, compile_native=False)
    if not initial.size:
        loss = 1 - objective.fidelity(initial)
        return (
            gates,
            loss,
            {
                "optimizer": "no_parameters_exact_evaluation",
                "evaluations": 1,
                "seed": seed,
                "records": [{"evaluation": 1, "loss": loss}],
            },
        )
    result = ortho_mads(
        lambda theta: 1 - objective.fidelity(theta),
        initial,
        evaluations=evaluations,
        seed=seed,
        initial_frame_size=initial_frame_size,
        min_frame_size=min_frame_size,
    )
    return resolve_angles(gates, result.theta), result.loss, result.receipt


def nacl_candidates(
    target: ArrayLike,
    *,
    hardware: str,
    seed: int = 300,
    search_steps: tuple[int, ...] = (0, 10, 50),
    inner_evaluations: int = 100,
    restarts: int = 1,
    max_native_gates: int = 300,
    initial_temperature: float = 0.01,
    cooling: float = 0.95,
    initial_frame_size: float = 0.5,
    min_frame_size: float = 1e-6,
    simplify_every: int = 5,
) -> list[Candidate]:
    """Search native circuits under noise, saving complete prefix-budget winners.

    Independent empty-start chains receive the same number of structural steps
    at each checkpoint. All searched restarts and rejected moves count toward
    cumulative construction cost. Histories are stored only in the checkpoint
    interval where they occurred; previous_checkpoint links earlier intervals.

    Args:
        target: Normalized complex vector for one through six qubits.
        hardware: Existing IBM or Google synthetic native-noise profile.
        seed: Root seed for independent structural-search/restart generators.
        search_steps: Strictly increasing nonnegative step checkpoints per restart.
        inner_evaluations: Maximum exact noisy loss evaluations per inner solve.
        restarts: Number of independent structural-search chains.
        max_native_gates: Explicit finite search resource cap.
        initial_temperature: Initial simulated-annealing temperature in loss units.
        cooling: Multiplicative temperature decay per structural proposal.
        initial_frame_size: OrthoMADS initial frame size in radians.
        min_frame_size: OrthoMADS minimum frame size in radians.
        simplify_every: Period for exact cancellation/commutation simplification;
            zero disables this optional published acceleration.

    Returns:
        Best-noisy-fidelity circuit at each checkpoint, with reproducibility data.

    Raises:
        ValueError: If target, profile or controls are invalid.
        ImportError: If pinned official NOMAD is not available.
    """
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    vector = validate_target(target)
    if (
        hardware not in {"IBM", "Google"}
        or type(seed) is not int
        or not 0 <= seed <= 2**32 - 1
        or not isinstance(search_steps, tuple)
        or not search_steps
        or any(type(step) is not int or step < 0 for step in search_steps)
        or any(right <= left for left, right in itertools.pairwise(search_steps))
        or any(type(value) is not int or value < 1 for value in (inner_evaluations, restarts))
        or type(max_native_gates) is not int
        or max_native_gates < 5
        or type(simplify_every) is not int
        or simplify_every < 0
        or not 0 <= initial_temperature < math.inf
        or not 0 < cooling <= 1
        or not 0 < min_frame_size < initial_frame_size < math.inf
    ):
        msg = "Invalid NACL profile, seed, search checkpoints, budgets or annealing/mesh controls."
        raise ValueError(msg)
    load_nomad()
    num_qubits = vector.size.bit_length() - 1
    streams = np.random.SeedSequence(seed).spawn(restarts)
    generators = [np.random.default_rng(stream) for stream in streams]
    initial_loss = 1 - float(abs(vector[0]) ** 2)
    chains: list[tuple[tuple[NativeGate, ...], float]] = [((), initial_loss) for _ in range(restarts)]
    best_gates: tuple[NativeGate, ...] = ()
    best_loss = initial_loss
    best_at = {"restart": 0, "step": 0}
    candidates = []
    previous_step = 0
    previous_cpu = previous_wall = 0.0
    total_evaluations = 1
    for checkpoint in search_steps:
        records = []
        interval_evaluations = 0
        for step in range(previous_step + 1, checkpoint + 1):
            temperature = initial_temperature * cooling ** (step - 1)
            for restart, rng in enumerate(generators):
                current, current_loss = chains[restart]
                proposed, move = propose_structure(current, num_qubits, rng, max_native_gates=max_native_gates)
                optimizer_seed = int(rng.integers(INNER_SEED_RANGE))
                optimized, proposed_loss, receipt = optimize_structure(
                    proposed,
                    vector,
                    hardware,
                    evaluations=inner_evaluations,
                    seed=optimizer_seed,
                    initial_frame_size=initial_frame_size,
                    min_frame_size=min_frame_size,
                )
                interval_evaluations += receipt["evaluations"]
                inner_runs = [receipt]
                simplified = simplify_gates(optimized) if simplify_every and step % simplify_every == 0 else optimized
                simplification_applied = simplified != optimized
                if simplification_applied:
                    # Evaluate the changed noisy channel, then optimize its remaining angles.
                    # Keep a superior pre-simplification circuit as a searched candidate too.
                    if proposed_loss < best_loss:
                        best_gates, best_loss = optimized, proposed_loss
                        best_at = {"restart": restart, "step": step}
                    optimized, proposed_loss, simplified_receipt = optimize_structure(
                        simplified,
                        vector,
                        hardware,
                        evaluations=inner_evaluations,
                        seed=int(rng.integers(INNER_SEED_RANGE)),
                        initial_frame_size=initial_frame_size,
                        min_frame_size=min_frame_size,
                    )
                    interval_evaluations += simplified_receipt["evaluations"]
                    inner_runs.append(simplified_receipt)
                uniform = float(rng.random())
                accepted = anneal_accept(current_loss, proposed_loss, temperature, uniform)
                if accepted:
                    chains[restart] = (optimized, proposed_loss)
                if proposed_loss < best_loss:
                    best_gates, best_loss = optimized, proposed_loss
                    best_at = {"restart": restart, "step": step}
                records.append({
                    "step": step,
                    "restart": restart,
                    **move,
                    "temperature": temperature,
                    "acceptance_uniform": uniform,
                    "accepted": accepted,
                    "proposed_fidelity": 1 - proposed_loss,
                    "current_fidelity": 1 - chains[restart][1],
                    "best_fidelity": 1 - best_loss,
                    "proposed_native_gates": len(optimized),
                    "best_native_gates": len(best_gates),
                    "simplification_applied": simplification_applied,
                    "inner_runs": inner_runs,
                })
        total_evaluations += interval_evaluations
        template, best_theta = native_template(best_gates, num_qubits)
        circuit = template.assign_parameters(best_theta)
        ideal_fidelity = float(abs(np.vdot(vector, Statevector.from_instruction(circuit).data)) ** 2)
        cumulative_cpu, cumulative_wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
        candidate_id = f"steps_{checkpoint:05d}"
        candidates.append(
            Candidate(
                circuit,
                {
                    "method": "nacl",
                    "algorithm": ALGORITHM,
                    "paper_doi": PAPER_DOI,
                    "implementation": "local_paper_based_reconstruction_not_author_code",
                    "candidate_id": candidate_id,
                    "status": "complete",
                    "deterministic": False,
                    "seed": seed,
                    "seed_used": True,
                    "hardware": hardware,
                    "search_steps_per_restart": checkpoint,
                    "restarts": restarts,
                    "restart_spawn_keys": [list(stream.spawn_key) for stream in streams],
                    "inner_seed_distribution": "uniform_integer_0_through_65535_from_restart_generator",
                    "inner_evaluations": inner_evaluations,
                    "max_native_gates": max_native_gates,
                    "initial_temperature": initial_temperature,
                    "cooling": cooling,
                    "initial_frame_size": initial_frame_size,
                    "min_frame_size": min_frame_size,
                    "simplify_every": simplify_every,
                    "proposal_distribution": "uniform_over_available_insert1_insert2_remove_adjacent_swap",
                    "initialization": "empty_circuit_independent_seeded_restarts",
                    "budget_reuse": "shared_prefix_checkpoints_across_all_restarts",
                    "previous_checkpoint": candidates[-1].metadata["candidate_id"] if candidates else None,
                    "history_scope": "only_steps_since_previous_checkpoint_no_duplicate_history",
                    "records": records,
                    "best_parameters": best_theta.tolist(),
                    "best_at": best_at.copy(),
                    "training_fidelity": 1 - best_loss,
                    "ideal_fidelity": ideal_fidelity,
                    "native_gates": len(best_gates),
                    "preserve_native_schedule": True,
                    "native_two_qubit_gates": sum(gate.name == "cx" for gate in best_gates),
                    "training_noise_placement": "every_native_RZ_SX_CX_including_RZ_no_idle_noise",
                    "training_objective": "exact_density_matrix_noisy_infidelity",
                    "noise_fine_tuning": True,
                    "optimizer": "official_PyNomadBBO_OrthoMADS",
                    "optimizer_version": NOMAD_VERSION,
                    "interval_objective_evaluations": interval_evaluations + (1 if not candidates else 0),
                    "cumulative_objective_evaluations": total_evaluations,
                    "cpu_seconds": cumulative_cpu - previous_cpu,
                    "cumulative_cpu_seconds": cumulative_cpu,
                    "wall_seconds": cumulative_wall - previous_wall,
                    "cumulative_wall_seconds": cumulative_wall,
                    "cpu_scope": "all_restarts_rejected_moves_inner_evaluations_and_bookkeeping",
                },
            )
        )
        previous_step, previous_cpu, previous_wall = checkpoint, cumulative_cpu, cumulative_wall
    return candidates
