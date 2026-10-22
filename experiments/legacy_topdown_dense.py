# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reproduce the archived q8 impact-pruning example with vectors and matrices.

The archived arm transcribes 6bdc86b:experiments/find_hero_state.py. Its
parameter-shift pruning derivative is replaced by the equivalent exact adjoint.
The literal arm separately tests layerwise hybrid training and a 0.01 cutoff.
Qiskit gates and channels provide an independent density-matrix evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit.library import RXGate, RXXGate, RYGate, RYYGate, RZGate, RZZGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, SuperOp, average_gate_fidelity
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    ParameterizedCircuit,
    StatevectorBackend,
    TJMNoiseInstruction,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqt.yaqs.optimization.backends import PreparationBackend
    from mqt.yaqs.optimization.gate_noise import GateNoiseContext

ARCHIVE = "6bdc86baeb7daa03756f153b49fda0d7606d04d5"
PRIMITIVES = {"rx": RXGate, "ry": RYGate, "rz": RZGate, "rxx": RXXGate, "ryy": RYYGate, "rzz": RZZGate}


def target_state(seed: int, family: str = "tfim", qubits: int = 8) -> np.ndarray:
    """Return the archived TFIM target or a seeded complex Haar vector.

    TFIM Kronecker factors intentionally retain the archive's ordering. These
    are ground states; the archive does not define dissipative steady states.
    """
    rng = np.random.RandomState(seed)
    dimension = 2**qubits
    if family == "haar":
        vector = rng.randn(dimension) + 1j * rng.randn(dimension)
        return vector / np.linalg.norm(vector)
    couplings = rng.uniform(0.8, 1.2, qubits - 1)
    fields = rng.uniform(0.8, 1.2, qubits)
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    for site, coupling in enumerate(couplings):
        label = "I" * site + "ZZ" + "I" * (qubits - site - 2)
        hamiltonian -= coupling * Pauli(label).to_matrix()
    for site, field in enumerate(fields):
        label = "I" * site + "X" + "I" * (qubits - site - 1)
        hamiltonian -= field * Pauli(label).to_matrix()
    return np.linalg.eigh(hamiltonian)[1][:, 0]


@cache
def local_model(hardware: str, sites: tuple[int, ...]) -> NoiseModel:
    """Transcribe the archived local single-site and correlated Pauli rates."""
    one, two, correlated = (3e-4, 3e-3, ("xx", "zz")) if hardware == "IBM" else (6e-4, 5e-3, ("xx", "yy", "zz"))
    processes = [
        {"name": "pauli_" + axis, "sites": [site], "strength": one / 3} for site in sites for axis in ("x", "y", "z")
    ]
    if len(sites) == 2:
        processes.extend(
            {"name": "crosstalk_" + axes, "sites": list(sites), "strength": two / len(correlated)}
            for axes in correlated
        )
    return NoiseModel(processes)


class LegacyNoise:
    """Expose the archived local Pauli law to the dense trajectory sampler."""

    def __init__(self, hardware: str = "IBM") -> None:
        """Select the archived rate profile."""
        self.hardware = hardware

    def __call__(self, context: GateNoiseContext, rng: np.random.Generator, /) -> TJMNoiseInstruction:
        """Return a deterministic instruction; the backend owns random draws."""
        del rng
        return TJMNoiseInstruction(local_model(self.hardware, context.sites))


@cache
def noise_channel(hardware: str, arity: int, law: str = "tjm") -> SuperOp:
    """Build a Qiskit channel independently from literal Pauli probabilities.

    ``tjm`` exactly averages the archive's at-most-one-jump gate window.
    ``lindblad`` exponentiates its generator, including repeated events.
    """
    one, two = (3e-4, 3e-3) if hardware == "IBM" else (6e-4, 5e-3)
    axes = ("X", "Y", "Z")
    labels = list(axes) if arity == 1 else [axis + "I" for axis in axes] + ["I" + axis for axis in axes]
    rates = [one / 3] * len(labels)
    if arity == 2:
        correlated = ("XX", "ZZ") if hardware == "IBM" else ("XX", "YY", "ZZ")
        labels.extend(correlated)
        rates.extend([two / len(correlated)] * len(correlated))
    identity = SuperOp(Operator(np.eye(2**arity))).data
    generator = sum(
        (rate * (SuperOp(Pauli(label)).data - identity) for label, rate in zip(labels, rates, strict=True)),
        start=np.zeros_like(identity),
    )
    total = sum(rates)
    matrix = expm(generator) if law == "lindblad" else identity + (-np.expm1(-total) / total) * generator
    return SuperOp(matrix)


def matrix_score(circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray, hardware: str, law: str) -> dict:
    """Independently propagate a density matrix through Qiskit gates/channels."""
    rho = DensityMatrix.from_label("0" * circuit.num_qubits)
    for gate in circuit.gates:
        angle = gate.angle_offset + (gate.angle_scale * theta[gate.param_index] if gate.param_index is not None else 0)
        rho = rho.evolve(PRIMITIVES[gate.name](angle), qargs=list(gate.sites))
        if law != "noiseless" and gate.noise_enabled:
            rho = rho.evolve(noise_channel(hardware, len(gate.sites), law), qargs=list(gate.sites))
    data = rho.data
    result = {
        "fidelity": float(np.vdot(target, data @ target).real),
        "trace_error": float(abs(np.trace(data) - 1)),
        "hermiticity_error": float(np.max(np.abs(data - data.conj().T))),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(data).min()),
    }
    assert result["trace_error"] < 1e-10
    assert result["hermiticity_error"] < 1e-10
    assert result["minimum_eigenvalue"] > -1e-10
    return result


def prune(circuit: ParameterizedCircuit, theta: np.ndarray, impact: np.ndarray, *, keep: int | None = None) -> tuple:
    """Physically delete parameters/gates, using rank or the literal 0.01 cutoff."""
    indices = np.flatnonzero(impact >= 0.01) if keep is None else np.argsort(-impact, kind="stable")[:keep]
    indices = np.sort(indices)
    mapping = {int(old): new for new, old in enumerate(indices)}
    gates = [
        replace(gate, param_index=mapping[gate.param_index]) if gate.param_index is not None else replace(gate)
        for gate in circuit.gates
        if gate.param_index is None or gate.param_index in mapping
    ]
    return ParameterizedCircuit(circuit.num_qubits, gates, len(indices)), theta[indices].copy()


def train(
    circuit: ParameterizedCircuit,
    theta: np.ndarray,
    target: np.ndarray | MPS,
    *,
    steps: int,
    rate: float,
    seed: int,
    decay: float = 0.0,
    noisy: bool = False,
    crn: bool = True,
    online: int = 0,
    stop: float = 2.0,
    hardware: str = "IBM",
    num_trajectories: int = 3,
    compress_no_jump: bool = False,
    record_noise_banks: bool = False,
    require_jump_coverage: bool = False,
    checkpoint: Callable[[int, np.ndarray, float], None] | None = None,
    engine: PreparationBackend | None = None,
) -> tuple[np.ndarray, dict]:
    """Run archived updates, optionally recording the initial state and every iterate.

    The checkpoint callback receives a private parameter copy and its fidelity.
    No selection or change to the update trajectory occurs inside this function.

    Raises:
        ValueError: If checkpoints are requested with changing training banks.
    """
    if checkpoint is not None and noisy and not crn:
        msg = "Iterate checkpoints require a fixed training bank for comparable fidelities."
        raise ValueError(msg)
    start = time.perf_counter()
    theta = theta.copy()
    if engine is None:
        if isinstance(target, MPS):
            msg = "MPS targets require an explicit preparation backend."
            raise ValueError(msg)
        engine = StatevectorBackend(circuit, target)
    options = KrotovTJMOptions(num_trajectories=num_trajectories, trajectory_update="cross", use_crn=crn)
    provider = LegacyNoise(hardware)
    banks = []

    def sample(counter: int) -> tuple:
        """Sample and inspect complete histories before any state propagation."""
        raw = engine.sample_noise_maps(theta, options, provider, iteration=counter)
        jumps = [sum(bool(m.operators) for m in row) for row in raw]
        if require_jump_coverage and any(g.noise_enabled for g in circuit.gates) and not any(jumps):
            msg = "The sampled bank has no jumps; stopped without rejecting/resampling it or changing its weights."
            raise RuntimeError(msg)
        retained, weights = engine.compress_no_jump_maps(raw) if compress_no_jump else (raw, None)
        counts = np.rint(weights * len(raw)).astype(int).tolist() if weights is not None else [1] * len(raw)
        bank = {
            "counter": counter,
            "sampled_paths": len(raw),
            "no_jump_paths": jumps.count(0),
            "jump_paths": sum(count > 0 for count in jumps),
            "raw_jump_counts": jumps,
            "retained_paths": len(retained),
            "multiplicities": counts,
            "weights": (np.asarray(counts) / len(raw)).tolist(),
        }
        if record_noise_banks:
            events = [
                [
                    {
                        "gate": k,
                        "operators": [
                            {"sites": list(sites), "real": matrix.real.tolist(), "imag": matrix.imag.tolist()}
                            for matrix, sites in noise_map.operators
                        ],
                    }
                    for k, noise_map in enumerate(row)
                    if noise_map.operators
                ]
                for row in raw
            ]
            bank["events"] = events
            bank["raw_sha256"] = hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()
            bank["first_three_sha256"] = hashlib.sha256(json.dumps(events[:3], sort_keys=True).encode()).hexdigest()
        banks.append(bank)
        return retained, weights

    maps, weights = sample(seed) if noisy and crn else (None, None)
    if checkpoint is not None:
        checkpoint(0, theta.copy(), float(engine.metrics(theta, maps, weights=weights)[1]))
    trace = []
    for iteration in range(1, steps + 1):
        if noisy and not crn:
            maps, weights = sample(seed + iteration)
        estimator = "cross" if noisy else "independent"
        phase_iteration = iteration if iteration <= online else iteration - online
        step = rate * np.exp(-decay * (phase_iteration - 1))
        if iteration <= online:
            theta, _ = engine.online_update(theta, step, maps, estimator=estimator, weights=weights)
        else:
            gradient, _, _ = engine.gradient(theta, maps, estimator=estimator, weights=weights)
            theta -= step * gradient
        fidelity = engine.metrics(theta, maps, weights=weights)[1]
        trace.append(fidelity)
        if checkpoint is not None:
            checkpoint(iteration, theta.copy(), float(fidelity))
        if fidelity >= stop:
            break
    return theta, {
        "steps": len(trace),
        "online_steps": min(online, len(trace)),
        "rate": rate,
        "decay": decay,
        "seed": seed,
        "crn": crn,
        "noisy": noisy,
        "hardware": hardware,
        "num_trajectories": num_trajectories if noisy else 1,
        "compress_no_jump": compress_no_jump,
        "noise_banks": banks,
        "fidelity_trace": trace,
        "last_map_jump_counts": [sum(bool(m.operators) for m in row) for row in maps] if maps else [],
        "wall_seconds": time.perf_counter() - start,
    }


def brickwall(depth: int) -> ParameterizedCircuit:
    """Return the archived eight-qubit BMPD primitive architecture."""
    return create_brickwall_matrix_product_disentangler_parameterized_circuit(8, depth, initial_single_qubit_layer=True)


def extend(theta: np.ndarray | None, count: int, seed: int, scale: float) -> np.ndarray:
    """Preserve archived parameter order and RandomState initialization."""
    previous = np.array([], dtype=float) if theta is None else theta
    rng = np.random.RandomState(seed)
    return np.concatenate((previous, rng.randn(count - len(previous)) * scale))


class Recorder:
    """Persist every selected circuit, parameters, traces, and exact scores."""

    def __init__(self, root: Path, target: np.ndarray, *, resume: bool = False) -> None:
        """Persist a target, allowing explicit resume only with an identical target."""
        self.root, self.target = root, target
        if resume and root.exists():
            np.testing.assert_array_equal(np.load(root / "target.npy"), target)
            return
        root.mkdir(parents=True, exist_ok=False)
        np.save(root / "target.npy", target)

    def save(self, name: str, circuit: ParameterizedCircuit, theta: np.ndarray, trace: dict | list) -> None:
        """Save a reviewable checkpoint and score under both archived profiles."""
        np.save(self.root / f"{name}.theta.npy", theta)
        record = {
            "circuit": {
                "num_qubits": circuit.num_qubits,
                "num_params": circuit.num_params,
                "gates": [asdict(g) for g in circuit.gates],
            },
            "training": trace,
            "gate_counts": {
                "one": sum(len(g.sites) == 1 for g in circuit.gates),
                "two": sum(len(g.sites) == 2 for g in circuit.gates),
            },
            "noiseless": matrix_score(circuit, theta, self.target, "IBM", "noiseless"),
            "noise": {
                hw: {law: matrix_score(circuit, theta, self.target, hw, law) for law in ("tjm", "lindblad")}
                for hw in ("IBM", "Google")
            },
        }
        dense = StatevectorBackend(circuit, self.target).metrics(theta)[1]
        record["vector_matrix_fidelity_error"] = abs(dense - record["noiseless"]["fidelity"])
        assert record["vector_matrix_fidelity_error"] < 1e-10
        (self.root / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            self.root.name,
            name,
            circuit.num_params,
            {"noiseless": dense, "IBM": record["noise"]["IBM"]["tjm"]["fidelity"]},
            flush=True,
        )


def archived_example(root: Path, seed: int = 300) -> None:
    """Reproduce all three archived arms and retain stage checkpoints."""
    target = target_state(seed)
    recorder = Recorder(root / f"archived_tfim_{seed}", target)
    circuit = brickwall(4)
    theta = extend(None, circuit.num_params, seed + 1, 0.1)
    theta, trace = train(circuit, theta, target, steps=300, rate=0.2, decay=0.01, seed=seed + 2, noisy=True)
    recorder.save("fixed_vqa", circuit, theta, trace)
    theta, traces = np.empty(0, dtype=float), []
    for depth in (1, 2):
        circuit = brickwall(depth)
        theta = extend(
            theta, circuit.num_params, seed + 100 if depth == 1 else seed + 200 + depth, 0.05 if depth == 1 else 0.001
        )
        theta, trace = train(circuit, theta, target, steps=100, rate=1.0, seed=seed + 300)
        traces.append(trace)
    recorder.save("layerwise_before_noise", circuit, theta, traces)
    theta, trace = train(circuit, theta, target, steps=100, rate=0.1, seed=seed + 400, noisy=True, crn=False)
    recorder.save("layerwise_final", circuit, theta, trace)
    theta, traces = np.empty(0, dtype=float), []
    for depth, rate in zip((4, 8, 12, 16), (5.0, 2.0, 0.5, 0.2), strict=True):
        circuit = brickwall(depth)
        theta = extend(
            theta, circuit.num_params, seed + 1000 if depth == 4 else seed + 2000 + depth, 0.05 if depth == 4 else 0.001
        )
        theta, trace = train(circuit, theta, target, steps=150, rate=rate, decay=0.01, seed=seed + 3000)
        traces.append(trace)
        print("archived pretrain", depth, trace["fidelity_trace"][-1], flush=True)
    assert theta is not None
    recorder.save("topdown_before_pruning", circuit, theta, traces)
    traces = []
    while circuit.num_params > 150:
        impact = np.abs(theta * StatevectorBackend(circuit, target).gradient(theta)[0])
        circuit, theta = prune(circuit, theta, impact, keep=max(150, circuit.num_params - 150))
        if circuit.num_params > 150:
            theta, trace = train(circuit, theta, target, steps=40, rate=0.2, seed=42)
            traces.append(trace)
    recorder.save("topdown_before_noise", circuit, theta, traces)
    theta, trace = train(circuit, theta, target, steps=200, rate=0.1, seed=seed + 5000, noisy=True)
    recorder.save("topdown_final", circuit, theta, trace)


def archived_baselines(root: Path, seed: int = 400) -> None:
    """Reproduce the separate baseline scripts that also label seed 400 State 4."""
    target = target_state(seed)
    recorder = Recorder(root / f"baseline_settings_tfim_{seed}", target)
    circuit = brickwall(4)
    theta = extend(None, circuit.num_params, 42, 0.1)
    theta, trace = train(circuit, theta, target, steps=300, rate=0.2, decay=0.01, seed=1, noisy=True)
    recorder.save("fixed_vqa", circuit, theta, trace)
    theta, traces = np.empty(0, dtype=float), []
    for depth in (1, 2):
        circuit = brickwall(depth)
        theta = extend(theta, circuit.num_params, 100 if depth == 1 else 200 + depth, 0.05 if depth == 1 else 0.001)
        theta, trace = train(circuit, theta, target, steps=100, rate=1.0, seed=2)
        traces.append(trace)
    recorder.save("layerwise_before_noise", circuit, theta, traces)
    theta, trace = train(circuit, theta, target, steps=100, rate=0.1, seed=800, noisy=True, crn=False)
    recorder.save("layerwise_final", circuit, theta, trace)


def literal_example(root: Path, family: str, seed: int = 300) -> None:
    """Test the user's layer-one growth, hybrid, and absolute impact cutoff."""
    target = target_state(seed, family)
    recorder = Recorder(root / f"literal_{family}_{seed}", target)
    theta, traces = np.empty(0, dtype=float), []
    for depth in range(1, 17):
        circuit = brickwall(depth)
        theta = extend(theta, circuit.num_params, seed + 1000 + depth, 0.05 if depth == 1 else 0.001)
        theta, trace = train(circuit, theta, target, steps=200, online=100, rate=0.7, seed=seed + 3000, stop=0.9)
        traces.append(trace)
        print("literal", family, "depth", depth, trace["fidelity_trace"][-1], flush=True)
        if trace["fidelity_trace"][-1] >= 0.9:
            break
    recorder.save("before_pruning", circuit, theta, traces)
    if trace["fidelity_trace"][-1] < 0.9:
        return
    impact = np.abs(theta * StatevectorBackend(circuit, target).gradient(theta)[0])
    np.save(recorder.root / "impact.npy", impact)
    pruned, values = prune(circuit, theta, impact)
    recorder.save("cutoff_before_noise", pruned, values, {"cutoff": 0.01})
    for hardware in ("IBM", "Google"):
        for name, current, initial in (("cutoff", pruned, values), ("unpruned_control", circuit, theta)):
            final, trace = train(
                current,
                initial,
                target,
                steps=200,
                online=100,
                rate=0.2,
                seed=seed + 5000,
                noisy=True,
                hardware=hardware,
            )
            recorder.save(name + "_" + hardware, current, final, trace)


def main() -> None:
    """Run an isolated, non-overwriting historical and literal verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("archived", "baselines", "literal", "all"), default="all")
    parser.add_argument("--seed", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    manifest = {
        "archive": ARCHIVE,
        "driver_sha256": hashlib.sha256(source).hexdigest(),
        "numpy": np.__version__,
        "arm": args.arm,
        "seed": args.seed,
        "literal_settings": (
            "8 qubits; seeds 300; depths 1..16; 200 updates/depth, first 100 online; rate 0.7; stop at 0.9; "
            "abs(theta*dF/dtheta)>=0.01; 200 noisy hybrid steps, rate 0.2, "
            "three frozen cross trajectories; IBM and Google"
        ),
        "scope": (
            "Archived rates are Lindblad coefficients, not calibrated measured gate infidelities. "
            "Gates count scalar logical rotations, not transpiled native operations. "
            "Archived layerwise growth is not operator-pool ADAPT-VQE."
        ),
        "channel_infidelities": {
            hw: {str(k): float(1 - average_gate_fidelity(noise_channel(hw, k))) for k in (1, 2)}
            for hw in ("IBM", "Google")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "driver_source.txt").write_bytes(source)
    git = shutil.which("git")
    assert git is not None
    for name in (
        "find_hero_state.py",
        "impact_pruning_benchmark.py",
        "baselines_benchmark.py",
        "adapt_noisy_test.py",
        "results/rigorous_benchmark_5states.csv",
    ):
        # The executable is resolved above; revision and paths are literal repository evidence.
        result = subprocess.run(  # noqa: S603
            [git, "show", ARCHIVE + ":experiments/" + name], check=True, capture_output=True
        )
        (args.output / (Path(name).name + ".txt")).write_bytes(result.stdout)
    with threadpool_limits(limits=1):
        if args.arm in {"all", "archived"}:
            archived_example(args.output, args.seed)
        if args.arm in {"all", "baselines"}:
            archived_baselines(args.output, args.seed)
        if args.arm in {"all", "literal"}:
            for family in ("tfim", "haar"):
                literal_example(args.output, family, args.seed)


if __name__ == "__main__":
    main()
