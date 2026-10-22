# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Select the best of random product states and zero before unchanged training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import load_target, load_target_collection
from experiments import topdown_random_mps_initialization as base
from mqt.yaqs.optimization import MPSBackend, ParameterizedCircuit, StatevectorBackend
from mqt.yaqs.optimization.mps_backend import mps_overlap

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.mps import MPS

VERSION = "topdown.product_screening.v1"
ARMS = ("near_zero", "random_product", "screened_product")
DEFAULT_ARMS = ("near_zero", "screened_product")
BUDGET_FIELDS = (
    "max_depth",
    "construction_steps",
    "construction_online",
    "noisy_steps",
    "noisy_online",
    "training_paths",
    "validation_paths",
    "final_paths_per_batch",
    "final_batches",
)
runner = base.capacity.topdown_mps
support = base.support


def sources() -> dict[str, str]:
    """Fingerprint selection and all reused numerical updates."""
    return {**base.sources(), str(Path(__file__).relative_to(runner.ROOT)): support.digest(Path(__file__))}


def screen_products(target: NDArray, count: int = 1000, seed: int = 300, batch_size: int = 64) -> tuple[dict, dict]:
    """Score zero and independent Haar single-qubit product states in batches.

    Returns:
        Selection metadata and angles/fidelities, with zero at candidate index zero.

    Raises:
        ValueError: If the normalized target, counts or random seed are invalid.
    """
    target, qubits = base.validate_target(target)
    if (
        type(count) is not int
        or count < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        msg = "Require positive integer sample/batch counts and a valid seed."
        raise ValueError(msg)
    # This stream is separate from both legacy angles and the unscreened control.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 730002]))
    draw = rng.random((count, 2, qubits))
    probability, phase = draw[:, 0], 2 * np.pi * draw[:, 1]
    fidelity = np.empty(count + 1)
    fidelity[0] = abs(target[0]) ** 2
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        vectors = np.ones((end - start, 1), dtype=np.complex128)
        for site in range(qubits):
            local = np.column_stack((
                np.sqrt(1 - probability[start:end, site]),
                np.exp(1j * phase[start:end, site]) * np.sqrt(probability[start:end, site]),
            ))
            vectors = (local[:, :, None] * vectors[:, None, :]).reshape(end - start, -1)
        fidelity[start + 1 : end + 1] = np.abs(vectors @ target.conj()) ** 2
    polar = np.vstack((np.zeros(qubits), np.arccos(1 - 2 * probability)))
    azimuth = np.vstack((np.zeros(qubits), phase))
    index = int(np.argmax(fidelity))
    return {
        "count": count,
        "candidate_count": count + 1,
        "seed_entropy": [seed, 730002],
        "batch_size": batch_size,
        "selected_index": index,
        "selected_source": "zero" if index == 0 else "random_product",
        "selected_product_fidelity": float(fidelity[index]),
        "zero_state_fidelity": float(fidelity[0]),
        "random_mean_fidelity": float(fidelity[1:].mean()),
        "random_maximum_fidelity": float(fidelity[1:].max()),
        "selection": "maximum exact target overlap before entanglers; zero wins exact ties",
        "guarantee_scope": "product-layer starting fidelity only; no final-fidelity or runtime guarantee",
        "basis_order": "little endian: site i is bit i",
    }, {"fidelity": fidelity, "polar": polar, "azimuth": azimuth}


def initial_parameters(
    target: NDArray, seed: int, arm: str, count: int = 1000, batch_size: int = 64
) -> tuple[dict, dict]:
    """Replace only the existing initial U3 angles with the selected product state.

    Returns:
        Metadata and compact arrays including the complete depth-one parameters.

    Raises:
        ValueError: If the arm, target or sampling settings are unsupported.
    """
    if arm not in ARMS:
        msg = "Unknown initialization arm."
        raise ValueError(msg)
    target, qubits = base.validate_target(target)
    start, wall = time.process_time(), time.perf_counter()
    theta, metadata = base.initial_parameters(target, "near_zero" if arm == "screened_product" else arm, seed)
    arrays = {}
    if arm == "screened_product":
        screening, arrays = screen_products(target, count, seed, batch_size)
        index = screening["selected_index"]
        u3 = np.column_stack((arrays["polar"][index], arrays["azimuth"][index], np.zeros(qubits)))
        theta[: 3 * qubits] = u3.ravel()
        metadata.update(screening)
    circuit = base.capacity.continuation.brickwall(qubits, 1, initial_single_qubit_layer=True)
    product = ParameterizedCircuit(qubits, circuit.gates[: 3 * qubits], num_params=3 * qubits)
    product_fidelity = StatevectorBackend(product, target).metrics(theta[: 3 * qubits])[1]
    if arm == "screened_product" and abs(product_fidelity - metadata["selected_product_fidelity"]) > 1e-10:
        msg = "Selected product state does not replay through the initial rotation layer."
        raise ValueError(msg)
    metadata.update({
        "arm": arm,
        "product_state_fidelity": product_fidelity,
        "initial_circuit_fidelity": StatevectorBackend(circuit, target).metrics(theta)[1],
        "cpu_seconds": time.process_time() - start,
        "wall_seconds": time.perf_counter() - wall,
        "paired_parameters": "unchanged outside the existing initial U3 layer",
    })
    return metadata, {**arrays, "theta": theta}


def screen_products_mps(target: MPS, count: int = 1000, seed: int = 300, batch_size: int = 64) -> tuple[dict, dict]:
    """Screen the identical product-state pool by polynomial-size MPS contractions.

    Returns:
        The original selection metadata and angles/scores, without dense vectors.

    Raises:
        ValueError: If tensor shapes, normalization or sampling settings are invalid.
    """
    if (
        type(count) is not int
        or count < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        msg = "Require positive integer sample/batch counts and a valid seed."
        raise ValueError(msg)
    if (
        target.length < 2
        or target.flipped
        or len(target.tensors) != target.length
        or any(t.ndim != 3 or t.shape[0] != 2 or not np.all(np.isfinite(t)) for t in target.tensors)
        or target.tensors[0].shape[1] != 1
        or target.tensors[-1].shape[2] != 1
        or any(a.shape[2] != b.shape[1] for a, b in zip(target.tensors[:-1], target.tensors[1:], strict=True))
        or abs(mps_overlap(target, target) - 1) > 1e-10
    ):
        msg = "Require a finite normalized, unflipped open-boundary qubit MPS."
        raise ValueError(msg)
    qubits = target.length
    rng = np.random.default_rng(np.random.SeedSequence([seed, 730002]))
    draw = rng.random((count, 2, qubits))
    probability, phase = draw[:, 0], 2 * np.pi * draw[:, 1]
    fidelity = np.empty(count + 1)
    zero = np.ones(1, dtype=complex)
    for tensor in target.tensors:
        zero = np.matmul(zero, tensor[0])
    fidelity[0] = abs(zero[0]) ** 2
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        environment = np.ones((end - start, 1), dtype=complex)
        for site, tensor in enumerate(target.tensors):
            local = np.column_stack((
                np.sqrt(1 - probability[start:end, site]),
                np.exp(1j * phase[start:end, site]) * np.sqrt(probability[start:end, site]),
            ))
            environment = np.einsum("bi,bp,pij->bj", environment, local.conj(), tensor, optimize=True)
        fidelity[start + 1 : end + 1] = np.abs(environment[:, 0]) ** 2
    index = int(np.argmax(fidelity))
    return {
        "count": count,
        "candidate_count": count + 1,
        "seed_entropy": [seed, 730002],
        "batch_size": batch_size,
        "selected_index": index,
        "selected_source": "zero" if index == 0 else "random_product",
        "selected_product_fidelity": float(fidelity[index]),
        "zero_state_fidelity": float(fidelity[0]),
        "random_mean_fidelity": float(fidelity[1:].mean()),
        "random_maximum_fidelity": float(fidelity[1:].max()),
        "selection": "maximum exact target overlap before entanglers; zero wins exact ties",
        "representation": "native_mps",
        "basis_order": "little endian: site i is bit i",
    }, {
        "fidelity": fidelity,
        "polar": np.vstack((np.zeros(qubits), np.arccos(1 - 2 * probability))),
        "azimuth": np.vstack((np.zeros(qubits), phase)),
    }


def initial_parameters_mps(target: MPS, seed: int, count: int = 1000, batch_size: int = 64) -> tuple[dict, dict]:
    """Encode the selected product in the unchanged brickwall initialization stream.

    Returns:
        Selection diagnostics and the complete depth-one parameter vector.
    """
    start = time.process_time()
    record, arrays = screen_products_mps(target, count, seed, batch_size)
    base.Budget().validate(seed)
    circuit = base.capacity.continuation.brickwall(target.length, 1, initial_single_qubit_layer=True)
    theta = base.capacity.continuation.legacy.extend(None, circuit.num_params, seed + 1001, 0.05)
    index = record["selected_index"]
    theta[: 3 * target.length] = np.column_stack((
        arrays["polar"][index],
        arrays["azimuth"][index],
        np.zeros(target.length),
    )).ravel()
    record.update({
        "arm": "screened_product",
        "seed": seed,
        "parameter_seed": seed + 1001,
        "product_state_fidelity": record["selected_product_fidelity"],
        "initial_circuit_fidelity": MPSBackend(circuit, target).metrics(theta)[1],
        "cpu_seconds": time.process_time() - start,
        "paired_parameters": "unchanged outside the existing initial U3 layer",
    })
    return record, {**arrays, "theta": theta}


def prepare_initialization(
    folder: Path, target: NDArray, config: runner.Config, arm: str, count: int = 1000, batch_size: int = 64
) -> tuple:
    """Save selection once; refuse input mixing and reuse only authenticated arrays.

    Returns:
        The sealed initialization record and arrays.

    Raises:
        ValueError: If settings, target, sources or cached inputs differ.
    """
    config.validate()
    target, qubits = base.validate_target(target)
    if qubits != config.qubits or arm not in ARMS:
        msg = "Initialization target or arm differs from the configuration."
        raise ValueError(msg)
    if type(count) is not int or count < 1 or type(batch_size) is not int or batch_size < 1:
        msg = "Require positive integer sample/batch counts."
        raise ValueError(msg)
    base.Budget().validate(config.seed)
    manifest = {
        "version": VERSION,
        "arm": arm,
        "count": count,
        "batch_size": batch_size,
        "config": asdict(config),
        "target_sha256": runner.target_digest(target),
        "sources": sources(),
    }
    path = folder / "screening-manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Screening manifest differs; use a fresh output directory."
        raise ValueError(msg)
    support.write_json(path, manifest)
    saved = support.load_checkpoint(folder / "initialization")
    if saved is None:
        record, arrays = initial_parameters(target, config.seed, arm, count, batch_size)
        support.save_checkpoint(folder / "initialization", record, arrays)
        saved = support.load_checkpoint(folder / "initialization")
    assert saved is not None
    return saved


def run_case(
    folder: Path,
    target: NDArray,
    config: runner.Config,
    *,
    target_id: str,
    target_index: int = 0,
    arm: str = "screened_product",
    count: int = 1000,
    batch_size: int = 64,
) -> dict:
    """Feed selected angles into unchanged construction, B pruning and noisy updates.

    Returns:
        Sealed results, exact small-system fidelities and sampling-inclusive costs.

    Raises:
        ValueError: If a dense assessment above six qubits or changed cache is requested.
    """
    if config.qubits > 6:
        msg = "Full noisy screening comparisons with exact density assessment are limited to six qubits."
        raise ValueError(msg)
    initial, arrays = prepare_initialization(folder, target, config, arm, count, batch_size)
    saved = support.load_checkpoint(folder / "final")
    if saved is not None:
        if (
            saved[0]["target_id"] != target_id
            or saved[0]["target_index"] != target_index
            or saved[0]["initialization_seal_sha256"] != support.digest(folder / "initialization/complete.json")
            or saved[0]["training_seal_sha256"] != support.digest(folder / "training/final/complete.json")
            or saved[0]["selection_sha256"] != support.digest(folder / "training/selection_locked.json")
        ):
            msg = "Screened result identity or downstream seals differ."
            raise ValueError(msg)
        return saved[0]
    # The exact oracle is evaluated below, after the unchanged validation lock.
    trained = runner.run_case(
        folder / "training",
        target,
        replace(config, exact_density_check=False),
        target_id=target_id,
        target_index=target_index,
        initial_theta=arrays["theta"],
    )
    start, wall = time.process_time(), time.perf_counter()
    endpoints = {}
    selected = trained["selected_depth"]
    if selected is not None:
        directory = folder / "training" / f"depth_{selected:02d}"
        built = support.load_checkpoint(directory / "construction")
        pruned = support.load_checkpoint(directory / "pruning")
        refined = support.load_checkpoint(directory / "refinement")
        assert built is not None
        assert pruned is not None
        assert refined is not None
        circuit = support.restore_circuit(pruned[0]["circuit"])
        endpoints = {
            "ideal_before_pruning": StatevectorBackend(support.restore_circuit(built[0]["circuit"]), target).metrics(
                built[1]["theta"]
            )[1],
            "ideal_after_pruning": StatevectorBackend(circuit, target).metrics(pruned[1]["theta"])[1],
            "noisy_before_fine_tuning": runner.legacy.matrix_score(
                circuit, pruned[1]["theta"], target, config.hardware, "tjm"
            ),
            "noisy_after_fine_tuning": runner.legacy.matrix_score(
                circuit, refined[1]["theta"], target, config.hardware, "tjm"
            ),
        }
    assessment_cpu = time.process_time() - start
    result = {
        "version": VERSION,
        "target_id": target_id,
        "target_index": target_index,
        "arm": arm,
        "seed": config.seed,
        "selected_depth": selected,
        "status": trained["status"],
        "endpoints": endpoints,
        "resources": trained.get("resources"),
        "trajectory_fidelity": trained.get("fidelity"),
        "trajectory_standard_error": trained.get("trajectory_standard_error"),
        "screening": "initialization/record.json",
        "training": "training/final/record.json",
        "initialization_seal_sha256": support.digest(folder / "initialization/complete.json"),
        "training_seal_sha256": support.digest(folder / "training/final/complete.json"),
        "selection_sha256": trained["selection_sha256"],
        "initialization_cpu_seconds": initial["cpu_seconds"],
        "initialization_wall_seconds": initial["wall_seconds"],
        "exact_assessment_cpu_seconds": assessment_cpu,
        "exact_assessment_wall_seconds": time.perf_counter() - wall,
        "accounted_stage_cpu_seconds": initial["cpu_seconds"]
        + trained["total_search_cpu_seconds"]
        + trained.get("test_cpu_seconds", 0)
        + assessment_cpu,
        "cost_scope": (
            "original initialization, construction, pruning, noisy training, validation, test and exact assessment; "
            "excludes imports, serialization, resource compilation and target factorization"
        ),
    }
    return support.save_checkpoint(folder / "final", result, {})


def prepare_plan(folder: Path, config: runner.Config, count: int = 1000, batch_size: int = 64) -> dict:
    """Plan 54 matched six-qubit pipelines without launching any training.

    Returns:
        Explicit commands for nine targets, two arms and three independent pool seeds.

    Raises:
        ValueError: If an unsupported backend/budget or existing plan differs.
    """
    config.validate()
    if config.qubits != 6 or config.backend != "statevector" or config.profile_memory or config.exact_density_check:
        msg = "The matched plan requires six-qubit statevector execution."
        raise ValueError(msg)
    if type(count) is not int or count < 1 or type(batch_size) is not int or batch_size < 1:
        msg = "Require positive integer sample/batch counts."
        raise ValueError(msg)
    targets = tuple(t.target_id for t in load_target_collection().select(num_qubits=(6,)))
    jobs = []
    for target_id in targets:
        for seed in base.SEEDS:
            for arm in DEFAULT_ARMS:
                job_id = f"{target_id}__{arm}__seed{seed}"
                command = [
                    sys.executable,
                    "-m",
                    "experiments.topdown_product_screening",
                    "--action",
                    "run",
                    "--target-id",
                    target_id,
                    "--seed",
                    str(seed),
                    "--arm",
                    arm,
                    "--output",
                    str(folder.resolve() / "jobs" / job_id),
                    "--count",
                    str(count),
                    "--batch-size",
                    str(batch_size),
                    "--hardware",
                    config.hardware,
                ]
                for field in BUDGET_FIELDS:
                    command.extend(["--" + field.replace("_", "-"), str(getattr(config, field))])
                jobs.append({"job_id": job_id, "target_id": target_id, "arm": arm, "seed": seed, "command": command})
    plan = {
        "version": VERSION,
        "config": asdict(config),
        "count": count,
        "batch_size": batch_size,
        "sources": sources(),
        "fixture_sha256": support.digest(base.DEFAULT_TARGET_PATH),
        "target_sha256": {
            target_id: runner.target_digest(load_target(6, target_id).state_vector_copy()) for target_id in targets
        },
        "jobs": jobs,
        "launched": False,
    }
    path = folder / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        msg = "Screening plan differs."
        raise ValueError(msg)
    support.write_json(path, plan)
    return plan


def main(argv: list[str] | None = None) -> None:
    """Plan the comparison or explicitly screen/train one target and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "screen", "run"), default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-id", choices=tuple(t.target_id for t in load_target_collection().select(num_qubits=(6,)))
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=ARMS, default="screened_product")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--qubits", type=int, choices=(6, 12), default=6)
    parser.add_argument("--hardware", choices=("IBM", "Google"), default="IBM")
    defaults = runner.Config(backend="statevector")
    for field in BUDGET_FIELDS:
        parser.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(defaults, field))
    args = parser.parse_args(argv)
    config = replace(
        defaults,
        qubits=args.qubits,
        hardware=args.hardware,
        seed=300 if args.seed is None else args.seed,
        **{field: getattr(args, field) for field in BUDGET_FIELDS},
    )
    with threadpool_limits(limits=1):
        if args.action == "plan":
            result = prepare_plan(args.output, config, args.count, args.batch_size)
            print(json.dumps({"new_runs": len(result["jobs"]), "launched": False}))
        else:
            if args.target_id is None or args.seed is None:
                parser.error("An explicit target and seed are required for screening or training.")
            target = load_target(args.qubits, args.target_id).state_vector_copy()
            if args.action == "screen":
                result = prepare_initialization(args.output, target, config, args.arm, args.count, args.batch_size)[0]
            else:
                targets = [t.target_id for t in load_target_collection().select(num_qubits=(6,))]
                result = run_case(
                    args.output,
                    target,
                    config,
                    target_id=args.target_id,
                    target_index=targets.index(args.target_id),
                    arm=args.arm,
                    count=args.count,
                    batch_size=args.batch_size,
                )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
