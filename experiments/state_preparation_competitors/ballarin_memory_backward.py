# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Measure one author SVD VJP with dense unit cotangents, without a circuit tape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.state_preparation_competitors.ballarin_memory import SVDRecorder, file_hash, write_receipt
from experiments.state_preparation_competitors.vendor.ballarin_comb.svd import SVDGESDD


def backward_probe(output: Path, capture: Path) -> dict:
    """Profile the unchanged regularized derivative with explicit synthetic inputs.

    Dense unit cotangents force all derivative terms and their storage to be
    exercised. This is a workspace diagnostic, not the circuit's gradient,
    an optimization update, or a rigorous bound on every backward execution.
    """
    metadata = json.loads((capture / "capture.json").read_text())
    matrix_path = capture / "matrix.npy"
    if file_hash(matrix_path) != metadata["matrix_sha256"]:
        msg = "Captured matrix hash does not match."
        raise ValueError(msg)
    matrix = torch.from_numpy(np.load(matrix_path, allow_pickle=False)).requires_grad_()
    recorder = SVDRecorder(output)
    with recorder.active():
        u, singular, v = SVDGESDD.apply(matrix, torch.tensor(1e-12, dtype=matrix.real.dtype), None)
        recorder.phase("backward_inputs", index=1)
        cotangents = tuple(torch.ones_like(tensor) for tensor in (u, singular, v))
        recorder.phase("svd_backward", index=1)
        gradient = torch.autograd.grad((u, singular, v), matrix, grad_outputs=cotangents)[0]
        recorder.phase("gradient_summary", index=1)
        result = {
            "source_capture": str(capture),
            "source_matrix_sha256": metadata["matrix_sha256"],
            "shape": list(matrix.shape),
            "source_strides": metadata["strides"],
            "loaded_strides": list(matrix.stride()),
            "regularization": 1e-12,
            "input_bytes": matrix.numel() * matrix.element_size(),
            "factor_bytes": sum(t.numel() * t.element_size() for t in (u, singular, v)),
            "cotangent_bytes": sum(t.numel() * t.element_size() for t in cotangents),
            "gradient_bytes": gradient.numel() * gradient.element_size(),
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "cotangents": "dense unit tensors; synthetic workspace probe, not circuit cotangents",
            "circuit_gradient_computed": False,
        }
    write_receipt(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the isolated derivative under the existing memory-diagnostic supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(backward_probe(args.output, args.capture)), flush=True)


if __name__ == "__main__":
    main()
