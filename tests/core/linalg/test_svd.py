# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for SciPy-style BLAS-thread-safe SVD."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import scipy.linalg

from mqt.yaqs.core import linalg


def test_svd_matches_scipy_reduced() -> None:
    """``linalg.svd`` with default driver should match SciPy on a small matrix."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
    a = a.astype(np.complex128)
    u_ref, s_ref, vh_ref = scipy.linalg.svd(a, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    u, s, vh = linalg.svd(a, full_matrices=False)
    np.testing.assert_allclose(s, s_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.abs(u), np.abs(u_ref), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.abs(vh), np.abs(vh_ref), rtol=1e-10, atol=1e-10)


def test_svd_compute_uv_false() -> None:
    """``compute_uv=False`` should return singular values only."""
    rng = np.random.default_rng(43)
    a = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    a = a.astype(np.complex128)
    s_only = linalg.svd(a, full_matrices=False, compute_uv=False)
    _, s_ref, _ = scipy.linalg.svd(a, full_matrices=False, compute_uv=True)
    np.testing.assert_allclose(s_only, s_ref, rtol=1e-10, atol=1e-10)


def test_svd_explicit_gesvd_driver() -> None:
    """Explicit ``lapack_driver='gesvd'`` should return consistent singular values."""
    rng = np.random.default_rng(44)
    a = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    a = a.astype(np.complex128)
    _, s_gesvd, _ = linalg.svd(a, full_matrices=False, lapack_driver="gesvd", check_finite=True)
    _, s_ref, _ = scipy.linalg.svd(a, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    np.testing.assert_allclose(s_gesvd, s_ref, rtol=1e-9, atol=1e-9)


def test_svd_gesdd_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``gesdd`` fails, ``linalg.svd`` retries with ``gesvd``."""
    calls: list[tuple[str, bool]] = []
    real_svd = scipy.linalg.svd

    def fake_svd(
        a_mat: np.ndarray,
        *,
        full_matrices: bool,
        compute_uv: bool = True,
        lapack_driver: str = "gesdd",
        check_finite: bool = True,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        calls.append((lapack_driver, check_finite))
        if lapack_driver == "gesdd":
            msg = "forced failure"
            raise scipy.linalg.LinAlgError(msg)
        return real_svd(
            a_mat,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver=lapack_driver,
            check_finite=check_finite,
        )

    monkeypatch.setattr(scipy.linalg, "svd", fake_svd)
    a = np.eye(3, dtype=np.complex128)
    u, s, vh = linalg.svd(a, full_matrices=False)
    assert calls[0] == ("gesdd", False)
    assert calls[1] == ("gesvd", True)
    np.testing.assert_allclose(u @ np.diag(s) @ vh, a, atol=1e-12)


@pytest.mark.parametrize("factor", [0, 1, 2, None])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
@pytest.mark.parametrize("check_finite", [False, True])
def test_svd_retries_nonfinite_output_without_lapack_exception(
    monkeypatch: pytest.MonkeyPatch, factor: int | None, invalid: float, *, check_finite: bool
) -> None:
    """An apparently successful divide-and-conquer SVD must not poison the next gate."""
    matrix = np.diag([1.0, 1e-16, 0.0]).astype(complex)
    original = scipy.linalg.svd
    calls = []

    def corrupted(a: np.ndarray, **kwargs: object) -> tuple | np.ndarray:
        calls.append((kwargs["lapack_driver"], kwargs["check_finite"]))
        result = original(a, **kwargs)
        if kwargs["lapack_driver"] == "gesdd":
            damaged = result[factor] if factor is not None else result
            damaged.flat[0] = invalid
        return result

    monkeypatch.setattr(scipy.linalg, "svd", corrupted)
    result = linalg.svd(matrix, full_matrices=False, compute_uv=factor is not None, check_finite=check_finite)
    assert calls == [("gesdd", check_finite), ("gesvd", True)]
    if isinstance(result, tuple):
        u, singular, vh = result
        np.testing.assert_allclose((u * singular) @ vh, matrix, atol=1e-15)
    else:
        np.testing.assert_allclose(result, [1, 1e-16, 0], rtol=1e-14, atol=0)


@pytest.mark.parametrize("driver", ["gesdd", "gesvd"])
@pytest.mark.parametrize("factor", [0, 1, 2, None])
def test_svd_rejects_nonfinite_fallback(
    monkeypatch: pytest.MonkeyPatch, driver: Literal["gesdd", "gesvd"], factor: int | None
) -> None:
    """Neither an explicit robust driver nor a fallback may return invalid tensors."""
    original = scipy.linalg.svd
    calls = []

    def corrupted(a: np.ndarray, **kwargs: object) -> tuple | np.ndarray:
        calls.append(kwargs["lapack_driver"])
        result = original(a, **kwargs)
        damaged = result[factor] if factor is not None else result
        damaged.flat[0] = np.nan
        return result

    monkeypatch.setattr(scipy.linalg, "svd", corrupted)
    with pytest.raises(scipy.linalg.LinAlgError, match=r"gesvd.*non-finite"):
        linalg.svd(np.eye(2, dtype=complex), lapack_driver=driver, compute_uv=factor is not None)
    assert calls == (["gesdd", "gesvd"] if driver == "gesdd" else ["gesvd"])


def test_svd_captured_ferromagnetic_input_reconstructs_without_corruption() -> None:
    """Replay the actual finite matrix whose gesdd vectors became non-finite on Leonhard."""
    with np.load(Path(__file__).parent / "data" / "ferromagnetic_svd_input.npz", allow_pickle=False) as fixture:
        matrix = fixture["matrix"]
    assert matrix.shape == (256, 256)
    assert np.all(np.isfinite(matrix))
    u, singular, vh = linalg.svd(matrix, full_matrices=False)
    assert all(np.all(np.isfinite(factor)) for factor in (u, singular, vh))
    assert np.linalg.norm((u * singular) @ vh - matrix) / np.linalg.norm(matrix) < 1e-12
    np.testing.assert_allclose(u.conj().T @ u, np.eye(len(singular)), atol=1e-12)
    np.testing.assert_allclose(vh @ vh.conj().T, np.eye(len(singular)), atol=1e-12)
