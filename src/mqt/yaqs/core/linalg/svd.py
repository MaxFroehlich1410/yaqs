# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style SVD with gesdd-to-gesvd fallback.

Uses multi-threaded LAPACK/OpenBLAS by default; the BLAS pool is *not* capped
here. The wrapper retries with the more robust ``gesvd`` driver (and
``check_finite=True``) when ``gesdd`` fails or returns non-finite factors.
Returned factors are checked even when input checking is disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

LapackDriver = Literal["gesdd", "gesvd"]


def _checked_svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool,
    compute_uv: bool,
    lapack_driver: LapackDriver,
    check_finite: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]] | NDArray[np.float64]:
    """Decompose and reject invalid output before it reaches another tensor kernel.

    LAPACK can report success while returning non-finite singular vectors for
    a finite, nearly rank-deficient matrix. Input validation alone cannot detect
    this case, and zero singular values do not make NaN vectors harmless:
    multiplying a NaN by zero still produces a NaN.

    Returns:
        Finite singular values and, when requested, finite singular vectors.

    Raises:
        scipy.linalg.LinAlgError: If LAPACK returns non-finite output without an error.
    """
    result = scipy.linalg.svd(
        a,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        lapack_driver=lapack_driver,
        check_finite=check_finite,
    )
    factors = result if isinstance(result, tuple) else (result,)
    if any(not np.all(np.isfinite(factor)) for factor in factors):
        msg = f"SVD driver {lapack_driver} returned non-finite factors."
        raise scipy.linalg.LinAlgError(msg)
    return result


@overload
def svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool = ...,
    compute_uv: Literal[True] = ...,
    lapack_driver: LapackDriver = ...,
    check_finite: bool = ...,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]]: ...


@overload
def svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool = ...,
    compute_uv: Literal[False],
    lapack_driver: LapackDriver = ...,
    check_finite: bool = ...,
) -> NDArray[np.float64]: ...


def svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    lapack_driver: LapackDriver = "gesdd",
    check_finite: bool = False,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]] | NDArray[np.float64]:
    """Singular value decomposition with automatic driver fallback.

    When ``lapack_driver`` is ``"gesdd"`` (default), retries with ``"gesvd"``
    and ``check_finite=True`` on failure or non-finite output. Finite output is
    required from both drivers; an invalid fallback result raises an error.

    Mirrors :func:`scipy.linalg.svd` for the supported keyword arguments.

    Args:
        a: Complex matrix to decompose.
        full_matrices: If ``True`` (default), return full-size ``U`` and ``Vh``;
            otherwise return reduced (thin) factors.
        compute_uv: If ``True`` (default), compute and return ``U`` and ``Vh``;
            if ``False``, return only the singular values.
        lapack_driver: LAPACK driver to use (``"gesdd"`` or ``"gesvd"``).
            Defaults to ``"gesdd"``.
        check_finite: Whether to check input arrays for finite values.
            Defaults to ``False``.

    Returns:
        ``(U, s, Vh)`` when ``compute_uv`` is ``True``, else the 1D array of
        singular values ``s``.
    """
    if lapack_driver != "gesdd":
        return _checked_svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver=lapack_driver,
            check_finite=check_finite,
        )
    try:
        return _checked_svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver="gesdd",
            check_finite=check_finite,
        )
    except (scipy.linalg.LinAlgError, ValueError, FloatingPointError):
        return _checked_svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver="gesvd",
            check_finite=True,
        )
