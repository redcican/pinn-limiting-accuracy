"""Least squares by Householder QR factorization in a chosen precision.

For binary32 and binary64 the LAPACK routine behind ``torch.linalg.qr`` is
used. For every other format the factorization is written out column by
column, and the result of each vector operation is rounded to the target
precision. Inner products are accumulated in double precision and rounded
once; this is the usual compromise of chop-style emulation and slightly
underestimates the error of a true low-precision inner product.
"""
from __future__ import annotations

import torch
from torch import Tensor

from pla.precision import Arithmetic, chop


def _native(a: Tensor, b: Tensor, arith: Arithmetic) -> tuple[Tensor, Tensor]:
    a_w, b_w = arith.cast(a), arith.cast(b)
    q, r = torch.linalg.qr(a_w, mode="reduced")
    x = torch.linalg.solve_triangular(r, (q.T @ b_w).unsqueeze(1), upper=True).squeeze(1)
    return x.to(torch.float64), r.to(torch.float64)


def _back_substitution(r: Tensor, c: Tensor, arith: Arithmetic) -> Tensor:
    p = r.shape[0]
    x = torch.zeros(p, dtype=torch.float64)
    for i in range(p - 1, -1, -1):
        tail = chop(r[i, i + 1:] @ x[i + 1:], arith.bits)
        x[i] = chop(chop(c[i] - tail, arith.bits) / r[i, i], arith.bits)
    return x


def _loop(a: Tensor, b: Tensor, arith: Arithmetic) -> tuple[Tensor, Tensor]:
    def rnd(x: Tensor) -> Tensor:  # also for native formats, which are stored in double here
        return chop(x, arith.bits)

    work = rnd(a.to(torch.float64).clone())
    rhs = rnd(b.to(torch.float64).clone())
    m, p = work.shape
    for k in range(p):
        x = work[k:, k]
        norm = rnd(torch.sqrt(rnd(x @ x)))
        if norm == 0.0:
            continue
        alpha = -norm if x[0] >= 0 else norm
        v = x.clone()
        v[0] = rnd(v[0] - alpha)
        beta = rnd(2.0 / rnd(v @ v))
        # Apply H = I - beta v v^T to the trailing columns and to the right-hand side.
        coef = rnd(beta * rnd(v @ work[k:, k:]))
        work[k:, k:] = rnd(work[k:, k:] - rnd(torch.outer(v, coef)))
        rhs[k:] = rnd(rhs[k:] - rnd(v * rnd(beta * rnd(v @ rhs[k:]))))
        work[k, k] = alpha
        work[k + 1:, k] = 0.0
    r = torch.triu(work[:p, :])
    return _back_substitution(r, rhs[:p], arith), r


def lstsq_qr(a: Tensor, b: Tensor, bits: int, *, force_loop: bool = False) -> tuple[Tensor, Tensor]:
    """Solve ``min ||a x - b||`` by Householder QR in ``bits``-bit arithmetic.

    Returns the solution and the triangular factor, both stored in double
    precision. ``force_loop`` selects the written-out factorization also for
    native formats (used to test it against LAPACK).
    """
    if a.ndim != 2 or a.shape[0] < a.shape[1]:
        raise ValueError("a must be a matrix with at least as many rows as columns")
    arith = Arithmetic(bits)
    if arith.native and not force_loop:
        return _native(a, b, arith)
    return _loop(a, b, arith)


def qr_factor(a: Tensor, bits: int) -> Tensor:
    """Triangular factor of the Householder QR factorization of ``a`` in
    ``bits``-bit arithmetic, stored in double precision."""
    if a.ndim != 2 or a.shape[0] < a.shape[1]:
        raise ValueError("a must be a matrix with at least as many rows as columns")
    arith = Arithmetic(bits)
    if arith.native:
        return torch.linalg.qr(arith.cast(a), mode="r").R.to(torch.float64)
    return _loop(a, torch.zeros(a.shape[0], dtype=torch.float64), arith)[1]
