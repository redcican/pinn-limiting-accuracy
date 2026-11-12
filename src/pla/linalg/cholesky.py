"""Symmetric positive definite solves by Cholesky factorization in a chosen precision."""
from __future__ import annotations

import torch
from torch import Tensor

from pla.precision import Arithmetic, chop


class CholeskyBreakdown(ArithmeticError):
    """The factorization met a nonpositive pivot: the matrix is not numerically
    positive definite in the working precision."""


def _native(g: Tensor, rhs: Tensor, arith: Arithmetic) -> Tensor:
    factor, info = torch.linalg.cholesky_ex(arith.cast(g))
    if int(info) != 0 or not bool(torch.isfinite(factor).all()):
        raise CholeskyBreakdown(f"nonpositive pivot at column {int(info)}")
    x = torch.cholesky_solve(arith.cast(rhs).unsqueeze(1), factor).squeeze(1)
    return x.to(torch.float64)


def _loop(g: Tensor, rhs: Tensor, arith: Arithmetic) -> Tensor:
    def rnd(x: Tensor) -> Tensor:  # also for native formats, which are stored in double here
        return chop(x, arith.bits)

    a = rnd(g.to(torch.float64))
    n = a.shape[0]
    low = torch.zeros_like(a)
    for j in range(n):
        pivot = rnd(a[j, j] - rnd(low[j, :j] @ low[j, :j]))
        if not bool(pivot > 0.0):
            raise CholeskyBreakdown(f"nonpositive pivot at column {j}")
        low[j, j] = rnd(torch.sqrt(pivot))
        if j + 1 < n:
            column = rnd(a[j + 1:, j] - rnd(low[j + 1:, :j] @ low[j, :j]))
            low[j + 1:, j] = rnd(column / low[j, j])
    y = torch.zeros(n, dtype=torch.float64)
    b = rnd(rhs.to(torch.float64))
    for i in range(n):
        y[i] = rnd(rnd(b[i] - rnd(low[i, :i] @ y[:i])) / low[i, i])
    x = torch.zeros(n, dtype=torch.float64)
    for i in range(n - 1, -1, -1):
        x[i] = rnd(rnd(y[i] - rnd(low[i + 1:, i] @ x[i + 1:])) / low[i, i])
    return x


def cholesky_solve(g: Tensor, rhs: Tensor, bits: int, *, force_loop: bool = False) -> Tensor:
    """Solve ``g x = rhs`` for symmetric positive definite ``g`` in ``bits``-bit
    arithmetic. Raises :class:`CholeskyBreakdown` if a pivot is not positive."""
    arith = Arithmetic(bits)
    if arith.native and not force_loop:
        return _native(g, rhs, arith)
    return _loop(g, rhs, arith)
