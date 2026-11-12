"""Levenberg-Marquardt steps computed in a chosen precision.

Three ways of computing the step (2.4) are provided: the normal equations with
the Gram matrix ``G = J^T J + mu I``, the residual-space form with
``K = J J^T + mu I`` (both Gram-type), and Householder QR factorization of the
augmented matrix ``[J; sqrt(mu) I]`` (orthogonal).
"""
from __future__ import annotations

import math
from enum import Enum

import torch
from torch import Tensor

from pla.linalg.cholesky import CholeskyBreakdown, cholesky_solve
from pla.linalg.householder import lstsq_qr, qr_factor
from pla.precision import Arithmetic


class Solver(str, Enum):
    """How the linear least squares problem of a step is solved."""

    QR = "qr"
    NE = "ne"
    DUAL = "dual"


class StepFailure(ArithmeticError):
    """The step could not be computed in the working precision."""


def augmented(j: Tensor, mu: float) -> Tensor:
    """The augmented matrix ``J_mu = [J; sqrt(mu) I]`` of (2.7)."""
    p = j.shape[1]
    return torch.cat([j, math.sqrt(mu) * torch.eye(p, dtype=j.dtype)])


def _gram(a: Tensor, mu: float, arith: Arithmetic) -> Tensor:
    """``a^T a + mu I`` formed in the working precision."""
    a_w = arith.cast(a)
    eye = torch.eye(a_w.shape[1], dtype=a_w.dtype)
    return arith.round(arith.round(a_w.T @ a_w) + arith.cast(torch.tensor(mu, dtype=torch.float64)) * eye)


def _check(delta: Tensor) -> Tensor:
    if not bool(torch.isfinite(delta).all()):
        raise StepFailure("the computed step is not finite")
    return delta


def lm_step(j: Tensor, r: Tensor, mu: float, solver: Solver, bits: int) -> Tensor:
    """Step ``delta`` of (2.4) for Jacobian ``j`` and residual ``r``, computed in
    ``bits``-bit arithmetic and returned in double precision.

    Raises :class:`StepFailure` if a Cholesky factorization breaks down or the
    result is not finite.
    """
    if mu < 0.0:
        raise ValueError("the damping parameter must be nonnegative")
    arith = Arithmetic(bits)
    j64, r64 = j.to(torch.float64), r.to(torch.float64)
    if solver is Solver.QR:
        rhs = torch.cat([-r64, torch.zeros(j64.shape[1], dtype=torch.float64)])
        delta, _ = lstsq_qr(augmented(j64, mu), rhs, bits)
        return _check(delta)
    try:
        if solver is Solver.NE:
            j_w = arith.cast(j64)
            rhs = arith.round(-(j_w.T @ arith.cast(r64))).to(torch.float64)
            return _check(cholesky_solve(_gram(j64, mu, arith).to(torch.float64), rhs, bits))
        if mu == 0.0:
            raise ValueError("the residual-space form requires positive damping")
        j_w = arith.cast(j64)
        y = cholesky_solve(_gram(j64.T, mu, arith).to(torch.float64), r64, bits)
        delta = arith.round(-(j_w.T @ arith.cast(y)))
        return _check(delta.to(torch.float64))
    except CholeskyBreakdown as exc:
        raise StepFailure(str(exc)) from exc


def triangular_factor(j: Tensor, mu: float, bits: int) -> Tensor:
    """Triangular factor of the augmented matrix, computed in ``bits``-bit
    arithmetic and returned in double precision (the preconditioner of MP-GN)."""
    return qr_factor(augmented(j.to(torch.float64), mu), bits)
