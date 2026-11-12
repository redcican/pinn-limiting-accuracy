"""LSQR with a right preconditioner and the stopping test of Algorithm 4.1.

The iteration is the Golub-Kahan bidiagonalization of Paige and Saunders (ACM
Trans. Math. Software, 8 (1982), pp. 43-71), written for operators that are
available only through products. No reorthogonalization is performed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

Operator = Callable[[Tensor], Tensor]
TINY: float = 1e-300
CONSISTENT: float = 1e-14  # relative residual at which a consistent system counts as solved


@dataclass(frozen=True)
class LSQRResult:
    """Outcome of an LSQR run.

    ``rnorm`` and ``arnorm`` are ``||b - A x||`` and ``||A^T (b - A x)||``,
    recomputed from ``x`` at the end; ``s_hat`` is the smallest singular value of the
    bidiagonal matrix (``nan`` if it was never needed).
    """

    x: Tensor
    iterations: int
    rnorm: float
    arnorm: float
    s_hat: float
    converged: bool


def _smallest_singular_value(alphas: list[float], betas: list[float]) -> float:
    """Smallest singular value of the lower bidiagonal matrix with diagonal
    ``alphas`` and subdiagonal ``betas`` (``k + 1`` rows, ``k`` columns)."""
    k = len(alphas)
    bidiagonal = torch.zeros(k + 1, k, dtype=torch.float64)
    index = torch.arange(k)
    bidiagonal[index, index] = torch.tensor(alphas, dtype=torch.float64)
    bidiagonal[index + 1, index] = torch.tensor(betas, dtype=torch.float64)
    return float(torch.linalg.svdvals(bidiagonal)[-1])


def lsqr(matvec: Operator, rmatvec: Operator, b: Tensor, n: int, *, omega: float,
         tau: float | None = None, max_iter: int) -> LSQRResult:
    """Minimize ``||A x - b||`` over ``x`` in ``R^n``, started from zero.

    The iteration stops when ``||A^T r|| <= min(omega, tau * s_hat) * ||r||``
    for the current residual ``r``. With ``tau=None`` the test is the plain
    forcing test ``||A^T r|| <= omega * ||r||``. The estimate ``s_hat`` is an
    upper bound of the smallest singular value of ``A`` that decreases to it
    as the iteration proceeds; it is evaluated only when the plain test holds.
    The test is run on the recurrence estimates during the iteration and is
    verified with recomputed norms at the end; ``converged`` reports the
    verified outcome.
    """
    x = torch.zeros(n, dtype=b.dtype)
    beta = float(torch.linalg.norm(b))
    if beta == 0.0:
        return LSQRResult(x, 0, 0.0, 0.0, math.nan, True)
    norm_b = beta
    u = b / beta
    v = rmatvec(u)
    alpha = float(torch.linalg.norm(v))
    if alpha == 0.0:  # b is orthogonal to the range of A
        return LSQRResult(x, 0, beta, 0.0, math.nan, True)
    v = v / alpha
    w = v.clone()
    phibar, rhobar = beta, alpha
    alphas: list[float] = []
    betas: list[float] = []
    s_hat = math.nan
    for iteration in range(1, max_iter + 1):
        alphas.append(alpha)
        u = matvec(v) - alpha * u
        beta = float(torch.linalg.norm(u))
        betas.append(beta)
        if beta > TINY:
            u = u / beta
            v = rmatvec(u) - beta * v
            alpha = float(torch.linalg.norm(v))
            if alpha > TINY:
                v = v / alpha
        else:
            alpha = 0.0
        rho = math.hypot(rhobar, beta)
        c, s = rhobar / rho, beta / rho
        theta = s * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        rnorm, arnorm = phibar, phibar * alpha * abs(c)
        done = beta <= TINY or alpha <= TINY  # exact termination of the bidiagonalization
        if not done and rnorm <= CONSISTENT * norm_b:  # consistent system, solved
            done = True
        if not done and arnorm <= omega * rnorm:
            if tau is not None:
                s_hat = _smallest_singular_value(alphas, betas)
            done = tau is None or arnorm <= min(omega, tau * s_hat) * rnorm
        if done:
            return _verified(matvec, rmatvec, b, x, iteration, s_hat, omega, tau)
    result = _verified(matvec, rmatvec, b, x, max_iter, s_hat, omega, tau)
    return result


def _verified(matvec: Operator, rmatvec: Operator, b: Tensor, x: Tensor, iteration: int,
              s_hat: float, omega: float, tau: float | None) -> LSQRResult:
    """Result with the norms recomputed from ``x``. The recurrences of LSQR
    drift from the true residual in finite precision, so convergence is
    declared only if the stopping test holds for the recomputed quantities
    (or the residual vanishes relative to ``b``)."""
    residual = b - matvec(x)
    rnorm = float(torch.linalg.norm(residual))
    arnorm = float(torch.linalg.norm(rmatvec(residual)))
    bound = omega if tau is None or s_hat != s_hat else min(omega, tau * s_hat)
    converged = arnorm <= bound * rnorm or rnorm <= CONSISTENT * float(torch.linalg.norm(b))
    return LSQRResult(x, iteration, rnorm, arnorm, s_hat, converged)


def preconditioned_lsqr(jv: Operator, jtw: Operator, r_hat: Tensor, mu: float, r: Tensor, *,
                        omega: float, tau: float | None, max_iter: int) -> LSQRResult:
    """Solve the damped problem (2.7), ``min ||J_mu delta + [r; 0]||``, by LSQR
    with the right preconditioner ``r_hat``.

    ``jv`` and ``jtw`` return products with ``J`` and ``J^T``. The iteration
    runs on ``B = J_mu r_hat^{-1}``, and the returned ``x`` is the step
    ``delta = r_hat^{-1} y``.
    """
    m, p = r.shape[0], r_hat.shape[0]
    root = math.sqrt(mu)

    def solve(rhs: Tensor, transpose: bool) -> Tensor:
        matrix = r_hat.T if transpose else r_hat
        return torch.linalg.solve_triangular(matrix, rhs.unsqueeze(1), upper=not transpose).squeeze(1)

    def matvec(y: Tensor) -> Tensor:
        z = solve(y, False)
        return torch.cat([jv(z), root * z])

    def rmatvec(w: Tensor) -> Tensor:
        return solve(jtw(w[:m]) + root * w[m:], True)

    rhs = torch.cat([-r, torch.zeros(p, dtype=r.dtype)])
    result = lsqr(matvec, rmatvec, rhs, p, omega=omega, tau=tau, max_iter=max_iter)
    return LSQRResult(solve(result.x, False), result.iterations, result.rnorm, result.arnorm,
                      result.s_hat, result.converged)
