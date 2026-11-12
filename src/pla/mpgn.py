"""Mixed-precision Gauss-Newton method (Algorithm 4.1).

Phase 1 runs Levenberg-Marquardt with Householder QR in single precision until
it stalls. In phase 2 the residual is evaluated in double precision, the
Jacobian and the triangular factor of the augmented matrix in single
precision, and the step is computed in double precision by LSQR with that
factor as right preconditioner and with matrix-free Jacobian products.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor

from pla.jacobian import jacobian, linearize
from pla.lm import LMConfig, half_square, has_stalled, levenberg_marquardt
from pla.lsqr import LSQRResult, preconditioned_lsqr
from pla.precision import DOUBLE_BITS, SINGLE_BITS, unit_roundoff
from pla.problems.base import Problem
from pla.solvers import Solver, triangular_factor

logger = logging.getLogger(__name__)


def _single_precision_phase() -> LMConfig:
    return LMConfig(solver=Solver.QR, bits_r=SINGLE_BITS, bits_j=SINGLE_BITS, bits_s=SINGLE_BITS)


@dataclass(frozen=True)
class MPGNConfig:
    """Settings of Algorithm 4.1: ``tau`` in (0, 1/2], forcing term ``omega``
    in (0, tau], damping factor ``nu``. ``phase1=None`` skips the first phase.
    ``c_qr`` is the constant of the orthogonal threshold (2.10)."""

    tau: float = 0.5
    omega: float = 1e-3
    nu: float = 4.0
    mu0: float = 1.0
    mu_max: float = 1e30
    c_qr: float = 1.0
    max_outer: int = 1000
    max_inner: int = 200
    stall_window: int = 20
    stall_tol: float = 1e-3
    chunk_size: int | None = None
    phase1: LMConfig | None = field(default_factory=_single_precision_phase)

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 0.5:
            raise ValueError("tau must lie in (0, 1/2]")
        if not 0.0 < self.omega <= self.tau:
            raise ValueError("the forcing term omega must lie in (0, tau]")
        if self.nu <= 1.0:
            raise ValueError("the damping factor nu must exceed one")
        if self.max_outer < 1 or self.max_inner < 1:
            raise ValueError("invalid iteration limits")


@dataclass(frozen=True)
class MPGNResult:
    """Outcome of a run. ``losses`` covers both phases (single-precision values
    in phase 1, double-precision values from ``switch_iteration`` on), and
    ``inner_iterations`` lists the LSQR iterations of each step of phase 2.
    ``inner_failures`` counts the steps whose inner iteration ended without
    meeting the stopping test; such a step is still accepted if it lowers the
    loss."""

    theta: Tensor
    losses: tuple[float, ...]
    switch_iteration: int
    inner_iterations: tuple[int, ...]
    inner_failures: int
    mu: float
    stalled: bool


def damping_floor(factor_norm: float, *, tau: float, c_qr: float = 1.0) -> float:
    """Lower limit ``mu_min`` of section 4.2 for a factor of norm ``factor_norm``,
    which keeps the damping threshold above the double-precision one."""
    eta64 = c_qr * unit_roundoff(DOUBLE_BITS)
    eta32 = c_qr * unit_roundoff(SINGLE_BITS)
    return (eta64 / tau) ** 2 * factor_norm ** 2 / (1.0 - eta32) ** 2


@dataclass(frozen=True)
class MixedStep:
    """A step of phase 2: the step vector, the LSQR record, the spectral norm
    of the factor and the damping parameter that was used."""

    delta: Tensor
    inner: LSQRResult
    factor_norm: float
    mu: float


def _factor(j_single: Tensor, mu: float) -> tuple[Tensor, float]:
    r_hat = triangular_factor(j_single, mu, SINGLE_BITS)
    return r_hat, float(torch.linalg.matrix_norm(r_hat, ord=2))


def mixed_precision_step(problem: Problem, theta: Tensor, r: Tensor, mu: float, config: MPGNConfig,
                         j_single: Tensor | None = None) -> MixedStep:
    """One step of phase 2 at the double-precision point ``theta`` with residual ``r``.

    If ``mu`` lies below the lower limit of section 4.2 for the computed
    factor, it is raised to that limit and the factor is computed again.
    """
    if j_single is None:
        j_single = jacobian(problem, theta.to(torch.float32), chunk_size=config.chunk_size)
    r_hat, factor_norm = _factor(j_single, mu)
    floor = damping_floor(factor_norm, tau=config.tau, c_qr=config.c_qr)
    if mu < floor:
        mu = floor
        r_hat, factor_norm = _factor(j_single, mu)
    jv, jtw = linearize(problem, theta)
    inner = preconditioned_lsqr(jv, jtw, r_hat, mu, r, omega=config.omega, tau=config.tau,
                                max_iter=config.max_inner)
    return MixedStep(inner.x, inner, factor_norm, mu)


def mpgn(problem: Problem, theta0: Tensor, config: MPGNConfig = MPGNConfig()) -> MPGNResult:
    """Run Algorithm 4.1 from ``theta0``.

    Phase 2 starts from the damping parameter of the last accepted step of
    phase 1; the value at the end of phase 1 has been raised by the rejected
    steps that make up the stall.
    """
    losses: list[float] = []
    mu = config.mu0
    theta = theta0.to(torch.float64)
    if config.phase1 is not None:
        first = levenberg_marquardt(problem, theta0, config.phase1)
        theta, mu = first.theta.to(torch.float64), first.mu_accepted
        losses.extend(first.losses[:-1])
    switch = len(losses)
    r = problem.residual(theta)
    loss = half_square(r)
    phase = [loss]
    inner_iterations: list[int] = []
    j_single: Tensor | None = None
    stalled, inner_failures = False, 0
    for iteration in range(1, config.max_outer + 1):
        if j_single is None:
            j_single = jacobian(problem, theta.to(torch.float32), chunk_size=config.chunk_size)
        step = mixed_precision_step(problem, theta, r, mu, config, j_single)
        mu = step.mu
        inner_iterations.append(step.inner.iterations)
        if not step.inner.converged:
            inner_failures += 1
            logger.warning("iteration %d: LSQR did not meet the stopping test in %d iterations",
                           iteration, step.inner.iterations)
        floor = damping_floor(step.factor_norm, tau=config.tau, c_qr=config.c_qr)
        improved = False
        if bool(torch.isfinite(step.delta).all()):
            trial = theta + step.delta
            r_trial = problem.residual(trial)
            loss_trial = half_square(r_trial)
            improved = loss_trial < loss
        if improved:
            theta, r, loss, j_single = trial, r_trial, loss_trial, None
            mu = max(floor, mu / config.nu)
        else:
            mu = config.nu * mu
        phase.append(loss)
        logger.debug("iteration %d: loss %.3e, damping %.1e, %d inner iterations",
                     iteration, loss, mu, step.inner.iterations)
        if loss == 0.0:
            break
        if has_stalled(phase, config.stall_window, config.stall_tol) or mu > config.mu_max:
            stalled = True
            break
    return MPGNResult(theta, tuple(losses + phase), switch, tuple(inner_iterations),
                      inner_failures, mu, stalled)
