"""Levenberg-Marquardt iteration with separate precisions for the residual,
the Jacobian and the linear solver (the three unit roundoffs of section 2.3)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from pla.jacobian import jacobian
from pla.precision import DOUBLE_BITS, SINGLE_BITS, Arithmetic
from pla.problems.base import Problem
from pla.solvers import Solver, StepFailure, lm_step

logger = logging.getLogger(__name__)
NATIVE_BITS: tuple[int, int] = (SINGLE_BITS, DOUBLE_BITS)


@dataclass(frozen=True)
class LMConfig:
    """Settings of a Levenberg-Marquardt run.

    ``bits_r`` and ``bits_j`` are the precisions of the residual and of the
    Jacobian (24 or 53, since both come from automatic differentiation);
    ``bits_s`` is the precision of the linear solver and may be any format.
    The parameters are stored in the precision of the residual. The run stops
    when ``F(theta_t) > (1 - stall_tol) F(theta_{t - stall_window})``.
    """

    solver: Solver = Solver.QR
    bits_r: int = DOUBLE_BITS
    bits_j: int = DOUBLE_BITS
    bits_s: int = DOUBLE_BITS
    mu0: float = 1.0
    mu_min: float = 1e-30
    mu_max: float = 1e30
    nu: float = 4.0
    max_iter: int = 1000
    stall_window: int = 20
    stall_tol: float = 1e-3
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.bits_r not in NATIVE_BITS or self.bits_j not in NATIVE_BITS:
            raise ValueError("residual and Jacobian precisions must be 24 or 53 bits")
        Arithmetic(self.bits_s)  # validates the range
        if self.nu <= 1.0:
            raise ValueError("the damping factor nu must exceed one")
        if not 0.0 < self.mu_min <= self.mu0 <= self.mu_max:
            raise ValueError("the damping limits must satisfy 0 < mu_min <= mu0 <= mu_max")
        if self.max_iter < 1 or self.stall_window < 1 or not 0.0 < self.stall_tol < 1.0:
            raise ValueError("invalid iteration limits")


@dataclass(frozen=True)
class LMResult:
    """Outcome of a run. ``losses[t]`` is the loss after ``t`` iterations,
    evaluated in the residual precision; ``mu`` is the damping parameter at the
    end and ``mu_accepted`` that of the last accepted step."""

    theta: Tensor
    losses: tuple[float, ...]
    mu: float
    mu_accepted: float
    accepted: int
    stalled: bool


def half_square(r: Tensor) -> float:
    """The loss ``0.5 ||r||^2`` in the precision of ``r``."""
    return 0.5 * float(r @ r)


def has_stalled(losses: list[float], window: int, tol: float) -> bool:
    """Stall test of Algorithm 4.1."""
    return len(losses) > window and losses[-1] > (1.0 - tol) * losses[-1 - window]


def levenberg_marquardt(problem: Problem, theta0: Tensor, config: LMConfig = LMConfig()) -> LMResult:
    """Minimize the loss of ``problem`` from ``theta0``."""
    dtype_r, dtype_j = Arithmetic(config.bits_r).dtype, Arithmetic(config.bits_j).dtype
    theta = theta0.to(dtype_r)
    r = problem.residual(theta)
    loss = half_square(r)
    losses = [loss]
    mu, mu_accepted, accepted, stalled = config.mu0, config.mu0, 0, False
    j: Tensor | None = None
    for iteration in range(1, config.max_iter + 1):
        if j is None:
            j = jacobian(problem, theta.to(dtype_j), chunk_size=config.chunk_size)
        improved = False
        try:
            delta = lm_step(j, r, mu, config.solver, config.bits_s)
            trial = theta + delta.to(dtype_r)
            r_trial = problem.residual(trial)
            loss_trial = half_square(r_trial)
            improved = loss_trial < loss  # false for a loss that is not a number
        except StepFailure as failure:
            logger.debug("iteration %d: %s", iteration, failure)
        if improved:
            theta, r, loss, j = trial, r_trial, loss_trial, None
            mu_accepted, accepted = mu, accepted + 1
            mu = max(config.mu_min, mu / config.nu)
        else:
            mu = config.nu * mu
        losses.append(loss)
        logger.debug("iteration %d: loss %.3e, damping %.1e", iteration, loss, mu)
        if loss == 0.0:
            break
        if has_stalled(losses, config.stall_window, config.stall_tol) or mu > config.mu_max:
            stalled = True
            break
    return LMResult(theta, tuple(losses), mu, mu_accepted, accepted, stalled)
