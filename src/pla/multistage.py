"""Multi-stage training for linear problems (section 4.3).

Stage ``j`` fits a new network to the residual of the sum of its predecessors,
scaled to unit norm, as in iterative refinement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from pla.problems.base import Problem

logger = logging.getLogger(__name__)
Trainer = Callable[[Problem, Tensor], Tensor]


@dataclass(frozen=True)
class MultistageResult:
    """Parameters and scales of the stages; ``residual_norms[j]`` is the norm of
    the residual ``z_j`` of the sum of the first ``j`` networks."""

    thetas: tuple[Tensor, ...]
    alphas: tuple[float, ...]
    residual_norms: tuple[float, ...]


def sum_residual(problem: Problem, thetas: Sequence[Tensor], alphas: Sequence[float],
                 dtype: torch.dtype = torch.float64) -> Tensor:
    """Residual ``z = S U - b`` of the sum ``U = sum_j alphas[j] u_{thetas[j]}``,
    evaluated in ``dtype``."""
    if len(thetas) != len(alphas):
        raise ValueError("thetas and alphas must have the same length")
    z = -problem.data.to(dtype)
    for theta, alpha in zip(thetas, alphas):
        z = z + alpha * problem.operator(theta.to(dtype))
    return z


def multistage(problem: Problem, trainer: Trainer, *, n_stages: int, seed: int,
               dtype: torch.dtype = torch.float64) -> MultistageResult:
    """Train ``n_stages`` networks in sequence.

    ``trainer(stage_problem, theta0)`` returns the trained parameters of one
    stage. Stage ``j`` starts from the initialization with seed ``seed + j``,
    and the residual of the sum is evaluated in ``dtype``.
    """
    if not problem.linear:
        raise ValueError("multi-stage training as formulated here requires a linear problem")
    if n_stages < 1:
        raise ValueError("at least one stage is required")
    thetas: list[Tensor] = []
    alphas: list[float] = []
    z = sum_residual(problem, thetas, alphas, dtype)
    norms = [float(torch.linalg.norm(z))]
    for stage in range(n_stages):
        alpha = norms[-1]
        if alpha == 0.0:
            break
        stage_problem = problem.with_data(-(z / alpha).to(torch.float64))
        theta = trainer(stage_problem, problem.net.init(seed + stage)).to(torch.float64)
        thetas.append(theta)
        alphas.append(alpha)
        z = sum_residual(problem, thetas, alphas, dtype)
        norms.append(float(torch.linalg.norm(z)))
        logger.info("stage %d: residual norm %.3e", stage + 1, norms[-1])
    return MultistageResult(tuple(thetas), tuple(alphas), tuple(norms))
