"""Energy natural gradient descent and its form in residual space.

For the discretized loss the energy Gram matrix is ``G = J^T J``. The primal
method (Mueller and Zeinhofer, ICML 2023) takes the direction
``G^+ J^T r``, computed by a least squares solve with ``G``; the dual method
(Jnini and Vella, 2025) takes ``J^T (J J^T + damping I)^+ r``, which requires a
matrix of the size of the residual only. Both use a line search on a
geometric grid of step lengths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from pla.baselines.common import BaselineResult, grid_line_search, step_grid
from pla.jacobian import jacobian
from pla.problems.base import Problem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ENGDConfig:
    dual: bool = False
    damping: float = 0.0
    max_iter: int = 200
    grid: tuple[float, ...] = step_grid()
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.damping < 0.0 or self.max_iter < 1 or not self.grid:
            raise ValueError("invalid settings")


def _pseudo_solve(matrix: Tensor, rhs: Tensor) -> Tensor:
    return torch.linalg.lstsq(matrix, rhs.unsqueeze(1), driver="gelsd").solution.squeeze(1)


def natural_gradient(j: Tensor, r: Tensor, *, dual: bool, damping: float = 0.0) -> Tensor:
    """Natural gradient direction in double precision; the update is
    ``theta - s * direction``."""
    if dual:
        k = j @ j.T + damping * torch.eye(j.shape[0], dtype=j.dtype)
        return j.T @ _pseudo_solve(k, r)
    g = j.T @ j + damping * torch.eye(j.shape[1], dtype=j.dtype)
    return _pseudo_solve(g, j.T @ r)


def engd(problem: Problem, theta0: Tensor, config: ENGDConfig = ENGDConfig()) -> BaselineResult:
    """Energy natural gradient descent in double precision."""
    theta = theta0.to(torch.float64)
    loss = problem.loss(theta)
    losses = [loss]
    for iteration in range(1, config.max_iter + 1):
        j = jacobian(problem, theta, chunk_size=config.chunk_size)
        direction = natural_gradient(j, problem.residual(theta), dual=config.dual,
                                     damping=config.damping)
        theta, loss = grid_line_search(problem.loss, theta, direction, loss, config.grid)
        losses.append(loss)
        logger.debug("iteration %d: loss %.3e", iteration, loss)
    return BaselineResult(theta, tuple(losses))
