"""Adam followed by L-BFGS, both from ``torch.optim``, in double precision."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from pla.baselines.common import BaselineResult
from pla.problems.base import Problem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FirstOrderConfig:
    adam_steps: int = 1000
    adam_lr: float = 1e-3
    lbfgs_steps: int = 1000
    lbfgs_history: int = 50

    def __post_init__(self) -> None:
        if self.adam_steps < 0 or self.lbfgs_steps < 0 or self.adam_lr <= 0.0:
            raise ValueError("invalid settings")


def _loss(problem: Problem, theta: Tensor) -> Tensor:
    r = problem.residual(theta)
    return 0.5 * (r @ r)


def adam_lbfgs(problem: Problem, theta0: Tensor,
               config: FirstOrderConfig = FirstOrderConfig()) -> BaselineResult:
    """Run ``adam_steps`` steps of Adam and then ``lbfgs_steps`` iterations of
    L-BFGS with a strong Wolfe line search."""
    theta = theta0.to(torch.float64).clone().requires_grad_(True)
    losses = [problem.loss(theta.detach())]
    adam = torch.optim.Adam([theta], lr=config.adam_lr)
    for _ in range(config.adam_steps):
        adam.zero_grad()
        value = _loss(problem, theta)
        value.backward()
        adam.step()
        losses.append(problem.loss(theta.detach()))
    lbfgs = torch.optim.LBFGS([theta], lr=1.0, max_iter=1, history_size=config.lbfgs_history,
                              line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        lbfgs.zero_grad()
        value = _loss(problem, theta)
        value.backward()
        return value

    for _ in range(config.lbfgs_steps):
        lbfgs.step(closure)
        losses.append(problem.loss(theta.detach()))
    logger.debug("final loss %.3e", losses[-1])
    return BaselineResult(theta.detach(), tuple(losses))
