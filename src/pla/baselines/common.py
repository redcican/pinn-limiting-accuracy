"""Shared pieces of the baseline optimizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class BaselineResult:
    """Final parameters and the loss after each iteration (``losses[0]`` is the
    initial loss)."""

    theta: Tensor
    losses: tuple[float, ...]


def step_grid(n_steps: int = 31, ratio: float = 0.5) -> tuple[float, ...]:
    """Geometric grid ``1, ratio, ratio^2, ...`` of trial step lengths."""
    if n_steps < 1 or not 0.0 < ratio < 1.0:
        raise ValueError("the grid needs at least one step and a ratio in (0, 1)")
    return tuple(ratio ** k for k in range(n_steps))


def grid_line_search(loss: Callable[[Tensor], float], theta: Tensor, direction: Tensor,
                     current: float, grid: tuple[float, ...]) -> tuple[Tensor, float]:
    """Minimize ``loss(theta - s * direction)`` over the step lengths ``s`` of
    ``grid``. The point is kept if no trial value lies below ``current``."""
    best_theta, best_loss = theta, current
    for step in grid:
        trial = theta - step * direction
        value = loss(trial)
        if value < best_loss:  # false for a value that is not a number
            best_theta, best_loss = trial, value
    return best_theta, best_loss
