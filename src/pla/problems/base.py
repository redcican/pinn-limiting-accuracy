"""Residual vector (2.2) of a collocation problem.

A problem is a list of blocks. Each block holds collocation points, a
quadrature weight and a pointwise operator; the weighted operator values of
all blocks form the vector ``S u``, and the residual is ``S u - b`` with the
data vector ``b``. For a linear problem ``S`` is linear in ``u``; for a
nonlinear one the data vector holds only the initial and boundary data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from pla.network import MLP

ScalarFunction = Callable[[Tensor], Tensor]
PointOperator = Callable[[ScalarFunction, Tensor], Tensor]


@dataclass(frozen=True)
class Block:
    """Collocation points with a weight and a pointwise operator.

    ``op(u, z)`` maps a scalar function ``u`` and one row ``z`` of ``points``
    to a tensor of ``width`` values. A row may carry more than the coordinates
    of a point, for example an outward normal.
    """

    name: str
    points: Tensor
    weight: float
    op: PointOperator
    width: int = 1

    @property
    def size(self) -> int:
        return self.points.shape[0] * self.width


@dataclass(frozen=True)
class Problem:
    """A collocation problem for the network ``net``. ``coefficients`` records
    the parameters of the operator for code that needs its structure."""

    net: MLP
    blocks: tuple[Block, ...]
    data: Tensor
    linear: bool
    exact: Callable[[Tensor], Tensor] | None = None
    test_points: Tensor | None = None
    coefficients: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.shape != (self.m,):
            raise ValueError(f"data has shape {tuple(self.data.shape)}, expected ({self.m},)")

    @property
    def m(self) -> int:
        """Number of residual entries."""
        return sum(block.size for block in self.blocks)

    def apply(self, u: ScalarFunction, dtype: torch.dtype) -> Tensor:
        """The vector ``S u`` for a scalar function ``u`` of one point."""
        parts = []
        for block in self.blocks:
            values = torch.vmap(lambda z, op=block.op: op(u, z))(block.points.to(dtype))
            parts.append(math.sqrt(block.weight) * values.reshape(-1))
        return torch.cat(parts)

    def operator(self, theta: Tensor) -> Tensor:
        """``S u_theta``, evaluated in the precision of ``theta``."""
        return self.apply(lambda x: self.net(theta, x)[0], theta.dtype)

    def residual(self, theta: Tensor) -> Tensor:
        """Residual vector ``r(theta)``, evaluated in the precision of ``theta``."""
        return self.operator(theta) - self.data.to(theta.dtype)

    def loss(self, theta: Tensor) -> float:
        r = self.residual(theta)
        return 0.5 * float(r @ r)

    def with_data(self, data: Tensor) -> "Problem":
        """The same operator with another data vector (used by multi-stage training)."""
        return replace(self, data=data)

    def solution(self, thetas: Sequence[Tensor], alphas: Sequence[float] | None, x: Tensor) -> Tensor:
        """Values of the sum ``sum_j alphas[j] * u_{thetas[j]}`` at the rows of ``x``."""
        scales = alphas if alphas is not None else [1.0] * len(thetas)
        if len(scales) != len(thetas):
            raise ValueError("thetas and alphas must have the same length")
        total = torch.zeros(x.shape[0], dtype=torch.float64)
        for theta, alpha in zip(thetas, scales):
            total = total + alpha * self.net(theta.to(torch.float64), x)[:, 0]
        return total

    def relative_error(self, thetas: Sequence[Tensor], alphas: Sequence[float] | None = None) -> float:
        """Relative L2 error on the test points, evaluated in double precision."""
        if self.exact is None or self.test_points is None:
            raise ValueError("the problem has no exact solution or no test points")
        reference = self.exact(self.test_points)
        error = self.solution(thetas, alphas, self.test_points) - reference
        return float(torch.linalg.norm(error) / torch.linalg.norm(reference))
