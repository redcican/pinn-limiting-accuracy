"""Jacobian of the residual vector: dense, and through matrix-free products."""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from pla.problems.base import Block, Problem

Product = Callable[[Tensor], Tensor]


def _block_jacobian(problem: Problem, block: Block, theta: Tensor, chunk_size: int | None) -> Tensor:
    def values(parameters: Tensor, z: Tensor) -> Tensor:
        return block.op(lambda x: problem.net(parameters, x)[0], z)

    per_point = torch.vmap(torch.func.jacrev(values), in_dims=(None, 0), chunk_size=chunk_size)
    rows = per_point(theta, block.points.to(theta.dtype))  # (points, width, parameters)
    return math.sqrt(block.weight) * rows.reshape(block.size, theta.shape[0])


def jacobian(problem: Problem, theta: Tensor, *, chunk_size: int | None = None) -> Tensor:
    """Dense Jacobian ``J(theta)`` of the residual, of shape ``(m, p)``, evaluated
    in the precision of ``theta``.

    Each row is obtained by reverse-mode differentiation of one operator value
    with respect to the parameters; ``chunk_size`` limits the number of points
    that are differentiated at once.
    """
    return torch.cat([_block_jacobian(problem, block, theta, chunk_size) for block in problem.blocks])


def linearize(problem: Problem, theta: Tensor) -> tuple[Product, Product]:
    """Products ``v -> J v`` and ``w -> J^T w`` at ``theta`` without the matrix,
    by forward-mode and reverse-mode differentiation in the precision of ``theta``."""
    _, pullback = torch.func.vjp(problem.operator, theta)

    def jv(v: Tensor) -> Tensor:
        return torch.func.jvp(problem.operator, (theta,), (v.to(theta.dtype),))[1]

    def jtw(w: Tensor) -> Tensor:
        return pullback(w.to(theta.dtype))[0]

    return jv, jtw
