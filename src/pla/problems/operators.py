"""Pointwise differential operators on scalar functions, by automatic differentiation."""
from __future__ import annotations

import torch
from torch import Tensor

from pla.problems.base import ScalarFunction


def gradient(u: ScalarFunction, x: Tensor) -> Tensor:
    return torch.func.jacrev(u)(x)


def hessian(u: ScalarFunction, x: Tensor) -> Tensor:
    return torch.func.jacfwd(torch.func.jacrev(u))(x)


def laplacian(u: ScalarFunction, x: Tensor) -> Tensor:
    return torch.diagonal(hessian(u, x)).sum()


def bilaplacian(u: ScalarFunction, x: Tensor) -> Tensor:
    return laplacian(lambda y: laplacian(u, y), x)
