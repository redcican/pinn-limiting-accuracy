"""Nonlinear test problems in space-time on ``(-1, 1) x (0, 1]`` with points
``z = (x, t)``: the viscous Burgers equation and the Allen-Cahn equation."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from pla.network import MLP
from pla.problems.base import Block, Problem, ScalarFunction
from pla.problems.operators import gradient, hessian
from pla.problems.sampling import sobol

SPACE_LENGTH: float = 2.0  # length of the interval (-1, 1)


def _space_time(n: int, seed: int) -> Tensor:
    unit = sobol(n, 2, seed)
    return torch.stack([2.0 * unit[:, 0] - 1.0, unit[:, 1]], dim=1)


def _initial_line(n: int, seed: int) -> Tensor:
    x = 2.0 * sobol(n, 1, seed)[:, 0] - 1.0
    return torch.stack([x, torch.zeros_like(x)], dim=1)


def _value(u: ScalarFunction, z: Tensor) -> Tensor:
    return u(z).reshape(1)


def _data(blocks: list[Block], values: list[Tensor]) -> Tensor:
    return torch.cat([math.sqrt(block.weight) * value.reshape(-1)
                      for block, value in zip(blocks, values)])


def burgers(net: MLP, *, n_interior: int, n_initial: int, n_boundary: int, seed: int,
            viscosity: float = 0.01 / math.pi, penalty: float = 50.0) -> Problem:
    """``u_t + u u_x - viscosity u_xx = 0`` with ``u(x, 0) = -sin(pi x)`` and
    ``u(-1, t) = u(1, t) = 0``."""
    interior, initial = _space_time(n_interior, seed), _initial_line(n_initial, seed + 1)
    n_left = (n_boundary + 1) // 2  # each side has its own sequence of times
    times = torch.cat([sobol(n_left, 1, seed + 2)[:, 0], sobol(n_boundary - n_left, 1, seed + 3)[:, 0]])
    sides = torch.cat([-torch.ones(n_left, dtype=torch.float64),
                       torch.ones(n_boundary - n_left, dtype=torch.float64)])
    boundary = torch.stack([sides, times], dim=1)

    def op(u: ScalarFunction, z: Tensor) -> Tensor:
        first, second = gradient(u, z), hessian(u, z)
        return (first[1] + u(z) * first[0] - viscosity * second[0, 0]).reshape(1)

    blocks = [Block("interior", interior, SPACE_LENGTH / n_interior, op),
              Block("initial", initial, penalty * SPACE_LENGTH / n_initial, _value),
              Block("boundary", boundary, penalty * 2.0 / n_boundary, _value)]
    values = [torch.zeros(n_interior, dtype=torch.float64), -torch.sin(math.pi * initial[:, 0]),
              torch.zeros(n_boundary, dtype=torch.float64)]
    return Problem(net=net, blocks=tuple(blocks), data=_data(blocks, values), linear=False)


def allen_cahn(net: MLP, *, n_interior: int, n_initial: int, n_boundary: int, seed: int,
               diffusion: float = 1e-4, reaction: float = 5.0, penalty: float = 50.0) -> Problem:
    """``u_t - diffusion u_xx + reaction (u^3 - u) = 0`` with
    ``u(x, 0) = x^2 sin(2 pi x)`` and periodic conditions on the value and on
    the space derivative, as in Kiyani et al. (2025)."""
    interior, initial = _space_time(n_interior, seed), _initial_line(n_initial, seed + 1)
    times = sobol(n_boundary, 1, seed + 2)

    def op(u: ScalarFunction, z: Tensor) -> Tensor:
        value = u(z)
        return (gradient(u, z)[1] - diffusion * hessian(u, z)[0, 0]
                + reaction * (value ** 3 - value)).reshape(1)

    def periodic(u: ScalarFunction, t: Tensor) -> Tensor:
        one = torch.ones_like(t)
        left, right = torch.cat([-one, t]), torch.cat([one, t])
        return torch.stack([u(left) - u(right), gradient(u, left)[0] - gradient(u, right)[0]])

    blocks = [Block("interior", interior, SPACE_LENGTH / n_interior, op),
              Block("initial", initial, penalty * SPACE_LENGTH / n_initial, _value),
              Block("periodic", times, penalty / n_boundary, periodic, width=2)]
    x0 = initial[:, 0]
    values = [torch.zeros(n_interior, dtype=torch.float64), x0 ** 2 * torch.sin(2 * math.pi * x0),
              torch.zeros(n_boundary, 2, dtype=torch.float64)]
    return Problem(net=net, blocks=tuple(blocks), data=_data(blocks, values), linear=False)
