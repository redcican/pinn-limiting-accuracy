"""Kronecker-factored approximation of the Gauss-Newton matrix for operators
with a Laplacian, in the manner of Dangel, Mueller and Zeinhofer (NeurIPS 2024).

The value, the first derivatives and the Laplacian of the network are
propagated forward together. Every layer then acts as one linear map on
``S = d + 2`` vectors per point (one vector on the boundary, where only the
value is needed), and the block of the Gauss-Newton matrix that belongs to a
layer is approximated, separately for the interior and the boundary term, by a
Kronecker product ``B (x) A``: ``A`` is the mean of the outer products of the
layer inputs, and ``B`` the sum of the outer products of the derivatives of
the residual entries with respect to the layer outputs. The sum of the two
Kronecker products is inverted exactly through a generalized eigenvalue
decomposition. The step length comes from a line search on a geometric grid.

This is a compact variant for the Poisson and Helmholtz problems of
:mod:`pla.problems.linear`; it has no momentum and no moving averages.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from pla.baselines.common import BaselineResult, grid_line_search, step_grid
from pla.network import MLP
from pla.problems.base import Problem
from pla.problems.linear import helmholtz_blocks

logger = logging.getLogger(__name__)
Factors = tuple[Tensor, Tensor]  # (A, B) of one layer and one term


@dataclass(frozen=True)
class KFACConfig:
    damping: float = 1e-8
    max_iter: int = 200
    grid: tuple[float, ...] = step_grid()

    def __post_init__(self) -> None:
        if self.damping <= 0.0 or self.max_iter < 1 or not self.grid:
            raise ValueError("invalid settings")


def _extended_weights(net: MLP, theta: Tensor) -> list[Tensor]:
    """Per layer the matrix ``[W, b]``."""
    return [torch.cat([weight, bias.unsqueeze(1)], dim=1) for weight, bias in net.layers(theta)]


def forward_laplacian(net: MLP, theta: Tensor, x: Tensor, *, derivatives: bool
                      ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    """Values ``u`` and Laplacians of the network at the rows of ``x``, with
    the inputs (shape ``(points, S, n_in + 1)``) and outputs (``(points, S,
    n_out)``) of every layer. With ``derivatives=False`` only the value is
    propagated (``S = 1``) and the returned Laplacian is zero."""
    points, dim = x.shape
    quantities = [x.unsqueeze(1)]
    if derivatives:
        quantities += [torch.eye(dim, dtype=x.dtype).expand(points, dim, dim),
                       torch.zeros(points, 1, dim, dtype=x.dtype)]
    z = torch.cat(quantities, dim=1)
    flag = torch.zeros(z.shape[1], dtype=x.dtype)
    flag[0] = 1.0  # the bias acts on the value only
    inputs, outputs = [], []
    weights = _extended_weights(net, theta)
    for index, weight in enumerate(weights):
        z = torch.cat([z, flag.expand(points, -1).unsqueeze(2)], dim=2)
        out = z @ weight.T
        inputs.append(z)
        outputs.append(out)
        if index == len(weights) - 1:
            break
        a = out[:, 0]
        h = torch.tanh(a)
        if not derivatives:
            z = h.unsqueeze(1)
            continue
        slope = 1.0 - h ** 2
        da, lap_a = out[:, 1:dim + 1], out[:, dim + 1]
        lap_h = -2.0 * h * slope * (da ** 2).sum(dim=1) + slope * lap_a
        z = torch.cat([h.unsqueeze(1), slope.unsqueeze(1) * da, lap_h.unsqueeze(1)], dim=1)
    last = outputs[-1]
    laplacian = last[:, dim + 1, 0] if derivatives else torch.zeros(points, dtype=x.dtype)
    return last[:, 0, 0], laplacian, inputs, outputs


def _terms(problem: Problem, theta: Tensor
           ) -> list[tuple[Tensor, list[Tensor], list[Tensor]]]:
    """Weighted operator values of the interior and the boundary term with the
    layer inputs and outputs."""
    interior, boundary, wavenumber = helmholtz_blocks(problem)
    u, lap, z_in, out_in = forward_laplacian(problem.net, theta, interior.points, derivatives=True)
    inside = math.sqrt(interior.weight) * (-lap - wavenumber ** 2 * u)
    g, _, z_bd, out_bd = forward_laplacian(problem.net, theta, boundary.points, derivatives=False)
    return [(inside, z_in, out_in), (math.sqrt(boundary.weight) * g, z_bd, out_bd)]


def kfac_factors(problem: Problem, theta: Tensor
                 ) -> tuple[Tensor, list[list[Factors]]]:
    """Gradient of the loss and, for every layer, the factors ``(A, B)`` of the
    interior and of the boundary term."""
    theta = theta.detach().to(torch.float64).requires_grad_(True)
    terms = _terms(problem, theta)
    r = torch.cat([values for values, _, _ in terms]) - problem.data
    gradient = torch.autograd.grad(0.5 * (r @ r), theta, retain_graph=True)[0]
    n_layers = len(problem.net.sizes) - 1
    factors: list[list[Factors]] = [[] for _ in range(n_layers)]
    for values, inputs, outputs in terms:
        sensitivities = torch.autograd.grad(values.sum(), outputs, retain_graph=True)
        for layer, (z, g) in enumerate(zip(inputs, sensitivities)):
            z2, g2 = z.detach().reshape(-1, z.shape[2]), g.reshape(-1, g.shape[2])
            factors[layer].append((z2.T @ z2 / z2.shape[0], g2.T @ g2))
    return gradient.detach(), factors


def solve_two_kronecker(b1: Tensor, a1: Tensor, b2: Tensor, a2: Tensor, v: Tensor) -> Tensor:
    """Solve ``b1 X a1 + b2 X a2 = v`` for ``X``, with symmetric positive
    semidefinite ``a1``, ``b1`` and symmetric positive definite ``a2``, ``b2``."""
    l_b, l_a = torch.linalg.cholesky(b2), torch.linalg.cholesky(a2)

    def whiten(matrix: Tensor, factor: Tensor) -> Tensor:
        half = torch.linalg.solve_triangular(factor, matrix, upper=False)
        return torch.linalg.solve_triangular(factor, half.T, upper=False).T

    d_b, e_b = torch.linalg.eigh(whiten(b1, l_b))
    d_a, e_a = torch.linalg.eigh(whiten(a1, l_a))
    rhs = torch.linalg.solve_triangular(l_b, v, upper=False)
    rhs = torch.linalg.solve_triangular(l_a, rhs.T, upper=False).T
    y = e_b @ ((e_b.T @ rhs @ e_a) / (torch.outer(d_b, d_a) + 1.0)) @ e_a.T
    x = torch.linalg.solve_triangular(l_b.T, y, upper=True)
    return torch.linalg.solve_triangular(l_a.T, x.T, upper=True).T


def kfac_direction(net: MLP, gradient: Tensor, factors: list[list[Factors]], damping: float) -> Tensor:
    """Apply the inverse of the damped Kronecker-factored matrix to ``gradient``.
    Each factor is damped by ``sqrt(damping)`` times the identity."""
    shift = math.sqrt(damping)
    parts, start = [], 0
    for (n_in, n_out), terms in zip(zip(net.sizes[:-1], net.sizes[1:]), factors):
        size = n_in * n_out
        block = torch.cat([gradient[start:start + size].reshape(n_out, n_in),
                           gradient[start + size:start + size + n_out].unsqueeze(1)], dim=1)
        start += size + n_out
        (a1, b1), (a2, b2) = ((a + shift * torch.eye(a.shape[0], dtype=a.dtype),
                               b + shift * torch.eye(b.shape[0], dtype=b.dtype)) for a, b in terms)
        x = solve_two_kronecker(b1, a1, b2, a2, block)
        parts.extend([x[:, :n_in].reshape(-1), x[:, n_in]])
    return torch.cat(parts)


def kfac(problem: Problem, theta0: Tensor, config: KFACConfig = KFACConfig()) -> BaselineResult:
    """Kronecker-factored Gauss-Newton iteration in double precision."""
    theta = theta0.to(torch.float64)
    loss = problem.loss(theta)
    losses = [loss]
    for iteration in range(1, config.max_iter + 1):
        gradient, factors = kfac_factors(problem, theta)
        direction = kfac_direction(problem.net, gradient, factors, config.damping)
        theta, loss = grid_line_search(problem.loss, theta, direction, loss, config.grid)
        losses.append(loss)
        logger.debug("iteration %d: loss %.3e", iteration, loss)
    return BaselineResult(theta, tuple(losses))
