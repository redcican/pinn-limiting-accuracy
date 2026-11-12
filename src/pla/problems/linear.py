"""Linear test problems on the unit cube with the exact solution
``u(x) = prod_k sin(pi x_k)``: regression, Poisson, Helmholtz and biharmonic."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from pla.network import MLP
from pla.problems.base import Block, Problem, ScalarFunction
from pla.problems.operators import bilaplacian, gradient, laplacian
from pla.problems.sampling import box_boundary, sobol

DEFAULT_TEST_POINTS: int = 100_000


def product_sine(x: Tensor) -> Tensor:
    """``prod_k sin(pi x_k)`` for the rows of ``x``."""
    return torch.prod(torch.sin(math.pi * x), dim=1)


def _product_sine_gradient(x: Tensor) -> Tensor:
    sines, cosines = torch.sin(math.pi * x), torch.cos(math.pi * x)
    columns = []
    for k in range(x.shape[1]):
        others = torch.cat([sines[:, :k], sines[:, k + 1:]], dim=1)
        columns.append(math.pi * cosines[:, k] * torch.prod(others, dim=1))
    return torch.stack(columns, dim=1)


def _dirichlet(u: ScalarFunction, x: Tensor) -> Tensor:
    return u(x).reshape(1)


def _assemble(net: MLP, dim: int, blocks: list[Block], data: list[Tensor], n_test: int,
              seed: int, coefficients: dict[str, float] | None = None) -> Problem:
    weighted = [math.sqrt(block.weight) * values.reshape(-1) for block, values in zip(blocks, data)]
    return Problem(net=net, blocks=tuple(blocks), data=torch.cat(weighted), linear=True,
                   exact=product_sine, test_points=sobol(n_test, dim, seed + 1000),
                   coefficients=dict(coefficients or {}))


def helmholtz(net: MLP, *, wavenumber: float, n_interior: int, n_boundary: int, seed: int,
              dim: int = 2, boundary_weight: float = 100.0,
              n_test: int = DEFAULT_TEST_POINTS) -> Problem:
    """``-Laplace(u) - wavenumber**2 u = f`` in the unit cube with Dirichlet data."""
    shift = wavenumber ** 2
    interior = sobol(n_interior, dim, seed)
    boundary, _ = box_boundary(n_boundary, dim, seed + 1)

    def op(u: ScalarFunction, x: Tensor) -> Tensor:
        return (-laplacian(u, x) - shift * u(x)).reshape(1)

    blocks = [Block("interior", interior, 1.0 / n_interior, op),
              Block("boundary", boundary, boundary_weight * 2 * dim / n_boundary, _dirichlet)]
    data = [(dim * math.pi ** 2 - shift) * product_sine(interior), product_sine(boundary)]
    return _assemble(net, dim, blocks, data, n_test, seed, {"wavenumber": float(wavenumber)})


def poisson(net: MLP, *, dim: int, n_interior: int, n_boundary: int, seed: int,
            boundary_weight: float = 100.0, n_test: int = DEFAULT_TEST_POINTS) -> Problem:
    """``-Laplace(u) = f`` in the unit cube of dimension ``dim`` with Dirichlet data."""
    return helmholtz(net, wavenumber=0.0, n_interior=n_interior, n_boundary=n_boundary, seed=seed,
                     dim=dim, boundary_weight=boundary_weight, n_test=n_test)


def regression(net: MLP, *, dim: int, n_interior: int, seed: int,
               n_test: int = DEFAULT_TEST_POINTS) -> Problem:
    """Least squares fit of the exact solution (operator of order zero)."""
    interior = sobol(n_interior, dim, seed)
    blocks = [Block("interior", interior, 1.0 / n_interior, _dirichlet)]
    return _assemble(net, dim, blocks, [product_sine(interior)], n_test, seed)


def biharmonic(net: MLP, *, n_interior: int, n_boundary: int, seed: int,
               boundary_weight: float = 400.0, n_test: int = DEFAULT_TEST_POINTS) -> Problem:
    """``Laplace^2(u) = f`` in the unit square with clamped conditions: the
    value and the normal derivative are prescribed on the boundary."""
    dim = 2
    interior = sobol(n_interior, dim, seed)
    boundary, normals = box_boundary(n_boundary, dim, seed + 1)

    def op(u: ScalarFunction, x: Tensor) -> Tensor:
        return bilaplacian(u, x).reshape(1)

    def clamped(u: ScalarFunction, z: Tensor) -> Tensor:
        x, normal = z[:dim], z[dim:]
        return torch.stack([u(x), gradient(u, x) @ normal])

    blocks = [Block("interior", interior, 1.0 / n_interior, op),
              Block("boundary", torch.cat([boundary, normals], dim=1),
                    boundary_weight * 2 * dim / n_boundary, clamped, width=2)]
    normal_derivative = torch.sum(_product_sine_gradient(boundary) * normals, dim=1)
    data = [(dim * math.pi ** 2) ** 2 * product_sine(interior),
            torch.stack([product_sine(boundary), normal_derivative], dim=1)]
    return _assemble(net, dim, blocks, data, n_test, seed)


def helmholtz_blocks(problem: Problem) -> tuple[Block, Block, float]:
    """Interior block, boundary block and wavenumber of a problem built by
    :func:`helmholtz` or :func:`poisson`; a ``ValueError`` is raised for any
    other problem."""
    names = tuple(block.name for block in problem.blocks)
    if (not problem.linear or names != ("interior", "boundary")
            or "wavenumber" not in problem.coefficients
            or any(block.width != 1 for block in problem.blocks)):
        raise ValueError("expected a problem built by helmholtz() or poisson()")
    return problem.blocks[0], problem.blocks[1], problem.coefficients["wavenumber"]
