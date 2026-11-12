import math

import pytest
import torch

from pla.network import MLP
from pla.problems.linear import biharmonic, helmholtz, poisson, product_sine, regression
from pla.problems.nonlinear import allen_cahn, burgers
from pla.problems.sampling import box_boundary, sobol

pytestmark = pytest.mark.unit


def _exact(x: torch.Tensor) -> torch.Tensor:
    return torch.prod(torch.sin(math.pi * x))


def test_sampling_is_reproducible_and_lies_on_the_boundary() -> None:
    assert torch.equal(sobol(16, 3, seed=1), sobol(16, 3, seed=1))
    points, normals = box_boundary(24, 3, seed=1)
    on_face = ((points == 0.0) | (points == 1.0)).sum(dim=1)
    assert bool((on_face >= 1).all())
    assert torch.allclose(normals.abs().sum(dim=1), torch.ones(24))
    assert set(torch.unique(normals).tolist()) == {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_poisson_operator_applied_to_the_exact_solution_gives_the_data(dim: int) -> None:
    problem = poisson(MLP((dim, 4, 1)), dim=dim, n_interior=32, n_boundary=12, seed=0)
    assert problem.m == 44
    applied = problem.apply(_exact, torch.float64)
    assert torch.allclose(applied, problem.data, atol=1e-10)
    assert problem.linear


def test_helmholtz_biharmonic_and_regression_are_consistent_with_their_data() -> None:
    net = MLP((2, 4, 1))
    for problem in (helmholtz(net, wavenumber=2 * math.pi, n_interior=20, n_boundary=8, seed=0),
                    biharmonic(net, n_interior=20, n_boundary=8, seed=0),
                    regression(net, dim=2, n_interior=20, seed=0)):
        assert torch.allclose(problem.apply(_exact, torch.float64), problem.data, atol=1e-8)
    assert biharmonic(net, n_interior=20, n_boundary=8, seed=0).m == 20 + 2 * 8
    assert regression(net, dim=2, n_interior=20, seed=0).m == 20


def test_residual_follows_the_precision_of_the_parameters() -> None:
    net = MLP((2, 6, 1))
    problem = poisson(net, dim=2, n_interior=30, n_boundary=10, seed=0)
    theta = net.init(seed=0)
    r64 = problem.residual(theta)
    r32 = problem.residual(theta.to(torch.float32))
    assert r64.dtype == torch.float64 and r32.dtype == torch.float32
    assert r64.shape == (40,)
    assert torch.allclose(r32.double(), r64, atol=1e-3)
    assert not torch.equal(r32.double(), r64)
    assert problem.loss(theta) == pytest.approx(0.5 * float(r64 @ r64))


def test_relative_error_of_a_sum_of_networks() -> None:
    net = MLP((2, 6, 1))
    problem = poisson(net, dim=2, n_interior=30, n_boundary=10, seed=0, n_test=200)
    theta = net.init(seed=0)
    error = problem.relative_error((theta,))
    assert error > 0.1
    assert problem.relative_error((theta, theta), (0.5, 0.5)) == pytest.approx(error)
    assert torch.allclose(product_sine(problem.test_points), torch.vmap(_exact)(problem.test_points))


def test_burgers_operator_on_a_polynomial() -> None:
    nu = 0.01 / math.pi
    problem = burgers(MLP((2, 4, 1)), n_interior=10, n_initial=6, n_boundary=6, seed=0)
    interior = problem.blocks[0]
    z = torch.tensor([0.4, 0.7])  # (x, t)
    value = interior.op(lambda y: y[0] ** 2 * y[1], z)
    # u = x^2 t: u_t + u u_x - nu u_xx = x^2 + 2 x^3 t^2 - 2 nu t
    expected = 0.4 ** 2 + 2 * 0.4 ** 3 * 0.7 ** 2 - 2 * nu * 0.7
    assert float(value) == pytest.approx(expected)
    assert not problem.linear
    assert problem.m == 10 + 6 + 6
    assert bool(torch.isfinite(problem.residual(MLP((2, 4, 1)).init(seed=0))).all())


def test_allen_cahn_operator_and_periodic_block() -> None:
    problem = allen_cahn(MLP((2, 4, 1)), n_interior=10, n_initial=6, n_boundary=6, seed=0)
    z = torch.tensor([0.3, 0.5])
    u = lambda y: torch.sin(math.pi * y[0]) * y[1]  # noqa: E731
    s = math.sin(0.3 * math.pi)
    expected = s + 1e-4 * math.pi ** 2 * s * 0.5 + 5 * (s * 0.5) ** 3 - 5 * s * 0.5
    assert float(problem.blocks[0].op(u, z)) == pytest.approx(expected)
    periodic = problem.blocks[2]
    # sin(pi x) t is periodic in value, its x-derivative is periodic as well
    assert torch.allclose(periodic.op(u, torch.tensor([0.5])), torch.zeros(2), atol=1e-12)
    assert problem.m == 10 + 6 + 2 * 6
    initial = problem.blocks[1]
    x0 = initial.points[0]
    assert float(problem.data[10]) == pytest.approx(
        math.sqrt(initial.weight) * float(x0[0] ** 2 * torch.sin(2 * math.pi * x0[0])))


@pytest.mark.unit
def test_every_face_of_the_boundary_is_covered() -> None:
    points, _ = box_boundary(1024, 2, seed=0)
    for axis in range(2):
        for side in (0.0, 1.0):
            tangential = points[points[:, axis] == side][:, 1 - axis]
            assert tangential.shape[0] == 256
            assert float(tangential.min()) < 0.02 and float(tangential.max()) > 0.98
            assert abs(float(tangential.mean()) - 0.5) < 0.02
    assert box_boundary(7, 2, seed=0)[0].shape == (7, 2)
    assert box_boundary(3, 1, seed=0)[0].shape == (3, 1)


@pytest.mark.unit
def test_burgers_boundary_times_cover_both_sides() -> None:
    from pla.problems.nonlinear import burgers
    pb = burgers(MLP((2, 4, 1)), n_interior=8, n_initial=4, n_boundary=256, seed=0)
    boundary = pb.blocks[2].points
    for side in (-1.0, 1.0):
        times = boundary[boundary[:, 0] == side][:, 1]
        assert times.shape[0] == 128
        assert float(times.min()) < 0.05 and float(times.max()) > 0.95
