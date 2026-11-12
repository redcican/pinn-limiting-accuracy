import math

import pytest
import torch

from pla.network import MLP
from pla.problems.linear import helmholtz, poisson, regression
from pla.problems.operators import gradient, hessian
from pla.rounding_bound import (TANH_BOUNDS, cancellation_scale, evaluation_floor, gamma,
                                propagate)

pytestmark = pytest.mark.unit


def _representable(problem):
    """The same problem with points that are exact in single precision."""
    blocks = tuple(type(b)(b.name, b.points.float().double(), b.weight, b.op, b.width)
                   for b in problem.blocks)
    return type(problem)(problem.net, blocks, problem.data, problem.linear,
                         coefficients=problem.coefficients)


def test_gamma() -> None:
    eps = 2.0 ** -24
    assert gamma(10, 24) == pytest.approx(10 * eps / (1 - 10 * eps))
    with pytest.raises(ValueError):
        gamma(2 ** 24, 24)


def test_bounds_of_the_hyperbolic_tangent_and_its_derivatives() -> None:
    a = torch.linspace(-20.0, 20.0, 80001, dtype=torch.float64)
    t = torch.tanh(a)
    s = 1.0 - t ** 2
    derivatives = [t, s, -2.0 * t * s, -2.0 * s ** 2 + 4.0 * t ** 2 * s]
    for bound, values in zip(TANH_BOUNDS, derivatives):
        assert float(values.abs().max()) == pytest.approx(bound, rel=1e-6)


def test_propagated_derivatives_match_automatic_differentiation() -> None:
    net = MLP((2, 7, 5, 1))
    theta = net.init(seed=1) + 0.1
    x = torch.rand(6, 2, dtype=torch.float64)
    out = propagate(net, theta, x)
    u = lambda z: net(theta, z)[0]
    for i in range(x.shape[0]):
        assert float(out.value.value[i]) == pytest.approx(float(u(x[i])), rel=1e-12)
        grad, hess = gradient(u, x[i]), hessian(u, x[i])
        for k in range(2):
            assert float(out.first[k].value[i]) == pytest.approx(float(grad[k]), rel=1e-10)
            assert float(out.second[k].value[i]) == pytest.approx(float(hess[k, k]), rel=1e-10)


def test_majorants_dominate_the_values() -> None:
    net = MLP((3, 6, 6, 1))
    theta = net.init(seed=0) - 0.2
    out = propagate(net, theta, torch.rand(10, 3, dtype=torch.float64) - 0.5)
    for tracked in (out.value, *out.first, *out.second):
        assert bool((tracked.majorant >= tracked.value.abs()).all())


def test_counts_of_a_network_with_one_hidden_layer() -> None:
    d, n, c_act = 2, 5, 3
    net = MLP((d, n, 1))
    out = propagate(net, net.init(seed=0), torch.rand(4, d, dtype=torch.float64), c_act=c_act)
    argument = d + 1                    # d products, d - 1 sums, one bias
    assert out.value.count == argument + c_act + n + 1
    # da/dx_k is a weight (exact); dh = psi'(a) * da; du = sum over n products
    assert out.first[0].count == (argument + c_act + 1) + n


def test_single_precision_error_lies_below_the_bound() -> None:
    net = MLP((2, 8, 8, 1))
    problem = _representable(helmholtz(net, wavenumber=3.0, n_interior=64, n_boundary=16,
                                       seed=0, n_test=10))
    theta = net.init(seed=0).float().double()
    bound = cancellation_scale(problem, theta)
    error = (problem.residual(theta.float()).double() - problem.residual(theta)).abs()
    limit = gamma(bound.count, 24) * bound.rho
    assert bound.rho.shape == (problem.m,)
    assert bool((error <= limit).all())
    assert float(error.max()) > 0.0
    assert evaluation_floor(bound, 24) == pytest.approx(
        gamma(bound.count, 24) * float(torch.linalg.norm(bound.rho)))
    assert evaluation_floor(bound, 53) < evaluation_floor(bound, 24)


def test_cancellation_scale_exceeds_the_data_and_does_not_vanish_with_the_residual() -> None:
    net = MLP((2, 6, 1))
    problem = poisson(net, dim=2, n_interior=20, n_boundary=8, seed=0, n_test=10)
    bound = cancellation_scale(problem, net.init(seed=0))
    assert bool((bound.rho >= problem.data.abs() - 1e-15).all())
    assert bool((bound.rho >= problem.residual(net.init(seed=0)).abs()).all())


def test_unsupported_problems_are_rejected() -> None:
    net = MLP((1, 4, 1))
    with pytest.raises(ValueError):
        cancellation_scale(regression(net, dim=1, n_interior=8, seed=0, n_test=10), net.init(seed=0))
    with pytest.raises(ValueError):
        propagate(MLP((2, 4, 2)), MLP((2, 4, 2)).init(seed=0), torch.rand(3, 2, dtype=torch.float64))


def test_the_wavenumber_is_taken_from_the_problem() -> None:
    net = MLP((2, 6, 1))
    theta = net.init(seed=0)
    plain = poisson(net, dim=2, n_interior=16, n_boundary=8, seed=0, n_test=10)
    shifted = helmholtz(net, wavenumber=5.0, n_interior=16, n_boundary=8, seed=0, n_test=10)
    rho_plain, rho_shifted = cancellation_scale(plain, theta).rho, cancellation_scale(shifted, theta).rho
    assert bool((rho_shifted[:16] > rho_plain[:16]).all())
    assert cancellation_scale(shifted, theta).count >= cancellation_scale(plain, theta).count
