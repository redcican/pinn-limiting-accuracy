import math

import pytest
import torch

from pla.baselines.common import grid_line_search, step_grid
from pla.baselines.engd import ENGDConfig, engd, natural_gradient
from pla.baselines.first_order import FirstOrderConfig, adam_lbfgs
from pla.baselines.kfac import (KFACConfig, forward_laplacian, kfac, kfac_direction, kfac_factors,
                                solve_two_kronecker)
from pla.jacobian import jacobian
from pla.network import MLP
from pla.problems.linear import helmholtz, poisson, regression

NET = MLP((2, 8, 8, 1))


@pytest.fixture(scope="module")
def problem():
    return poisson(NET, dim=2, n_interior=100, n_boundary=40, seed=0, n_test=500)


@pytest.mark.unit
def test_grid_line_search_takes_the_best_step_and_never_increases_the_loss() -> None:
    loss = lambda t: float((t - 1.0) @ (t - 1.0))
    theta = torch.zeros(2)
    best, value = grid_line_search(loss, theta, -4.0 * torch.ones(2), loss(theta), step_grid(5))
    assert torch.allclose(best, torch.ones(2)) and value == 0.0  # step 1/4
    same, value = grid_line_search(loss, theta, torch.ones(2), loss(theta), step_grid(5))
    assert same is theta and value == loss(theta)
    with pytest.raises(ValueError):
        step_grid(0)


@pytest.mark.unit
def test_natural_gradient_is_the_gauss_newton_direction() -> None:
    j, r = torch.randn(30, 8), torch.randn(30)
    expected = torch.linalg.lstsq(j, r.unsqueeze(1)).solution.squeeze(1)
    assert torch.allclose(natural_gradient(j, r, dual=False), expected, atol=1e-10)
    wide, s = torch.randn(8, 30), torch.randn(8)
    primal, dual = natural_gradient(wide, s, dual=False), natural_gradient(wide, s, dual=True)
    assert torch.allclose(primal, torch.linalg.pinv(wide) @ s, atol=1e-8)
    assert torch.allclose(dual, primal, atol=1e-8)


@pytest.mark.unit
def test_forward_laplacian_reproduces_the_operator(problem) -> None:
    theta = NET.init(seed=3) + 0.05
    helm = helmholtz(NET, wavenumber=2.0, n_interior=20, n_boundary=8, seed=0, n_test=10)
    interior, boundary = helm.blocks
    u, lap, inputs, outputs = forward_laplacian(NET, theta, interior.points, derivatives=True)
    g, zero, _, _ = forward_laplacian(NET, theta, boundary.points, derivatives=False)
    values = torch.cat([math.sqrt(interior.weight) * (-lap - 4.0 * u), math.sqrt(boundary.weight) * g])
    assert torch.allclose(values, helm.operator(theta), rtol=1e-10, atol=1e-12)
    assert float(zero.abs().max()) == 0.0
    assert inputs[0].shape == (20, 4, 3) and outputs[-1].shape == (20, 4, 1)


@pytest.mark.unit
def test_two_term_kronecker_solve_matches_a_dense_solve() -> None:
    def spd(n: int) -> torch.Tensor:
        m = torch.randn(n, n)
        return m @ m.T + 0.1 * torch.eye(n)

    b1, a1, b2, a2, v = spd(3), spd(4), spd(3), spd(4), torch.randn(3, 4)
    x = solve_two_kronecker(b1, a1, b2, a2, v)
    assert torch.allclose(b1 @ x @ a1 + b2 @ x @ a2, v, atol=1e-9)
    dense = torch.kron(b1, a1) + torch.kron(b2, a2)  # row-major vectorization
    assert torch.allclose(x.reshape(-1), torch.linalg.solve(dense, v.reshape(-1)), atol=1e-9)


@pytest.mark.unit
def test_factors_are_exact_for_one_point_of_a_regression_term() -> None:
    """With one boundary point and no interior weight the Gauss-Newton block of
    a layer is the Kronecker product of the boundary factors."""
    net = MLP((2, 3, 1))
    pb = poisson(net, dim=2, n_interior=4, n_boundary=1, seed=0, n_test=10)
    theta = net.init(seed=0) + 0.1
    gradient, factors = kfac_factors(pb, theta)
    j = jacobian(pb, theta)
    assert torch.allclose(gradient, j.T @ pb.residual(theta), atol=1e-10)
    row = j[-1]  # the boundary point
    first = torch.cat([row[:6].reshape(3, 2), row[6:9].unsqueeze(1)], dim=1).reshape(-1)
    a, b = factors[0][1]
    assert torch.allclose(torch.kron(b, a), torch.outer(first, first), atol=1e-10)


@pytest.mark.unit
def test_kfac_gradient_uses_the_operator_of_the_problem() -> None:
    helm = helmholtz(NET, wavenumber=3.0, n_interior=30, n_boundary=12, seed=0, n_test=10)
    theta = NET.init(seed=2)
    gradient, _ = kfac_factors(helm, theta)
    assert torch.allclose(gradient, jacobian(helm, theta).T @ helm.residual(theta), atol=1e-9)


@pytest.mark.unit
def test_kfac_direction_is_a_descent_direction(problem) -> None:
    theta = NET.init(seed=0)
    gradient, factors = kfac_factors(problem, theta)
    direction = kfac_direction(NET, gradient, factors, 1e-8)
    assert direction.shape == gradient.shape
    assert float(direction @ gradient) > 0.0


@pytest.mark.integration
@pytest.mark.parametrize("dual", [False, True])
def test_engd_lowers_the_loss(problem, dual: bool) -> None:
    result = engd(problem, NET.init(seed=0), ENGDConfig(dual=dual, damping=1e-3, max_iter=15))
    assert result.losses[-1] < 1e-2 * result.losses[0]
    assert all(b <= a for a, b in zip(result.losses, result.losses[1:]))


@pytest.mark.integration
def test_kfac_lowers_the_loss(problem) -> None:
    result = kfac(problem, NET.init(seed=0), KFACConfig(max_iter=30))
    assert result.losses[-1] < 0.1 * result.losses[0]
    assert all(b <= a for a, b in zip(result.losses, result.losses[1:]))


@pytest.mark.integration
def test_adam_then_lbfgs_lowers_the_loss() -> None:
    net = MLP((1, 8, 1))
    pb = regression(net, dim=1, n_interior=40, seed=0, n_test=100)
    result = adam_lbfgs(pb, net.init(seed=0), FirstOrderConfig(adam_steps=50, adam_lr=1e-2, lbfgs_steps=30))
    assert len(result.losses) == 81
    assert result.losses[50] < result.losses[0]
    assert result.losses[-1] < 0.1 * result.losses[50]
    assert not result.theta.requires_grad


@pytest.mark.unit
def test_invalid_baseline_settings() -> None:
    with pytest.raises(ValueError):
        ENGDConfig(damping=-1.0)
    with pytest.raises(ValueError):
        KFACConfig(damping=0.0)
    with pytest.raises(ValueError):
        FirstOrderConfig(adam_lr=0.0)
    with pytest.raises(ValueError):
        kfac_factors(regression(MLP((1, 4, 1)), dim=1, n_interior=8, seed=0, n_test=10),
                     MLP((1, 4, 1)).init(seed=0))
