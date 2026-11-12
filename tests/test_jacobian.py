import pytest
import torch

from pla.jacobian import jacobian, linearize
from pla.network import MLP
from pla.problems.linear import biharmonic, poisson
from pla.problems.nonlinear import allen_cahn

pytestmark = pytest.mark.unit


def _setup():
    net = MLP((2, 6, 6, 1))
    return poisson(net, dim=2, n_interior=24, n_boundary=8, seed=0), net.init(seed=0)


def test_jacobian_matches_reverse_mode_differentiation_of_the_residual() -> None:
    problem, theta = _setup()
    dense = jacobian(problem, theta)
    reference = torch.autograd.functional.jacobian(problem.residual, theta)
    assert dense.shape == (32, problem.net.num_params)
    assert torch.allclose(dense, reference, rtol=1e-10, atol=1e-12)


def test_jacobian_of_blocks_with_several_values_per_point() -> None:
    net = MLP((2, 5, 1))
    theta = net.init(seed=0)
    for problem in (biharmonic(net, n_interior=6, n_boundary=4, seed=0),
                    allen_cahn(net, n_interior=6, n_initial=4, n_boundary=4, seed=0)):
        dense = jacobian(problem, theta, chunk_size=3)
        reference = torch.autograd.functional.jacobian(problem.residual, theta)
        assert torch.allclose(dense, reference, rtol=1e-9, atol=1e-10)


def test_jacobian_precision_follows_the_parameters() -> None:
    problem, theta = _setup()
    single = jacobian(problem, theta.to(torch.float32))
    assert single.dtype == torch.float32
    assert torch.allclose(single.double(), jacobian(problem, theta), rtol=1e-3, atol=1e-3)


def test_matrix_free_products_agree_with_the_dense_jacobian() -> None:
    problem, theta = _setup()
    dense = jacobian(problem, theta)
    jv, jtw = linearize(problem, theta)
    v, w = torch.randn(problem.net.num_params), torch.randn(problem.m)
    assert torch.allclose(jv(v), dense @ v, rtol=1e-10, atol=1e-12)
    assert torch.allclose(jtw(w), dense.T @ w, rtol=1e-10, atol=1e-12)
    # the products can be repeated
    assert torch.allclose(jtw(2.0 * w), 2.0 * (dense.T @ w), rtol=1e-10, atol=1e-12)
