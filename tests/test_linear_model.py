import pytest
import torch

from pla.linear_model import (LinearModelConfig, build, cg_iterations, exception_count,
                              filtered_components, preconditioned_matrix, predicted_count,
                              weak_residual)
from pla.solvers import Solver

pytestmark = pytest.mark.unit

CONFIG = LinearModelConfig(m=150, p=50, log_kappa=10.0)
ZETA = 0.5


def test_model_has_the_prescribed_spectrum() -> None:
    model = build(CONFIG)
    assert torch.allclose(torch.linalg.svdvals(model.j)[:10], model.sigma[:10], rtol=1e-8)
    assert float(model.sigma[0] / model.sigma[-1]) == pytest.approx(1e10)
    assert torch.linalg.norm(model.r) == pytest.approx(1.0)
    assert torch.linalg.norm(model.r - model.u @ (model.u.T @ model.r)) <= 1e-12


def test_failed_step_is_reported_as_none() -> None:
    model = build(CONFIG)
    assert filtered_components(model, 1e-5, Solver.NE, 24) is None
    assert filtered_components(model, 1e-5, Solver.QR, 24).shape == (50,)


def test_preconditioner_is_accurate_for_strong_damping() -> None:
    """Theorem 4.1(a) with k = 0: all singular values of B lie near one."""
    model = build(CONFIG)
    eta_mu = 1e-5
    s = torch.linalg.svdvals(preconditioned_matrix(model, eta_mu))
    assert predicted_count(model, eta_mu, ZETA) == 0
    assert exception_count(s, ZETA) == 0


def test_exceptions_are_bounded_by_the_predicted_count() -> None:
    """Theorem 4.1(a): at most k singular values of B lie on either side of the interval."""
    model = build(CONFIG)
    for eta_mu in (1e-8, 1e-9, 1e-10):
        s = torch.linalg.svdvals(preconditioned_matrix(model, eta_mu))
        k = predicted_count(model, eta_mu, ZETA)
        assert int((s < 1.0 - ZETA).sum()) <= k
        assert int((s > 1.0 + ZETA).sum()) <= k
        assert float(s[-1]) >= eta_mu / 2.0


def test_inner_iterations_grow_with_the_number_of_exceptions() -> None:
    model = build(CONFIG)
    rhs_r = weak_residual(model, 1e-6, seed=3)
    assert torch.linalg.norm(rhs_r) == pytest.approx(1.0)
    counts = []
    for eta_mu in (1e-5, 1e-10):
        b_mat = preconditioned_matrix(model, eta_mu)
        rhs = torch.cat([rhs_r, torch.zeros(50)])
        counts.append(cg_iterations(b_mat, rhs, 1e-3, 500))
    assert counts[0] <= 10
    assert counts[1] > counts[0]
