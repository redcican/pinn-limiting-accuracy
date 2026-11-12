import pytest
import torch

from pla.linear_model import LinearModelConfig, build
from pla.lsqr import lsqr, preconditioned_lsqr
from pla.solvers import triangular_factor

pytestmark = pytest.mark.unit


def test_lsqr_matches_a_dense_least_squares_solve() -> None:
    a = torch.randn(60, 20)
    b = torch.randn(60)
    result = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 20, omega=1e-12, max_iter=200)
    exact = torch.linalg.lstsq(a, b).solution
    assert result.converged
    assert torch.linalg.norm(result.x - exact) <= 1e-9 * torch.linalg.norm(exact)
    residual = b - a @ result.x
    assert torch.linalg.norm(a.T @ residual) <= 1e-10 * torch.linalg.norm(residual)


def test_lsqr_stops_by_the_forcing_term_and_reports_its_estimates() -> None:
    a = torch.randn(80, 30)
    b = torch.randn(80)
    loose = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 30, omega=1e-1, max_iter=200)
    tight = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 30, omega=1e-8, max_iter=200)
    assert loose.iterations < tight.iterations
    residual = b - a @ loose.x
    true_ratio = torch.linalg.norm(a.T @ residual) / torch.linalg.norm(residual)
    assert true_ratio <= 1.5e-1
    assert loose.rnorm == pytest.approx(float(torch.linalg.norm(residual)), rel=1e-6)


def test_singular_value_estimate_bounds_the_smallest_singular_value_from_above() -> None:
    a = torch.randn(50, 15)
    b = torch.randn(50)
    result = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 15, omega=1e-10, tau=0.5, max_iter=100)
    s_min = float(torch.linalg.svdvals(a)[-1])
    assert result.s_hat >= s_min * (1.0 - 1e-8)
    assert result.s_hat <= float(torch.linalg.svdvals(a)[0])


def test_safeguard_is_inactive_before_the_estimate_has_converged() -> None:
    """With B = diag(1, 1e-4) and right-hand side (1, 1), the forcing test passes
    after one iteration although the weak direction is uncorrected. The estimate
    from one bidiagonalization step is close to one, so the safeguard does not
    catch this case: it is rigorous only once the estimate has converged."""
    a = torch.diag(torch.tensor([1.0, 1e-4]))
    b = torch.ones(2)
    plain = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 2, omega=1e-3, max_iter=10)
    safe = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 2, omega=1e-3, tau=0.5, max_iter=10)
    assert plain.iterations == 1 and safe.iterations == 1
    assert safe.s_hat == pytest.approx(1.0, rel=1e-6)
    assert torch.linalg.norm(a @ safe.x - b) > 0.9


def test_safeguard_tightens_the_test_once_a_small_singular_value_is_seen() -> None:
    u, _ = torch.linalg.qr(torch.randn(40, 8))
    v, _ = torch.linalg.qr(torch.randn(8, 8))
    sigma = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 1e-2])
    a = (u * sigma) @ v.T
    b = torch.randn(40)
    plain = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 8, omega=0.2, max_iter=50)
    safe = lsqr(lambda x: a @ x, lambda y: a.T @ y, b, 8, omega=0.2, tau=0.5, max_iter=50)
    assert safe.iterations >= plain.iterations
    assert safe.arnorm <= min(0.2, 0.5 * safe.s_hat) * safe.rnorm * (1.0 + 1e-12)


def test_zero_right_hand_side_returns_zero() -> None:
    a = torch.randn(10, 4)
    result = lsqr(lambda x: a @ x, lambda y: a.T @ y, torch.zeros(10), 4, omega=1e-3, max_iter=5)
    assert result.iterations == 0 and result.converged
    assert torch.count_nonzero(result.x) == 0


def test_single_precision_factor_preconditions_the_damped_problem() -> None:
    """Theorem 4.1: the refined step is not subject to a single-precision threshold."""
    model = build(LinearModelConfig(m=150, p=50, log_kappa=10.0))
    eta_mu = 1e-5
    mu = eta_mu ** 2
    r_hat = triangular_factor(model.j, mu, 24)
    result = preconditioned_lsqr(lambda v: model.j @ v, lambda w: model.j.T @ w, r_hat, mu,
                                 model.r, omega=1e-10, tau=0.5, max_iter=200)
    coef = model.sigma / (model.sigma ** 2 + mu) * (model.u.T @ model.r)
    exact = -(model.v @ coef)
    assert result.converged
    assert result.iterations <= 30
    j_mu = torch.cat([model.j, eta_mu * torch.eye(50)])
    assert torch.linalg.norm(j_mu @ (result.x - exact)) <= 1e-8 * torch.linalg.norm(model.r)


@pytest.mark.unit
def test_convergence_is_verified_with_recomputed_norms() -> None:
    """With a forcing term below what double precision can deliver, the
    recurrence estimates still fall below it; the flag must report the truth."""
    import torch
    from pla.lsqr import lsqr
    a = torch.diag(torch.tensor([1.0, 1e-12], dtype=torch.float64))
    a = torch.cat([a, torch.zeros(1, 2, dtype=torch.float64)])
    b = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    result = lsqr(lambda v: a @ v, lambda w: a.T @ w, b, 2, omega=1e-12, tau=0.5, max_iter=100)
    residual = b - a @ result.x
    explicit = float(torch.linalg.norm(a.T @ residual))
    assert result.arnorm == pytest.approx(explicit, rel=1e-9, abs=1e-300)
    assert result.rnorm == pytest.approx(float(torch.linalg.norm(residual)))
    bound = min(1e-12, 0.5 * result.s_hat) if result.s_hat == result.s_hat else 1e-12
    assert result.converged == (explicit <= bound * result.rnorm)


@pytest.mark.unit
def test_consistent_system_is_reported_as_solved() -> None:
    import torch
    from pla.lsqr import lsqr
    a = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
    b = torch.ones(2, dtype=torch.float64)
    result = lsqr(lambda v: a @ v, lambda w: a.T @ w, b, 2, omega=1e-3, tau=0.5, max_iter=10)
    assert result.converged and result.iterations <= 3
    assert torch.allclose(result.x, torch.tensor([1.0, 0.5], dtype=torch.float64))
