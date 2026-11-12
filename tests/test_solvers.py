import pytest
import torch

from pla.linear_model import LinearModelConfig, build
from pla.solvers import Solver, StepFailure, lm_step, triangular_factor

pytestmark = pytest.mark.unit

CONFIG = LinearModelConfig(m=150, p=50, log_kappa=10.0)
TAU = 0.5


def _exact_step(model, mu: float) -> torch.Tensor:
    coef = model.sigma / (model.sigma ** 2 + mu) * (model.u.T @ model.r)
    return -(model.v @ coef)


@pytest.mark.parametrize("solver", list(Solver))
def test_double_precision_steps_agree_with_the_filter_formula(solver: Solver) -> None:
    model = build(CONFIG)
    mu = (1e-3 * float(model.sigma[0])) ** 2
    delta = lm_step(model.j, model.r, mu, solver, 53)
    exact = _exact_step(model, mu)
    assert torch.linalg.norm(delta - exact) <= 1e-6 * torch.linalg.norm(exact)


def test_single_precision_qr_acts_as_a_spectral_filter() -> None:
    """Theorem 3.1, case (O): cleared above the band, almost unchanged below it."""
    model = build(CONFIG)
    eta_mu = 1e-5
    delta = lm_step(model.j, model.r, eta_mu ** 2, Solver.QR, 24)
    before = model.u.T @ model.r
    after = model.u.T @ (model.r + model.j @ delta)
    above = model.sigma >= eta_mu / TAU
    far_above = model.sigma >= 1e3 * eta_mu
    assert (model.sigma[far_above] ** 2 / eta_mu ** 2).min() >= 1e6
    below = model.sigma < eta_mu * TAU
    assert torch.linalg.norm(after[above]) <= 10.0 * TAU * torch.linalg.norm(model.r)
    # Far above the band the exact filter is below 1e-6; what remains is the
    # backward error of the factorization, of order u / eta_mu.
    assert after[far_above].abs().max() <= 10.0 * 2.0 ** -24 / eta_mu
    change = torch.linalg.norm(after[below] - before[below])
    assert change <= 10.0 * TAU * torch.linalg.norm(model.r)
    assert (after[below] / before[below] - 1.0).abs().max() <= 0.3


def test_single_precision_normal_equations_fail_below_their_threshold() -> None:
    """Lemma 2.2(b): u * kappa(G) is about 600 here, and Cholesky breaks down."""
    model = build(CONFIG)
    with pytest.raises(StepFailure):
        lm_step(model.j, model.r, 1e-5 ** 2, Solver.NE, 24)


@pytest.mark.parametrize("solver", [Solver.NE, Solver.DUAL])
def test_single_precision_gram_solvers_work_above_their_threshold(solver: Solver) -> None:
    model = build(CONFIG)
    eta_mu = 3e-3
    delta = lm_step(model.j, model.r, eta_mu ** 2, solver, 24)
    after = model.u.T @ (model.r + model.j @ delta)
    assert after[model.sigma >= 1e2 * eta_mu].abs().max() <= 1e-2


def test_emulated_precision_step_lies_between_the_native_ones() -> None:
    model = build(CONFIG)
    mu = 1e-4 ** 2
    exact = _exact_step(model, mu)
    errors = [float(torch.linalg.norm(lm_step(model.j, model.r, mu, Solver.QR, bits) - exact))
              for bits in (24, 36, 53)]
    assert errors[0] > errors[1] > errors[2]


def test_triangular_factor_matches_the_augmented_matrix() -> None:
    model = build(CONFIG)
    mu = 1e-2
    r_hat = triangular_factor(model.j, mu, 53)
    gram = model.j.T @ model.j + mu * torch.eye(CONFIG.p)
    assert torch.allclose(r_hat.T @ r_hat, gram, rtol=1e-9, atol=1e-12)


def test_negative_damping_and_zero_damping_for_the_dual_form_are_rejected() -> None:
    model = build(CONFIG)
    with pytest.raises(ValueError):
        lm_step(model.j, model.r, -1.0, Solver.QR, 53)
    with pytest.raises(ValueError):
        lm_step(model.j, model.r, 0.0, Solver.DUAL, 53)
