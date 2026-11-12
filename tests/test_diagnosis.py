import math

import pytest
import torch

from pla.diagnosis import Cause, diagnose, fit_beta, residual_split, thresholds

pytestmark = pytest.mark.unit


def _jacobian(m: int = 80, p: int = 30):
    u, _ = torch.linalg.qr(torch.randn(m, m))
    v, _ = torch.linalg.qr(torch.randn(p, p))
    sigma = torch.logspace(0.0, -12.0, p)
    return (u[:, :p] * sigma) @ v.T, u, sigma


def test_split_recovers_constructed_shares() -> None:
    j, u, sigma = _jacobian()
    eta = 1e-6
    strong = sigma >= eta
    coef = torch.where(strong, 2.0, 3.0) / math.sqrt(30)
    r = u[:, :30] @ (coef * torch.sign(torch.randn(30))) + 6.0 * u[:, 40]
    split = residual_split(j, r, eta)
    n_strong = int(strong.sum())
    norm = math.sqrt(4.0 * n_strong / 30 + 9.0 * (30 - n_strong) / 30 + 36.0)
    # Singular vectors that belong to relative singular values near 1e-12 are
    # resolved only to about 1e-4 in double precision, hence the tolerance.
    assert split.e_perp == pytest.approx(6.0 / norm, rel=1e-5)
    assert split.e_opt == pytest.approx(math.sqrt(4.0 * n_strong / 30) / norm, rel=1e-5)
    assert split.e_prec == pytest.approx(math.sqrt(9.0 * (30 - n_strong) / 30) / norm, rel=1e-5)
    assert split.e_perp ** 2 + split.e_prec ** 2 + split.e_opt ** 2 == pytest.approx(1.0)


def test_fit_recovers_the_exponent_of_a_power_law_profile() -> None:
    sigma_rel = torch.logspace(0.0, -14.0, 200)
    beta = 0.8
    # g(eta)^2 = sum of squared coefficients below eta = eta^(2 beta)
    g2 = sigma_rel ** (2 * beta)
    coef = torch.sqrt(torch.cat([g2[:-1] - g2[1:], g2[-1:]]))
    # coefficient i carries the mass between sigma_i and sigma_{i+1}; shift by one
    # so that the mass strictly below sigma_i equals sigma_i^(2 beta)
    shifted = torch.cat([torch.zeros(1), coef[:-1]])
    estimate = fit_beta(sigma_rel, shifted, 1e-12, 1e-4)
    assert estimate == pytest.approx(beta, abs=0.02)


def test_thresholds_follow_the_unit_roundoff() -> None:
    values = thresholds(tau=0.5)
    assert values["ne64"] == pytest.approx(math.sqrt(2.0 ** -53))
    assert values["qr32"] == pytest.approx(2.0 ** -24 / 0.5)
    assert values["qr64"] == pytest.approx(2.0 ** -53 / 0.5)
    assert thresholds(tau=0.5, c_ne=4.0)["ne64"] == pytest.approx(2.0 * math.sqrt(2.0 ** -53))


@pytest.mark.parametrize("dominant,expected", [("perp", Cause.TANGENT_SPACE),
                                               ("weak", Cause.PRECISION),
                                               ("strong", Cause.OPTIMIZATION)])
def test_cause_is_the_largest_term_at_the_solver_threshold(dominant: str, expected: Cause) -> None:
    j, u, sigma = _jacobian()
    weak = sigma < math.sqrt(2.0 ** -53)
    coef = torch.full((30,), 1e-3)
    r = u[:, :30] @ coef + 1e-3 * u[:, 50]
    if dominant == "perp":
        r = r + u[:, 50]
    elif dominant == "weak":
        r = r + u[:, :30] @ weak.double()
    else:
        r = r + u[:, :30] @ (~weak).double()
    result = diagnose(j, r, solver="ne64", tau=0.5)
    assert result.cause is expected
    assert result.k_bad == int((sigma < 2.0 ** -24 / 0.5).sum())
    assert result.picard.shape == (30,) and result.sigma_rel[0] == 1.0
    assert set(result.splits) == {"ne64", "qr32", "qr64"}


def test_unknown_solver_threshold_is_rejected() -> None:
    j, u, _ = _jacobian()
    with pytest.raises(KeyError):
        diagnose(j, u[:, 0], solver="lu", tau=0.5)


@pytest.mark.unit
def test_rank_tolerance_assigns_null_directions_to_the_tangent_orthogonal_part() -> None:
    import torch
    j = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    r = torch.tensor([1.0, -1.0], dtype=torch.float64)
    split = residual_split(j, r, 1e-8, rank_tol=2 * 2.0 ** -53)
    assert split.e_perp == pytest.approx(1.0) and split.e_prec == pytest.approx(0.0, abs=1e-12)
    result = diagnose(j, r, solver="qr64", tau=0.5, rank_tol=2 * 2.0 ** -53)
    assert result.cause is Cause.TANGENT_SPACE


@pytest.mark.unit
def test_profile_fit_does_not_depend_on_the_order_of_equal_singular_values() -> None:
    import math
    import torch
    sigma = torch.tensor([1.0, 1e-4, 1e-4, 1e-6, 1e-8], dtype=torch.float64)
    first = torch.tensor([0.0, 1.0, 100.0, 0.01, 1e-4], dtype=torch.float64)
    second = torch.tensor([0.0, 100.0, 1.0, 0.01, 1e-4], dtype=torch.float64)
    assert fit_beta(sigma, first, 1e-7, 1e-3) == pytest.approx(fit_beta(sigma, second, 1e-7, 1e-3))
    tied = torch.tensor([1e-6, 1e-6, 1e-6], dtype=torch.float64)
    assert math.isnan(fit_beta(tied, torch.ones(3, dtype=torch.float64), 1e-8, 1e-4))
