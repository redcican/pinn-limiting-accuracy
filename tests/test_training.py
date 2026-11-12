"""End-to-end tests on small problems (CPU, a few seconds each).

Two problems are used: a Poisson problem in two variables for the mechanics of
the loops, and a regression problem in one variable, on which single precision
stalls within a few hundred iterations, for the effects of precision.
"""
import math

import pytest
import torch

from pla.jacobian import jacobian
from pla.lm import LMConfig, has_stalled, levenberg_marquardt
from pla.mpgn import MPGNConfig, damping_floor, mixed_precision_step, mpgn
from pla.multistage import multistage, sum_residual
from pla.network import MLP
from pla.problems.linear import poisson, regression
from pla.problems.nonlinear import burgers
from pla.solvers import Solver, lm_step

pytestmark = pytest.mark.integration

NET = MLP((2, 10, 10, 1))
TINY = MLP((1, 6, 1))
SINGLE = dict(bits_r=24, bits_j=24, bits_s=24)


@pytest.fixture(scope="module")
def problem():
    return poisson(NET, dim=2, n_interior=300, n_boundary=80, seed=0, n_test=2000)


@pytest.fixture(scope="module")
def tiny():
    return regression(TINY, dim=1, n_interior=40, seed=0, n_test=500)


@pytest.fixture(scope="module")
def tiny_single(tiny):
    return levenberg_marquardt(tiny, TINY.init(seed=0),
                               LMConfig(solver=Solver.QR, max_iter=600, **SINGLE))


@pytest.fixture(scope="module")
def tiny_double(tiny):
    return levenberg_marquardt(tiny, TINY.init(seed=0), LMConfig(solver=Solver.QR, max_iter=1000))


def test_lm_qr_reduces_the_loss(problem) -> None:
    result = levenberg_marquardt(problem, NET.init(seed=0), LMConfig(solver=Solver.QR, max_iter=60))
    assert result.losses[-1] < 1e-3 * result.losses[0]
    assert all(b <= a for a, b in zip(result.losses, result.losses[1:]))
    assert result.theta.dtype == torch.float64
    assert 0 < result.accepted <= 60
    assert problem.relative_error((result.theta,)) < 0.1


@pytest.mark.parametrize("solver", [Solver.NE, Solver.DUAL])
def test_gram_type_solvers_train_in_double_precision(problem, solver: Solver) -> None:
    result = levenberg_marquardt(problem, NET.init(seed=0), LMConfig(solver=solver, max_iter=40))
    assert result.losses[-1] < 1e-2 * result.losses[0]


def test_single_precision_run_stalls_above_the_double_precision_one(tiny_single, tiny_double) -> None:
    assert tiny_single.theta.dtype == torch.float32
    assert tiny_single.stalled
    assert len(tiny_single.losses) < 601
    assert tiny_double.losses[-1] < 0.5 * tiny_single.losses[-1]


def test_stall_test() -> None:
    assert not has_stalled([1.0] * 20, 20, 1e-3)  # too short
    assert has_stalled([1.0] * 21, 20, 1e-3)
    assert not has_stalled([1.0] * 20 + [0.9], 20, 1e-3)


def test_emulated_solver_precision_is_accepted(problem) -> None:
    config = LMConfig(solver=Solver.QR, bits_s=30, max_iter=6)
    result = levenberg_marquardt(problem, NET.init(seed=0), config)
    assert result.losses[-1] < result.losses[0]


def test_failed_steps_raise_the_damping(problem) -> None:
    """Normal equations in single precision with tiny damping fail at first;
    the loop must recover by increasing the damping."""
    config = LMConfig(solver=Solver.NE, bits_s=24, mu0=1e-8, mu_min=1e-8, max_iter=40)
    result = levenberg_marquardt(problem, NET.init(seed=0), config)
    assert result.losses[1] == result.losses[0]
    assert result.mu > 1e-8
    assert result.losses[-1] < result.losses[0]


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError):
        LMConfig(bits_r=30)
    with pytest.raises(ValueError):
        LMConfig(nu=1.0)
    with pytest.raises(ValueError):
        LMConfig(mu0=1e-40)
    with pytest.raises(ValueError):
        MPGNConfig(tau=0.7)
    with pytest.raises(ValueError):
        MPGNConfig(omega=0.9, tau=0.5)
    with pytest.raises(ValueError):
        MPGNConfig(nu=0.5)


def test_damping_floor_formula() -> None:
    eps32, eps64 = 2.0 ** -24, 2.0 ** -53
    expected = (eps64 / 0.5) ** 2 * 3.0 ** 2 / (1.0 - eps32) ** 2
    assert damping_floor(3.0, tau=0.5, c_qr=1.0) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_mixed_precision_step_matches_the_double_precision_step(problem) -> None:
    """The refined step agrees with the step from QR in double precision."""
    theta = levenberg_marquardt(problem, NET.init(seed=0),
                                LMConfig(solver=Solver.QR, max_iter=10)).theta
    mu = 1e-10
    r = problem.residual(theta)
    j = jacobian(problem, theta)
    exact = lm_step(j, r, mu, Solver.QR, 53)
    step = mixed_precision_step(problem, theta, r, mu, MPGNConfig(omega=1e-7, max_inner=400))
    delta, factor_norm = step.delta, step.factor_norm
    assert step.inner.converged and step.mu == mu
    sigma_1 = float(torch.linalg.matrix_norm(j, ord=2))
    assert factor_norm == pytest.approx(sigma_1, rel=1e-3)
    difference = delta - exact
    energy = torch.linalg.norm(j @ difference) ** 2 + mu * torch.linalg.norm(difference) ** 2
    assert math.sqrt(float(energy)) <= 1e-6 * float(torch.linalg.norm(r))


def test_mpgn_passes_the_single_precision_floor(tiny, tiny_single) -> None:
    result = mpgn(tiny, TINY.init(seed=0),
                  MPGNConfig(max_outer=400, phase1=LMConfig(solver=Solver.QR, max_iter=600, **SINGLE)))
    assert result.theta.dtype == torch.float64
    assert result.switch_iteration == len(tiny_single.losses) - 1
    assert len(result.inner_iterations) == len(result.losses) - result.switch_iteration - 1
    assert min(result.inner_iterations) >= 1
    assert tiny.loss(result.theta) < 0.5 * tiny.loss(tiny_single.theta.double())


def test_damping_below_the_floor_is_raised_before_the_solve(problem) -> None:
    theta = NET.init(seed=0)
    step = mixed_precision_step(problem, theta, problem.residual(theta), 1e-100, MPGNConfig())
    assert step.mu == pytest.approx(damping_floor(step.factor_norm, tau=0.5), rel=1e-6, abs=0.0)
    assert bool(torch.isfinite(step.delta).all())


def test_unconverged_inner_iterations_are_counted(problem) -> None:
    result = mpgn(problem, NET.init(seed=0),
                  MPGNConfig(phase1=None, mu0=1e-12, omega=1e-12, max_inner=1, max_outer=3))
    assert result.inner_failures == 3


def test_mpgn_without_first_phase(problem) -> None:
    result = mpgn(problem, NET.init(seed=0), MPGNConfig(phase1=None, max_outer=12))
    assert result.switch_iteration == 0
    assert result.losses[-1] < result.losses[0]


def test_second_stage_lowers_the_residual_of_the_sum(tiny) -> None:
    def trainer(stage_problem, theta0):
        return levenberg_marquardt(stage_problem, theta0,
                                   LMConfig(solver=Solver.QR, max_iter=200)).theta

    result = multistage(tiny, trainer, n_stages=2, seed=0)
    assert len(result.thetas) == 2 and len(result.alphas) == 2
    assert result.alphas[0] == pytest.approx(float(torch.linalg.norm(tiny.data)))
    z0, z1, z2 = result.residual_norms
    assert z0 == pytest.approx(result.alphas[0])
    assert result.alphas[1] == pytest.approx(z1)
    assert z2 < z1 < z0
    z = sum_residual(tiny, result.thetas, result.alphas)
    assert float(torch.linalg.norm(z)) == pytest.approx(z2)
    assert tiny.relative_error(result.thetas, result.alphas) < tiny.relative_error(result.thetas[:1])


def test_multistage_rejects_invalid_input(tiny) -> None:
    nonlinear = burgers(MLP((2, 4, 1)), n_interior=8, n_initial=4, n_boundary=4, seed=0)
    with pytest.raises(ValueError):
        multistage(nonlinear, lambda p, t: t, n_stages=1, seed=0)
    with pytest.raises(ValueError):
        multistage(tiny, lambda p, t: t, n_stages=0, seed=0)
    with pytest.raises(ValueError):
        sum_residual(tiny, [TINY.init(seed=0)], [])
