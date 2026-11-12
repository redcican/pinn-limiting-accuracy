import pytest
import torch

from pla.linalg.cholesky import CholeskyBreakdown, cholesky_solve
from pla.linalg.householder import lstsq_qr
from pla.precision import chop

pytestmark = pytest.mark.unit


def _problem(m: int = 40, p: int = 12, log_kappa: float = 2.0):
    u, _ = torch.linalg.qr(torch.randn(m, p))
    v, _ = torch.linalg.qr(torch.randn(p, p))
    sigma = torch.logspace(0.0, -log_kappa, p)
    return (u * sigma) @ v.T, torch.randn(m)


@pytest.mark.parametrize("bits", [53, 24])
def test_native_qr_solves_least_squares(bits: int) -> None:
    a, b = _problem()
    x, r = lstsq_qr(a, b, bits)
    exact = torch.linalg.lstsq(a, b).solution
    tol = 1e-10 if bits == 53 else 1e-3
    assert torch.linalg.norm(x - exact) <= tol * torch.linalg.norm(exact)
    assert x.dtype == torch.float64 and r.shape == (12, 12)
    assert torch.allclose(r, torch.triu(r))


def test_emulated_qr_at_53_bits_matches_native() -> None:
    a, b = _problem()
    x_native, r_native = lstsq_qr(a, b, 53)
    x_loop, r_loop = lstsq_qr(a, b, 53, force_loop=True)
    assert torch.allclose(x_loop, x_native, rtol=1e-9, atol=1e-12)
    assert torch.allclose(r_loop.abs(), r_native.abs(), rtol=1e-9, atol=1e-12)


def test_emulated_qr_error_scales_with_the_unit_roundoff() -> None:
    a, b = _problem()
    exact = torch.linalg.lstsq(a, b).solution
    errors = []
    for bits in (11, 20, 30, 40):
        x, r = lstsq_qr(a, b, bits)
        assert torch.equal(r, chop(r, bits))
        errors.append(float(torch.linalg.norm(x - exact) / torch.linalg.norm(exact)))
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[3] < 1e-8 and errors[0] < 1.0


@pytest.mark.parametrize("bits", [53, 24, 30])
def test_cholesky_solves_a_well_conditioned_system(bits: int) -> None:
    a, _ = _problem()
    g = a.T @ a + 0.1 * torch.eye(12)
    rhs = torch.randn(12)
    x = cholesky_solve(g, rhs, bits)
    exact = torch.linalg.solve(g, rhs)
    tol = {53: 1e-10, 24: 1e-3, 30: 1e-5}[bits]
    assert torch.linalg.norm(x - exact) <= tol * torch.linalg.norm(exact)


@pytest.mark.parametrize("bits", [24, 20])
def test_cholesky_breaks_down_when_the_matrix_is_numerically_indefinite(bits: int) -> None:
    a, _ = _problem(log_kappa=7.0)
    g = a.T @ a  # condition number 1e14, far above 1/u for both formats
    with pytest.raises(CholeskyBreakdown):
        cholesky_solve(g, torch.randn(12), bits)


@pytest.mark.unit
def test_forced_loops_round_to_single_precision() -> None:
    import torch
    from pla.linalg.cholesky import cholesky_solve
    from pla.linalg.householder import lstsq_qr, qr_factor
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    b = torch.randn(12, generator=generator, dtype=torch.float64)
    x_loop, r_loop = lstsq_qr(a, b, 24, force_loop=True)
    x_native, r_native = lstsq_qr(a, b, 24)
    assert torch.equal(r_loop, r_loop.float().double()) and torch.equal(x_loop, x_loop.float().double())
    assert not torch.equal(x_loop, lstsq_qr(a, b, 53, force_loop=True)[0])
    assert torch.allclose(x_loop, x_native, rtol=1e-4, atol=1e-5)
    assert torch.allclose(r_loop.abs(), r_native.abs(), rtol=1e-4, atol=1e-5)
    assert torch.allclose(qr_factor(a, 24).abs(), r_native.abs(), rtol=1e-6, atol=1e-6)
    assert torch.allclose(qr_factor(a, 20).abs(), r_native.abs(), rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError):
        qr_factor(a.T, 24)
    g = a.T @ a + torch.eye(5, dtype=torch.float64)
    y_loop = cholesky_solve(g, b[:5], 24, force_loop=True)
    assert torch.equal(y_loop, y_loop.float().double())
    assert torch.allclose(y_loop, cholesky_solve(g, b[:5], 24), rtol=1e-4, atol=1e-5)
