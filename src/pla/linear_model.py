"""Linear least squares model with a prescribed spectrum.

The model serves to observe Theorems 3.1 and 4.1 directly: a matrix with known
singular triplets takes the place of the Jacobian, steps are computed in single
precision and evaluated in double precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from pla.precision import SINGLE_BITS, unit_roundoff
from pla.solvers import Solver, StepFailure, augmented, lm_step, triangular_factor


@dataclass(frozen=True)
class LinearModelConfig:
    """Size and spectrum of the model: ``sigma_i = 10**(-log_kappa * t_i**shape)``
    with ``t_i`` equispaced in ``[0, 1]``."""

    m: int = 600
    p: int = 200
    log_kappa: float = 10.0
    shape: float = 1.3
    seed: int = 7


@dataclass(frozen=True)
class LinearModel:
    """Matrix ``j = u diag(sigma) v^T`` and a unit residual ``r`` in its range."""

    j: Tensor
    u: Tensor
    v: Tensor
    sigma: Tensor
    r: Tensor


def build(config: LinearModelConfig) -> LinearModel:
    """Random orthogonal singular vectors, the prescribed singular values and a
    random residual in the range of the matrix."""
    rng = np.random.default_rng(config.seed)
    u, _ = np.linalg.qr(rng.standard_normal((config.m, config.p)))
    v, _ = np.linalg.qr(rng.standard_normal((config.p, config.p)))
    sigma = 10.0 ** (-config.log_kappa * np.linspace(0.0, 1.0, config.p) ** config.shape)
    r = u @ rng.standard_normal(config.p)
    r = r / np.linalg.norm(r)
    as_tensor = torch.from_numpy
    return LinearModel(as_tensor((u * sigma) @ v.T), as_tensor(u), as_tensor(v), as_tensor(sigma),
                       as_tensor(r))


def weak_residual(model: LinearModel, cut: float, seed: int) -> Tensor:
    """Unit residual concentrated on the directions with ``sigma_i < cut`` and a
    profile exponent of one, as at the end of a single-precision phase."""
    rng = np.random.default_rng(seed)
    sigma = model.sigma.numpy()
    p = sigma.shape[0]
    coef = np.where(sigma < cut, sigma / cut, 1e-3 * rng.random(p))
    r = model.u.numpy() @ (coef * rng.standard_normal(p))
    return torch.from_numpy(r / np.linalg.norm(r))


def filtered_components(model: LinearModel, eta_mu: float, solver: Solver, bits: int) -> Tensor | None:
    """Components ``|u_i^T ell|`` of the linearized residual after one step with
    relative damping threshold ``eta_mu``; ``None`` if the step fails."""
    mu = (eta_mu * float(model.sigma[0])) ** 2
    try:
        delta = lm_step(model.j, model.r, mu, solver, bits)
    except StepFailure:
        return None
    return (model.u.T @ (model.r + model.j @ delta)).abs()


def preconditioned_matrix(model: LinearModel, eta_mu: float, bits: int = SINGLE_BITS) -> Tensor:
    """``B = J_mu R^{-1}`` with the factor ``R`` computed in ``bits``-bit arithmetic."""
    mu = (eta_mu * float(model.sigma[0])) ** 2
    r_hat = triangular_factor(model.j, mu, bits)
    j_mu = augmented(model.j, mu)
    return torch.linalg.solve_triangular(r_hat.T, j_mu.T, upper=False).T


def exception_count(singular_values: Tensor, zeta: float) -> int:
    """Number of singular values outside ``[1 - zeta, 1 + zeta]``."""
    return int(((singular_values < 1.0 - zeta) | (singular_values > 1.0 + zeta)).sum())


def predicted_count(model: LinearModel, eta_mu: float, zeta: float, bits: int = SINGLE_BITS) -> int:
    """The count (4.1), with the constant of the threshold set to one."""
    mu = (eta_mu * float(model.sigma[0])) ** 2
    level = 2.0 * unit_roundoff(bits) * float(model.sigma[0]) / zeta
    return int((torch.sqrt(model.sigma ** 2 + mu) < level).sum())


def cg_iterations(b_mat: Tensor, rhs: Tensor, tolerance: float, max_iter: int) -> int:
    """Iterations of the conjugate gradient method on the normal equations of
    ``min ||b_mat y + rhs||`` until the energy-norm error is at most
    ``tolerance * ||rhs||`` (the criterion of Theorem 4.1(c)); ``max_iter`` if
    it is not reached."""
    y_star = torch.linalg.lstsq(b_mat, -rhs.unsqueeze(1)).solution.squeeze(1)
    y = torch.zeros(b_mat.shape[1], dtype=b_mat.dtype)
    res = -(b_mat.T @ rhs)
    direction = res.clone()
    rho = float(res @ res)
    target = tolerance * float(torch.linalg.norm(rhs))
    for iteration in range(1, max_iter + 1):
        image = b_mat @ direction
        step = rho / float(image @ image)
        y = y + step * direction
        res = res - step * (b_mat.T @ image)
        if float(torch.linalg.norm(b_mat @ (y - y_star))) <= target:
            return iteration
        rho_new = float(res @ res)
        direction = res + (rho_new / rho) * direction
        rho = rho_new
    return max_iter
