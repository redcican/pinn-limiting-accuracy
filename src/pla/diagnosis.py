"""Floor diagnosis (Algorithm 3.1).

At a point where training has stalled, the residual is split along the
singular directions of the Jacobian into a tangent-orthogonal part, a
precision-truncated part and a reducible part (3.2), and the largest of the
three at the threshold of the solver is reported as the cause of the floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch
from torch import Tensor

from pla.precision import DOUBLE_BITS, SINGLE_BITS, unit_roundoff


class Cause(str, Enum):
    TANGENT_SPACE = "tangent space"
    PRECISION = "precision"
    OPTIMIZATION = "optimization"


@dataclass(frozen=True)
class Split:
    """Shares of the residual norm at a relative threshold ``eta``."""

    eta: float
    e_perp: float
    e_prec: float
    e_opt: float


@dataclass(frozen=True)
class Diagnosis:
    """Result of Algorithm 3.1; ``sigma_rel`` and ``picard`` are the data of a Picard plot."""

    sigma_rel: Tensor
    picard: Tensor
    splits: Mapping[str, Split]
    beta: float
    k_bad: int
    cause: Cause


def thresholds(tau: float, c_ne: float = 1.0, c_qr: float = 1.0) -> dict[str, float]:
    """Relative thresholds of (2.10) for the three solver settings of the paper.

    The constants ``c_ne`` and ``c_qr`` are the modest polynomials of
    Lemma 2.2; the default of one gives the nominal values.
    """
    eps32, eps64 = unit_roundoff(SINGLE_BITS), unit_roundoff(DOUBLE_BITS)
    return {"ne64": math.sqrt(c_ne * eps64), "qr32": c_qr * eps32 / tau, "qr64": c_qr * eps64 / tau}


def _decompose(j: Tensor, r: Tensor, rank_tol: float = 0.0) -> tuple[Tensor, Tensor, float]:
    """Singular values, coefficients ``u_i^T r`` and the norm of the part of ``r``
    outside the range of ``j``. Directions with ``sigma_i <= rank_tol * sigma_1``
    count as outside; their singular values are returned as zero."""
    u, sigma, _ = torch.linalg.svd(j.to(torch.float64), full_matrices=False)
    r64 = r.to(torch.float64)
    coef = u.T @ r64
    null = sigma <= rank_tol * sigma[0]
    outside = torch.linalg.norm(r64 - u @ coef) ** 2 + torch.sum(coef[null] ** 2)
    return torch.where(null, torch.zeros_like(sigma), sigma), coef, float(torch.sqrt(outside))


def _split(sigma: Tensor, coef: Tensor, perp: float, eta: float) -> Split:
    positive = sigma > 0.0
    strong = positive & (sigma >= eta * sigma[0])
    weak = positive & ~strong
    opt = float(torch.linalg.norm(coef[strong]))
    prec = float(torch.linalg.norm(coef[weak]))
    norm = math.sqrt(perp ** 2 + prec ** 2 + opt ** 2)
    if norm == 0.0:
        return Split(eta, 0.0, 0.0, 0.0)
    return Split(eta, perp / norm, prec / norm, opt / norm)


def residual_split(j: Tensor, r: Tensor, eta: float, *, rank_tol: float = 0.0) -> Split:
    """The split (3.2) of ``r`` at the relative threshold ``eta``, as shares of ``||r||``.

    With the default ``rank_tol=0`` every direction with a positive computed
    singular value belongs to the tangent space, as in the definition. For a
    Jacobian that is rank deficient, the singular values of the null
    directions are computed as rounding noise; a positive ``rank_tol`` (for
    example ``max(m, p) * 2**-53``) assigns such directions to the
    tangent-orthogonal part.
    """
    sigma, coef, perp = _decompose(j, r, rank_tol)
    return _split(sigma, coef, perp, eta)


def fit_beta(sigma_rel: Tensor, coef: Tensor, eta_low: float, eta_high: float) -> float:
    """Exponent of the truncation profile ``g(eta) ~ eta**beta`` on
    ``[eta_low, eta_high]``, by least squares on a logarithmic scale.

    ``g`` is sampled at the relative singular values; ``g(sigma_i)`` is the
    norm of the coefficients that belong to strictly smaller singular values.
    Returns ``nan`` if fewer than two samples lie in the interval.
    """
    s, inverse = torch.unique(sigma_rel, sorted=True, return_inverse=True)  # ascending, ties merged
    energy = torch.zeros_like(s).index_add_(0, inverse, coef.to(s.dtype) ** 2)
    below = torch.cumsum(energy, 0) - energy
    keep = (s >= eta_low) & (s <= eta_high) & (below > 0.0) & (s > 0.0)
    if int(keep.sum()) < 2:
        return math.nan
    x = torch.log(s[keep])
    y = 0.5 * torch.log(below[keep])
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    return float(torch.linalg.lstsq(design, y.unsqueeze(1)).solution[0, 0])


def diagnose(j: Tensor, r: Tensor, *, solver: str, tau: float, c_ne: float = 1.0,
             c_qr: float = 1.0, rank_tol: float = 0.0) -> Diagnosis:
    """Algorithm 3.1 for the Jacobian ``j`` and residual ``r`` at a stall point.

    ``solver`` names the threshold of the solver that produced the run
    (``"ne64"``, ``"qr32"`` or ``"qr64"``); ``rank_tol`` is explained in
    :func:`residual_split`.
    """
    levels = thresholds(tau, c_ne, c_qr)
    if solver not in levels:
        raise KeyError(f"unknown solver threshold {solver!r}; expected one of {sorted(levels)}")
    sigma, coef, perp = _decompose(j, r, rank_tol)
    norm = float(torch.linalg.norm(r.to(torch.float64)))
    sigma_rel = sigma / sigma[0]
    splits = {name: _split(sigma, coef, perp, eta) for name, eta in levels.items()}
    beta = fit_beta(sigma_rel, coef, levels["qr64"], levels["ne64"])
    k_bad = int((sigma_rel < unit_roundoff(SINGLE_BITS) / tau).sum())
    at_solver = splits[solver]
    cause = max(((at_solver.e_perp, Cause.TANGENT_SPACE), (at_solver.e_prec, Cause.PRECISION),
                 (at_solver.e_opt, Cause.OPTIMIZATION)), key=lambda item: item[0])[1]
    picard = coef.abs() / norm if norm > 0.0 else coef.abs()
    return Diagnosis(sigma_rel, picard, splits, beta, k_bad, cause)
