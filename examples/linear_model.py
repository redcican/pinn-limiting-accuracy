"""Linear model with a prescribed spectrum: the filter of Theorem 3.1 and the
preconditioner of Theorem 4.1 on a computed example.

Everything is logged; no file is written. Usage:

    python examples/linear_model.py
"""
from __future__ import annotations

import logging

import torch

from pla.linear_model import (LinearModelConfig, build, cg_iterations, exception_count,
                              filtered_components, preconditioned_matrix, predicted_count,
                              weak_residual)
from pla.precision import SINGLE_BITS, unit_roundoff
from pla.solvers import Solver

logger = logging.getLogger("linear_model")
ZETA: float = 0.5
TOLERANCE: float = 1e-3


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_default_dtype(torch.float64)
    model = build(LinearModelConfig())
    sigma_rel = model.sigma / model.sigma[0]
    logger.info("m = %d, p = %d, condition number %.1e", *model.j.shape, 1.0 / float(sigma_rel[-1]))

    eta_mu = 1e-5
    exact = eta_mu ** 2 / (sigma_rel ** 2 + eta_mu ** 2) * (model.u.T @ model.r).abs()
    for solver in (Solver.QR, Solver.NE):
        computed = filtered_components(model, eta_mu, solver, SINGLE_BITS)
        if computed is None:
            logger.info("filter, %s in single precision: the step fails", solver.value)
            continue
        deviation = float((computed - exact).abs().max() / torch.linalg.norm(model.r))
        logger.info("filter, %s in single precision: largest deviation from mu/(sigma^2+mu) "
                    "relative to the residual %.2e", solver.value, deviation)

    for eta in (1e-5, 1e-7, 1e-9):
        b_mat = preconditioned_matrix(model, eta)
        values = torch.linalg.svdvals(b_mat)
        rhs = torch.cat([weak_residual(model, unit_roundoff(SINGLE_BITS) * float(model.sigma[0]), 1),
                         torch.zeros(model.j.shape[1])])
        logger.info("preconditioner, eta_mu = %.0e: singular values in [%.2e, %.2e], %d outside "
                    "[1 - zeta, 1 + zeta] (count (4.1): %d), %d conjugate gradient iterations",
                    eta, float(values[-1]), float(values[0]), exception_count(values, ZETA),
                    predicted_count(model, eta, ZETA), cg_iterations(b_mat, rhs, TOLERANCE, 500))


if __name__ == "__main__":
    main()
