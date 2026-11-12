"""Small Poisson problem on the unit square: Levenberg-Marquardt with normal
equations and with Householder QR, the mixed-precision Gauss-Newton method,
and the diagnosis of Algorithm 3.1 at the end of each run.

Everything is logged; no file is written. Usage:

    python examples/poisson2d_small.py [--width 16] [--iterations 500]
"""
from __future__ import annotations

import argparse
import logging

import torch

from pla.diagnosis import diagnose
from pla.jacobian import jacobian
from pla.lm import LMConfig, levenberg_marquardt
from pla.mpgn import MPGNConfig, mpgn
from pla.network import MLP
from pla.problems.base import Problem
from pla.problems.linear import poisson
from pla.solvers import Solver

logger = logging.getLogger("poisson2d_small")
TAU: float = 0.5


def report(name: str, problem: Problem, theta: torch.Tensor, threshold: str) -> None:
    theta = theta.to(torch.float64)
    result = diagnose(jacobian(problem, theta), problem.residual(theta), solver=threshold, tau=TAU)
    split = result.splits[threshold]
    logger.info("%-12s loss %.3e, relative error %.3e", name, problem.loss(theta),
                problem.relative_error((theta,)))
    logger.info("%-12s shares at the threshold %.1e: tangent-orthogonal %.2f, precision %.2f, "
                "optimization %.2f; beta %.2f; cause: %s", "", split.eta, split.e_perp,
                split.e_prec, split.e_opt, result.beta, result.cause.value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_default_dtype(torch.float64)

    net = MLP((2, args.width, args.width, 1))
    problem = poisson(net, dim=2, n_interior=1200, n_boundary=300, seed=args.seed, n_test=10_000)
    theta0 = net.init(seed=args.seed)
    logger.info("%d residuals, %d parameters", problem.m, net.num_params)

    runs = {
        "LM-NE (64)": (LMConfig(solver=Solver.NE, max_iter=args.iterations), "ne64"),
        "LM-QR (32)": (LMConfig(solver=Solver.QR, bits_r=24, bits_j=24, bits_s=24,
                                max_iter=args.iterations), "qr32"),
        "LM-QR (64)": (LMConfig(solver=Solver.QR, max_iter=args.iterations), "qr64"),
    }
    for name, (config, threshold) in runs.items():
        result = levenberg_marquardt(problem, theta0, config)
        logger.info("%-12s %d iterations, %d accepted, stalled: %s", name, len(result.losses) - 1,
                    result.accepted, result.stalled)
        report(name, problem, result.theta, threshold)

    phase1 = LMConfig(solver=Solver.QR, bits_r=24, bits_j=24, bits_s=24, max_iter=args.iterations)
    result = mpgn(problem, theta0, MPGNConfig(tau=TAU, max_outer=args.iterations, phase1=phase1))
    inner = result.inner_iterations
    logger.info("%-12s switch after %d iterations, %d steps in phase 2, LSQR iterations: "
                "mean %.1f, largest %d", "MP-GN", result.switch_iteration, len(inner),
                sum(inner) / max(len(inner), 1), max(inner, default=0))
    report("MP-GN", problem, result.theta, "qr64")


if __name__ == "__main__":
    main()
