# pinn-limiting-accuracy

Reference implementation for the manuscript *Limiting accuracy of physics-informed neural network training in
floating-point arithmetic*.

Gauss-Newton training of a physics-informed neural network stalls at a level set by the linear solver of the
step and by the precision it runs in. This package contains the algorithms needed to study that level: steps
computed in a chosen precision, a diagnosis that splits a stalled residual into its parts, a mixed-precision
Gauss-Newton method, multi-stage training, and the optimizers these are compared with.

**Status.** This release contains the methods and their tests. Experiment drivers and results are not included
yet. Only small problems have been run so far, on a CPU.

## Installation

```bash
git clone https://github.com/redcican/pinn-limiting-accuracy.git
cd pinn-limiting-accuracy
pip install -e ".[dev]"
python -m pytest -q
```

Python 3.10 or later and PyTorch 2.2 or later are required.

## Contents

| Module | Content | Manuscript |
|---|---|---|
| `pla.precision` | floating-point formats, rounding of double-precision values to `t` significand bits | Table 1 |
| `pla.network` | tanh multilayer perceptron as a function of one parameter vector | section 2.1 |
| `pla.problems` | regression, Poisson, Helmholtz, biharmonic, Burgers and Allen-Cahn residuals | (2.2), Table 2 |
| `pla.jacobian` | dense Jacobian of the residual, matrix-free products | section 2.2 |
| `pla.linalg` | Householder QR and Cholesky factorization in a chosen precision | Lemma 2.2 |
| `pla.solvers` | Levenberg-Marquardt step by normal equations, residual-space form, Householder QR | (2.4)-(2.7) |
| `pla.lm` | Levenberg-Marquardt loop with precisions for residual, Jacobian and solver | section 2 |
| `pla.rounding_bound` | cancellation scale and rounding-error count of the residual | Lemma 2.1 |
| `pla.diagnosis` | split of a residual into tangent-orthogonal, precision and optimization parts | Algorithm 3.1 |
| `pla.lsqr` | LSQR with right preconditioner and the stopping test of the method | Algorithm 4.1 |
| `pla.mpgn` | mixed-precision Gauss-Newton method | Algorithm 4.1 |
| `pla.multistage` | multi-stage training as iterative refinement | section 4.3 |
| `pla.linear_model` | least squares model with a prescribed spectrum | Theorems 3.1, 4.1 |
| `pla.baselines` | energy natural gradient (primal, dual), Kronecker-factored Gauss-Newton, Adam with L-BFGS | section 5 |

## Example

```python
import torch
from pla.lm import LMConfig, levenberg_marquardt
from pla.mpgn import MPGNConfig, mpgn
from pla.network import MLP
from pla.problems.linear import poisson
from pla.solvers import Solver

torch.set_default_dtype(torch.float64)
net = MLP((2, 16, 16, 1))
problem = poisson(net, dim=2, n_interior=1200, n_boundary=300, seed=0)
theta0 = net.init(seed=0)

# Levenberg-Marquardt with Householder QR, everything in single precision
single = levenberg_marquardt(problem, theta0, LMConfig(solver=Solver.QR, bits_r=24, bits_j=24, bits_s=24))

# mixed-precision Gauss-Newton method
mixed = mpgn(problem, theta0, MPGNConfig())
print(problem.relative_error((single.theta,)), problem.relative_error((mixed.theta,)))
```

The scripts in `examples/` log a comparison of the solvers on a small Poisson problem and the checks on the
linear model.

## Precision emulation

The formats binary32 and binary64 are native. Every other format is emulated in the linear solver by rounding
double-precision values to the given number of significand bits. The result of each vector operation is rounded
once, and inner products are accumulated in double precision before they are rounded, so the emulation slightly
underestimates the error of a true low-precision computation. Overflow, underflow and subnormal numbers are not
modelled. Residuals and Jacobians are evaluated in binary32 or binary64 only.

## Scope of this release

- `pla.rounding_bound` covers the Poisson and Helmholtz problems (operators up to second order).
- The Kronecker-factored baseline is a compact variant for the same two problems, without momentum and moving
  averages.
- `pla.multistage` is formulated for linear problems.
- The dense Jacobian limits the problem size to what fits in memory.

## License

MIT, see `LICENSE`.
