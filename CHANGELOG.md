# Changelog

## [0.1.0] - 2026-11-12
### Features
- Precision emulation (`pla.precision`), Householder QR and Cholesky in a chosen precision (`pla.linalg`).
- Levenberg-Marquardt steps by normal equations, residual-space form and Householder QR (`pla.solvers`), and
  the training loop with separate precisions for residual, Jacobian and solver (`pla.lm`).
- Diagnosis of a floor (`pla.diagnosis`), the rounding bound for the residual (`pla.rounding_bound`).
- Mixed-precision Gauss-Newton method with preconditioned LSQR (`pla.mpgn`, `pla.lsqr`) and multi-stage
  training (`pla.multistage`).
- Test problems: regression, Poisson, Helmholtz, biharmonic, Burgers, Allen-Cahn (`pla.problems`).
- Baselines: energy natural gradient (primal and dual), a Kronecker-factored Gauss-Newton method, Adam
  followed by L-BFGS (`pla.baselines`).
- Linear model with a prescribed spectrum (`pla.linear_model`).

### Design Rationale
- PyTorch with `torch.func`: per-point operators are written once for a scalar function and are batched and
  differentiated by transforms, so that the same code yields residuals, dense Jacobians and matrix-free products.
- Residual and Jacobian precisions are limited to binary32 and binary64, since they come from automatic
  differentiation; only the linear solver is emulated in other formats.
- The parameters are one flat vector, which keeps the solvers independent of the network.

### Notes & Caveats
- Emulated formats round the result of each vector operation once; inner products are accumulated in double
  precision. The exponent range is not modelled.
- Only small problems have been run (CPU). No experiment drivers and no results are included.
- The Kronecker-factored baseline is a compact variant for the Poisson and Helmholtz problems.
