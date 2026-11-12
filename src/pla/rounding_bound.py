"""Cancellation scale and rounding-error count of Lemma 2.1.

The network and its first and second derivatives with respect to the inputs
are evaluated by forward propagation. Every quantity carries a majorant, which
bounds its absolute value, and a count of the rounding errors that accumulate
in it:

* inputs, weights, biases and data are exact (majorant ``|v|``, count 0);
* a sum or difference has the sum of the majorants and the larger count plus one;
* a product has the product of the majorants and the sum of the counts plus one;
* a value of the activation or of its ``k``-th derivative has the majorant
  ``bound_k + bound_{k+1} * majorant(argument)`` and the count of the argument
  plus ``c_act``.

Sums of several terms are counted in sequential order. The count is kept as
one integer per array, the largest over its entries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from pla.network import MLP
from pla.precision import unit_roundoff
from pla.problems.base import Problem
from pla.problems.linear import helmholtz_blocks

# sup |tanh^(k)| for k = 0, ..., 3
TANH_BOUNDS: tuple[float, ...] = (1.0, 1.0, 4.0 / (3.0 * math.sqrt(3.0)), 2.0)


def gamma(n: int, bits: int) -> float:
    """``gamma_n = n eps / (1 - n eps)`` for the format with ``bits`` significand bits."""
    product = n * unit_roundoff(bits)
    if product >= 1.0:
        raise ValueError("gamma_n requires n * eps < 1")
    return product / (1.0 - product)


@dataclass(frozen=True)
class Tracked:
    """An array of values with majorants and a rounding-error count."""

    value: Tensor
    majorant: Tensor
    count: int

    def __add__(self, other: "Tracked") -> "Tracked":
        return Tracked(self.value + other.value, self.majorant + other.majorant,
                       max(self.count, other.count) + 1)

    def __sub__(self, other: "Tracked") -> "Tracked":
        return Tracked(self.value - other.value, self.majorant + other.majorant,
                       max(self.count, other.count) + 1)

    def __mul__(self, other: "Tracked") -> "Tracked":
        return Tracked(self.value * other.value, self.majorant * other.majorant,
                       self.count + other.count + 1)


def exact(value: Tensor) -> Tracked:
    return Tracked(value, value.abs(), 0)


def _tanh_derivative(a: Tensor, order: int) -> Tensor:
    t = torch.tanh(a)
    s = 1.0 - t ** 2
    return (t, s, -2.0 * t * s)[order]


def _activation(a: Tracked, order: int, c_act: int) -> Tracked:
    majorant = TANH_BOUNDS[order] + TANH_BOUNDS[order + 1] * a.majorant
    return Tracked(_tanh_derivative(a.value, order), majorant, a.count + c_act)


def _linear(weight: Tensor, h: Tracked, bias: Tensor | None = None) -> Tracked:
    """``h W^T (+ bias)`` for rows ``h``: ``n`` products, ``n - 1`` sums and
    one more sum for the bias."""
    value, majorant = h.value @ weight.T, h.majorant @ weight.abs().T
    count = h.count + weight.shape[1]
    if bias is None:
        return Tracked(value, majorant, count)
    return Tracked(value + bias, majorant + bias.abs(), count + 1)


@dataclass(frozen=True)
class Propagated:
    """Network values with the derivatives ``du/dx_k`` and ``d^2u/dx_k^2``,
    each of shape ``(points,)``."""

    value: Tracked
    first: tuple[Tracked, ...]
    second: tuple[Tracked, ...]


def propagate(net: MLP, theta: Tensor, x: Tensor, *, c_act: int = 1) -> Propagated:
    """Forward propagation of values, first and pure second derivatives of a
    scalar network at the rows of ``x``, with majorants and counts."""
    if net.sizes[-1] != 1:
        raise ValueError("the bound is implemented for scalar networks")
    layers = net.layers(theta.to(torch.float64))
    points, dim = x.shape
    h = exact(x.to(torch.float64))
    first: list[Tracked] | None = None
    second: list[Tracked] | None = None
    for index, (weight, bias) in enumerate(layers):
        a = _linear(weight, h, bias)
        if first is None:  # derivatives of the first argument are weights
            da = [exact(weight[:, k].expand(points, -1)) for k in range(dim)]
            d2a: list[Tracked] | None = None
        else:
            da = [_linear(weight, g) for g in first]
            d2a = [_linear(weight, g) for g in second] if second is not None else None
        if index == len(layers) - 1:
            zero = exact(torch.zeros(points, 1, dtype=torch.float64))
            last = d2a if d2a is not None else [zero] * dim
            return Propagated(_squeeze(a), tuple(_squeeze(g) for g in da),
                              tuple(_squeeze(g) for g in last))
        slope, curvature = _activation(a, 1, c_act), _activation(a, 2, c_act)
        h = _activation(a, 0, c_act)
        first = [slope * g for g in da]
        curved = [curvature * (g * g) for g in da]
        second = curved if d2a is None else [c + slope * g for c, g in zip(curved, d2a)]
    raise AssertionError("unreachable")  # pragma: no cover


def _squeeze(tracked: Tracked) -> Tracked:
    return Tracked(tracked.value[:, 0], tracked.majorant[:, 0], tracked.count)


@dataclass(frozen=True)
class RoundingBound:
    """Cancellation scale ``rho`` of all residual entries and the largest count ``N``."""

    rho: Tensor
    count: int


def cancellation_scale(problem: Problem, theta: Tensor, *, c_act: int = 1) -> RoundingBound:
    """Cancellation scale (2.9) for a problem built by
    :func:`pla.problems.linear.helmholtz` or :func:`pla.problems.linear.poisson`,
    with residual entries ``sqrt(w) (-Laplace(u) - wavenumber^2 u - f)`` in the
    interior and ``sqrt(w) (u - g)`` on the boundary.
    """
    interior, boundary, wavenumber = helmholtz_blocks(problem)
    n_interior = interior.size
    shift = wavenumber ** 2
    inside = propagate(problem.net, theta, interior.points, c_act=c_act)
    majorant = sum(term.majorant for term in inside.second)
    count = max(term.count for term in inside.second) + len(inside.second) - 1
    if shift != 0.0:  # one product with the coefficient and one more sum
        majorant = majorant + shift * inside.value.majorant
        count = max(count, inside.value.count + 1) + 1
    # subtraction of the data and product with the square root of the weight
    rho_interior = math.sqrt(interior.weight) * majorant + problem.data[:n_interior].abs()
    outside = propagate(problem.net, theta, boundary.points, c_act=c_act)
    rho_boundary = math.sqrt(boundary.weight) * outside.value.majorant + problem.data[n_interior:].abs()
    total = max(count, outside.value.count) + 2
    return RoundingBound(torch.cat([rho_interior, rho_boundary]), total)


def evaluation_floor(bound: RoundingBound, bits: int) -> float:
    """The level ``gamma_N ||rho||`` below which a residual evaluated with
    ``bits`` significand bits cannot be told apart from its rounding error."""
    return gamma(bound.count, bits) * float(torch.linalg.norm(bound.rho))
