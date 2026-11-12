"""Multilayer perceptron with hyperbolic tangent activation, as a pure function
of a flat parameter vector."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MLP:
    """Fully connected network with layer widths ``sizes`` (input first, output last).

    The parameters are stored in one vector: for each layer the weight matrix
    in row-major order, followed by the bias.
    """

    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.sizes) < 2 or min(self.sizes) < 1:
            raise ValueError("sizes must list at least an input and an output width")

    @property
    def num_params(self) -> int:
        return sum((n_in + 1) * n_out for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]))

    def init(self, seed: int, dtype: torch.dtype = torch.float64) -> Tensor:
        """Glorot-uniform weights and zero biases, reproducible from ``seed``."""
        generator = torch.Generator().manual_seed(seed)
        parts = []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            bound = math.sqrt(6.0 / (n_in + n_out))
            weight = (2.0 * torch.rand(n_out, n_in, generator=generator, dtype=torch.float64) - 1.0) * bound
            parts.extend([weight.reshape(-1), torch.zeros(n_out, dtype=torch.float64)])
        return torch.cat(parts).to(dtype)

    def layers(self, theta: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Weight matrices and bias vectors as views of ``theta``."""
        if theta.shape != (self.num_params,):
            raise ValueError(f"expected {self.num_params} parameters, got {tuple(theta.shape)}")
        out, start = [], 0
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            weight = theta[start:start + n_in * n_out].reshape(n_out, n_in)
            start += n_in * n_out
            out.append((weight, theta[start:start + n_out]))
            start += n_out
        return out

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        """Network output for one point ``x`` of shape ``(d,)`` or a batch ``(n, d)``."""
        layers = self.layers(theta)
        h = x
        for weight, bias in layers[:-1]:
            h = torch.tanh(h @ weight.T + bias)
        weight, bias = layers[-1]
        return h @ weight.T + bias
