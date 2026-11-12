"""Floating-point formats and emulation of low-precision arithmetic.

A format is identified by its number ``t`` of significand bits, and its unit
roundoff is ``2**-t``. The formats binary32 and binary64 are native; all other
formats are emulated by rounding double-precision values to ``t`` bits, in the
manner of the ``chop`` function of Higham and Pranesh (SIAM J. Sci. Comput.,
41 (2019), pp. C585-C602). The emulation covers the significand only: the
exponent range of the target format (overflow, underflow, subnormal numbers)
is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SINGLE_BITS: int = 24
DOUBLE_BITS: int = 53


def unit_roundoff(bits: int) -> float:
    """Unit roundoff ``2**-bits`` of a binary format with ``bits`` significand bits."""
    if not 1 <= bits <= DOUBLE_BITS:
        raise ValueError(f"bits must lie in [1, {DOUBLE_BITS}], got {bits}")
    return 2.0 ** -bits


@dataclass(frozen=True)
class Format:
    """A binary floating-point format."""

    name: str
    bits: int

    @property
    def unit_roundoff(self) -> float:
        return unit_roundoff(self.bits)


FORMATS: dict[str, Format] = {
    f.name: f
    for f in (
        Format("bfloat16", 8),
        Format("binary16", 11),
        Format("binary32", SINGLE_BITS),
        Format("binary64", DOUBLE_BITS),
    )
}


def chop(x: Tensor, bits: int) -> Tensor:
    """Round a double-precision tensor to ``bits`` significand bits (ties to even).

    For ``bits == 53`` the input is returned unchanged.
    """
    if not 1 <= bits <= DOUBLE_BITS:
        raise ValueError(f"bits must lie in [1, {DOUBLE_BITS}], got {bits}")
    if x.dtype != torch.float64:
        raise TypeError("chop expects a double-precision tensor")
    if bits == DOUBLE_BITS:
        return x
    mantissa, exponent = torch.frexp(x)  # x = mantissa * 2**exponent, |mantissa| in [0.5, 1)
    rounded = torch.round(torch.ldexp(mantissa, torch.tensor(bits)))  # ties to even
    return torch.ldexp(rounded, exponent - bits)


@dataclass(frozen=True)
class Arithmetic:
    """Working precision of a computation: native for 24 and 53 bits, emulated otherwise."""

    bits: int

    def __post_init__(self) -> None:
        unit_roundoff(self.bits)  # validates the range

    @property
    def native(self) -> bool:
        return self.bits in (SINGLE_BITS, DOUBLE_BITS)

    @property
    def unit_roundoff(self) -> float:
        return unit_roundoff(self.bits)

    @property
    def dtype(self) -> torch.dtype:
        """Storage type: float32 for binary32, float64 for every other format."""
        return torch.float32 if self.bits == SINGLE_BITS else torch.float64

    def cast(self, x: Tensor) -> Tensor:
        """Round ``x`` to this precision (stored as float32 or as rounded float64)."""
        if self.bits == SINGLE_BITS:
            return x.to(torch.float32)
        return chop(x.to(torch.float64), self.bits)

    def round(self, x: Tensor) -> Tensor:
        """Round the result of an operation; a no-op for native formats, whose
        operations already round to the working precision."""
        return x if self.native else chop(x, self.bits)
