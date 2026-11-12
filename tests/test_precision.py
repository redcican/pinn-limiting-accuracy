import numpy as np
import pytest
import torch

from pla.precision import FORMATS, Arithmetic, chop, unit_roundoff

pytestmark = pytest.mark.unit


def test_unit_roundoff_of_standard_formats() -> None:
    assert unit_roundoff(53) == 2.0 ** -53
    assert FORMATS["binary32"].unit_roundoff == 2.0 ** -24
    assert FORMATS["binary16"].bits == 11
    assert FORMATS["bfloat16"].bits == 8


def test_chop_matches_numpy_single_precision() -> None:
    x = torch.randn(4000) * torch.logspace(-20, 20, 4000)
    expected = torch.from_numpy(x.numpy().astype(np.float32).astype(np.float64))
    assert torch.equal(chop(x, 24), expected)


def test_chop_matches_numpy_half_precision_in_normal_range() -> None:
    x = (torch.rand(4000) + 0.5) * torch.logspace(-3, 3, 4000) * torch.sign(torch.randn(4000))
    expected = torch.from_numpy(x.numpy().astype(np.float16).astype(np.float64))
    assert torch.equal(chop(x, 11), expected)


def test_chop_rounds_ties_to_even_and_keeps_zero() -> None:
    # 1 + 2^-3 is a tie for 3 significand bits: neighbours 1.0 (even) and 1.25.
    x = torch.tensor([1.125, 1.375, 0.0, -1.125])
    assert chop(x, 3).tolist() == [1.0, 1.5, 0.0, -1.0]


def test_chop_is_identity_at_double_precision() -> None:
    x = torch.randn(10)
    assert chop(x, 53) is x


def test_chop_rejects_invalid_bits() -> None:
    with pytest.raises(ValueError):
        chop(torch.ones(1), 0)


def test_arithmetic_native_and_emulated_casts() -> None:
    x = torch.randn(5)
    assert Arithmetic(24).cast(x).dtype == torch.float32
    assert Arithmetic(53).cast(x).dtype == torch.float64
    emulated = Arithmetic(11)
    assert not emulated.native
    assert torch.equal(emulated.cast(x), chop(x, 11))
    assert emulated.unit_roundoff == 2.0 ** -11
