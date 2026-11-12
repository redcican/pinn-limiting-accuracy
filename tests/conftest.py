"""Shared fixtures: every test runs in double precision by default and on the CPU."""
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
