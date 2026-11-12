import pytest
import torch

from pla.network import MLP

pytestmark = pytest.mark.unit


def test_parameter_count_and_layout() -> None:
    net = MLP((2, 32, 32, 32, 1))
    assert net.num_params == 2241  # the network of problem P1
    assert MLP((3, 48, 48, 48, 1)).num_params == 4945


def test_forward_agrees_with_an_explicit_evaluation() -> None:
    net = MLP((2, 3, 1))
    theta = net.init(seed=1)
    w1, b1 = theta[:6].reshape(3, 2), theta[6:9]
    w2, b2 = theta[9:12].reshape(1, 3), theta[12:]
    x = torch.tensor([0.3, -0.7])
    expected = w2 @ torch.tanh(w1 @ x + b1) + b2
    assert torch.allclose(net(theta, x), expected)
    batch = torch.randn(5, 2)
    assert net(theta, batch).shape == (5, 1)
    assert torch.allclose(net(theta, batch)[2], net(theta, batch[2]))


def test_initialization_is_reproducible_and_typed() -> None:
    net = MLP((2, 8, 1))
    assert torch.equal(net.init(seed=3), net.init(seed=3))
    assert not torch.equal(net.init(seed=3), net.init(seed=4))
    assert net.init(seed=3).dtype == torch.float64
    theta32 = net.init(seed=3).to(torch.float32)
    assert net(theta32, torch.zeros(2, dtype=torch.float32)).dtype == torch.float32


def test_invalid_layout_and_parameter_vector_are_rejected() -> None:
    with pytest.raises(ValueError):
        MLP((2,))
    with pytest.raises(ValueError):
        MLP((2, 4, 1))(torch.zeros(5), torch.zeros(2))
