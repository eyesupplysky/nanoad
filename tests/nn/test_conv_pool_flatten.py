"""nn.Conv2d / nn.MaxPool2d / nn.Flatten Module-wrapper tests."""

from __future__ import annotations

import numpy as np

from nanoad import Tensor
from nanoad.nn import Conv2d, Flatten, MaxPool2d


def test_conv2d_module_forward_shape() -> None:
    layer = Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 3, 5, 5))
    out = layer(x)
    assert out.shape == (2, 4, 5, 5)


def test_conv2d_module_he_init_std() -> None:
    layer = Conv2d(in_channels=4, out_channels=8, kernel_size=3)
    fan_in = 4 * 3 * 3
    expected_std = np.sqrt(2.0 / fan_in)
    np.testing.assert_allclose(layer.weight.data.std(), expected_std, rtol=0.15)


def test_conv2d_module_bias_starts_at_zero() -> None:
    layer = Conv2d(in_channels=2, out_channels=4, kernel_size=3)
    assert layer.bias is not None
    np.testing.assert_allclose(layer.bias.data, 0.0)


def test_conv2d_module_bias_false_yields_no_bias_param() -> None:
    layer = Conv2d(in_channels=2, out_channels=4, kernel_size=3, bias=False)
    assert layer.bias is None
    params = list(layer.parameters())
    assert len(params) == 1


def test_maxpool2d_module_forward_shape() -> None:
    layer = MaxPool2d(kernel_size=2)
    x = Tensor(np.random.randn(1, 3, 4, 4))
    out = layer(x)
    assert out.shape == (1, 3, 2, 2)


def test_flatten_module_collapses_to_2d() -> None:
    layer = Flatten()
    x = Tensor(np.random.randn(7, 2, 3, 4))
    out = layer(x)
    assert out.shape == (7, 24)


def test_flatten_grads_propagate() -> None:
    """Flatten is a reshape — backward should restore original shape."""
    layer = Flatten()
    x = Tensor(np.random.randn(2, 3, 4))
    y = layer(x)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones_like(x.data))
