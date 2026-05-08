"""Linear layer tests."""

import numpy as np

from nanoad import Tensor
from nanoad.nn import Linear


def test_linear_forward_shape():
    layer = Linear(3, 5)
    x = Tensor(np.random.randn(7, 3))
    y = layer(x)
    assert y.shape == (7, 5)


def test_linear_he_init_std_is_sqrt_2_over_fan_in():
    np.random.seed(0)
    layer = Linear(100, 50)
    expected_std = float(np.sqrt(2.0 / 100))
    actual_std = float(layer.weight.data.std())
    assert abs(actual_std - expected_std) < 0.05 * expected_std


def test_linear_bias_starts_at_zero():
    layer = Linear(3, 4)
    assert (layer.bias.data == 0).all()


def test_linear_grads_propagate():
    np.random.seed(0)
    layer = Linear(3, 2)
    x = Tensor(np.random.randn(4, 3))
    out = layer(x).sum()
    out.backward()
    assert layer.weight.grad.shape == (3, 2)
    assert layer.bias.grad.shape == (2,)
    assert not np.allclose(layer.weight.grad, 0)
    assert not np.allclose(layer.bias.grad, 0)


def test_linear_forward_value():
    """y = x @ W + b. Pin W and b to known values and verify."""
    layer = Linear(2, 3)
    layer.weight.data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer.bias.data = np.array([10.0, 20.0, 30.0])
    x = Tensor([[1.0, 1.0]])
    y = layer(x)
    assert np.allclose(y.data, [[15.0, 27.0, 39.0]])
