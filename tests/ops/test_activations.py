"""Finite-difference gradient checks for activations."""

from nanoad import Tensor, relu, tanh


def test_relu_positive(grad_check):
    grad_check(lambda x: relu(x), 2.0)


def test_relu_negative(grad_check):
    grad_check(lambda x: relu(x), -2.0)


def test_relu_at_zero():
    """ReLU at exactly 0: convention is grad = 0 (right sub-gradient)."""
    x = Tensor(0.0)
    y = relu(x)
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0


def test_tanh_zero(grad_check):
    grad_check(lambda x: tanh(x), 0.0)


def test_tanh_nonzero(grad_check):
    grad_check(lambda x: tanh(x), 0.5)


def test_tanh_saturating(grad_check):
    """Saturating regime — gradient should be near 0 but still finite."""
    grad_check(lambda x: tanh(x), 3.0)
