"""M8: ``grad`` — closed-form scalar checks plus int and tuple ``argnum``."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor, tanh
from nanoad.functional import grad


def test_grad_of_x_squared() -> None:
    """d(x²)/dx = 2x — the smallest possible grad test."""
    grad_fn = grad(lambda x: x * x)
    g = grad_fn(3.0)
    np.testing.assert_allclose(g.data, 6.0)


def test_grad_of_quadratic_form() -> None:
    """d(xᵀx)/dx = 2x — vector input, scalar output."""
    x_data = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    grad_fn = grad(lambda x: (x.transpose() @ x).sum())
    g = grad_fn(x_data)
    np.testing.assert_allclose(g.data.flatten(), 2.0 * x_data.flatten())


def test_grad_of_tanh_at_zero() -> None:
    """d(tanh)/dx |₀ = 1."""
    grad_fn = grad(lambda x: tanh(x).sum())
    g = grad_fn(0.0)
    np.testing.assert_allclose(g.data, 1.0)


def test_grad_returns_tensor() -> None:
    """grad's return type is Tensor, not ndarray."""
    grad_fn = grad(lambda x: x * x)
    result = grad_fn(2.0)
    assert isinstance(result, Tensor)


def test_grad_accepts_existing_tensor_as_input() -> None:
    """Passing a Tensor as input works the same as passing an ndarray (input is copied to a fresh leaf)."""
    grad_fn = grad(lambda x: (x * x).sum())
    g_from_arr = grad_fn(np.array([1.0, 2.0]))
    g_from_tensor = grad_fn(Tensor(np.array([1.0, 2.0])))
    np.testing.assert_allclose(g_from_arr.data, g_from_tensor.data)


def test_grad_argnum_int_selects_one_arg() -> None:
    """grad(fn, argnum=1) differentiates wrt the second positional arg only."""
    grad_fn = grad(lambda x, y: (x * y).sum(), argnum=1)
    g = grad_fn(np.array([2.0, 3.0]), np.array([4.0, 5.0]))
    # d/dy of x·y is x.
    np.testing.assert_allclose(g.data, [2.0, 3.0])


def test_grad_argnum_negative_resolves_from_end() -> None:
    """argnum=-1 means the last arg."""
    grad_fn = grad(lambda x, y, z: (x + y + z).sum(), argnum=-1)
    g = grad_fn(1.0, 2.0, 3.0)
    np.testing.assert_allclose(g.data, 1.0)


def test_grad_argnum_tuple_returns_tuple() -> None:
    """grad(fn, argnum=(0, 1)) returns a tuple of gradients in argnum order."""
    grad_fn = grad(lambda x, y: (x * y).sum(), argnum=(0, 1))
    gs = grad_fn(np.array([2.0, 3.0]), np.array([4.0, 5.0]))
    assert isinstance(gs, tuple) and len(gs) == 2
    np.testing.assert_allclose(gs[0].data, [4.0, 5.0])
    np.testing.assert_allclose(gs[1].data, [2.0, 3.0])


def test_grad_argnum_tuple_preserves_order() -> None:
    """argnum=(2, 0) returns (grad wrt arg 2, grad wrt arg 0) in that order."""
    grad_fn = grad(lambda x, y, z: x * 1.0 + y * 2.0 + z * 3.0, argnum=(2, 0))
    g_z, g_x = grad_fn(1.0, 1.0, 1.0)
    np.testing.assert_allclose(g_z.data, 3.0)
    np.testing.assert_allclose(g_x.data, 1.0)


def test_grad_rejects_non_scalar_output() -> None:
    """fn must return a 0-d Tensor."""
    grad_fn = grad(lambda x: x * x)  # vector in -> vector out, no reduction
    with pytest.raises(ValueError, match="scalar-output"):
        grad_fn(np.array([1.0, 2.0]))


def test_grad_argnum_out_of_range_raises() -> None:
    """argnum=5 with two args is an IndexError, not a silent miss."""
    grad_fn = grad(lambda x, y: (x + y).sum(), argnum=5)
    with pytest.raises(IndexError, match="out of range"):
        grad_fn(1.0, 2.0)


def test_grad_disconnected_input_returns_zeros() -> None:
    """An input that doesn't reach the output gets a zeros-shaped grad rather than None."""
    grad_fn = grad(lambda x, y: (x * x).sum(), argnum=1)
    g = grad_fn(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    np.testing.assert_allclose(g.data, [0.0, 0.0])
