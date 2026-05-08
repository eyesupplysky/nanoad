"""M8: ``vjp`` — eager forward + cotangent backward as a callable closure."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor, tanh
from nanoad.functional import vjp


def test_vjp_scalar_agrees_with_grad() -> None:
    """For scalar fn with cotangent=1, vjp_fn returns the gradient."""
    out, vjp_fn = vjp(lambda x: x * x, 3.0)
    np.testing.assert_allclose(out.data, 9.0)
    (g,) = vjp_fn(1.0)
    np.testing.assert_allclose(g.data, 6.0)


def test_vjp_matmul_returns_jt_cotangent() -> None:
    """For y = A @ x, cotangent c gives Aᵀ c — the canonical VJP."""
    A_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
    x_data = np.array([7.0, 8.0]).reshape(2, 1)
    A = Tensor(A_data)
    out, vjp_fn = vjp(lambda x: A @ x, x_data)
    c = np.array([1.0, 0.0, 0.0]).reshape(3, 1)
    (g,) = vjp_fn(c)
    np.testing.assert_allclose(g.data.flatten(), A_data.T @ c.flatten())


def test_vjp_multi_arg_returns_one_grad_per_primal() -> None:
    """Two-arg fn returns a 2-tuple from vjp_fn."""
    out, vjp_fn = vjp(lambda x, y: x * y, 2.0, 3.0)
    np.testing.assert_allclose(out.data, 6.0)
    g_x, g_y = vjp_fn(1.0)
    np.testing.assert_allclose(g_x.data, 3.0)
    np.testing.assert_allclose(g_y.data, 2.0)


def test_vjp_closure_supports_multiple_calls() -> None:
    """Calling vjp_fn twice with different cotangents yields independent results."""
    out, vjp_fn = vjp(lambda x: tanh(x), np.array([0.0, 1.0]))
    (g1,) = vjp_fn(np.array([1.0, 0.0]))
    (g2,) = vjp_fn(np.array([0.0, 1.0]))
    # d tanh/dx at x: 1 - tanh²(x). At 0: 1, at 1: 1 - tanh²(1) ≈ 0.4199.
    np.testing.assert_allclose(g1.data, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(g2.data, [0.0, 1.0 - np.tanh(1.0) ** 2], rtol=1e-12)


def test_vjp_linearity_in_cotangent() -> None:
    """vjp_fn(a + b) == vjp_fn(a) + vjp_fn(b)."""
    rng = np.random.default_rng(0)
    A_data = rng.standard_normal((4, 3))
    x_data = rng.standard_normal((3, 1))
    a = rng.standard_normal((4, 1))
    b = rng.standard_normal((4, 1))
    A = Tensor(A_data)
    _, vjp_fn = vjp(lambda x: A @ x, x_data)
    (g_a,) = vjp_fn(a)
    (g_b,) = vjp_fn(b)
    (g_ab,) = vjp_fn(a + b)
    np.testing.assert_allclose(g_ab.data, g_a.data + g_b.data, rtol=1e-12)


def test_vjp_returns_tensor_outputs() -> None:
    """Both ``out`` and the entries of vjp_fn's return are Tensor."""
    out, vjp_fn = vjp(lambda x: x * 2.0, np.array([1.0, 2.0]))
    assert isinstance(out, Tensor)
    (g,) = vjp_fn(np.array([1.0, 1.0]))
    assert isinstance(g, Tensor)


def test_vjp_cotangent_shape_mismatch_raises() -> None:
    """A wrong-shape cotangent fails fast with a shape-naming message."""
    _, vjp_fn = vjp(lambda x: x * 2.0, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="cotangent shape"):
        vjp_fn(np.array([1.0, 2.0, 3.0]))
