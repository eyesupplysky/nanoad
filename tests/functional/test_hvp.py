"""M8: ``hvp`` — Hessian-vector product via two reverse passes."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor, tanh
from nanoad.functional import hvp


def test_hvp_quadratic_form_recovers_symmetrized_matrix_action() -> None:
    """For f(x) = xᵀAx, H·v = (A + Aᵀ)v."""
    n = 3
    rng = np.random.default_rng(0)
    A_data = rng.standard_normal((n, n))
    x_data = rng.standard_normal((n, 1))
    v_data = rng.standard_normal((n, 1))
    A = Tensor(A_data)

    def f(x: Tensor) -> Tensor:
        return ((x.transpose() @ A) @ x).sum()

    out, hv = hvp(f, (x_data,), (v_data,))
    expected = (A_data + A_data.T) @ v_data
    np.testing.assert_allclose(hv[0].data, expected, rtol=1e-10)


def test_hvp_diagonal_quadratic_is_2v() -> None:
    """For f(x) = sum(x²), H = 2I, so H·v = 2v."""
    rng = np.random.default_rng(1)
    x_data = rng.standard_normal(5)
    v_data = rng.standard_normal(5)
    out, hv = hvp(lambda x: (x * x).sum(), (x_data,), (v_data,))
    np.testing.assert_allclose(hv[0].data, 2.0 * v_data, rtol=1e-12)


def test_hvp_matches_finite_difference_through_tanh_mlp() -> None:
    """HVP through a tanh-MLP loss matches finite differences on the gradient."""
    rng = np.random.default_rng(2)
    n_in, n_hidden, n_out, batch = 3, 4, 2, 2
    x_data = rng.standard_normal((batch, n_in))
    y_data = rng.standard_normal((batch, n_out))
    w1_data = rng.standard_normal((n_in, n_hidden)) * 0.3
    w2_data = rng.standard_normal((n_hidden, n_out)) * 0.3
    v_data = rng.standard_normal(w1_data.shape)

    def loss(w1: Tensor, w2: Tensor) -> Tensor:
        x = Tensor(x_data)
        y = Tensor(y_data)
        h = tanh(x @ w1)
        pred = h @ w2
        diff = pred - y
        return (diff * diff).mean()

    _, hv = hvp(loss, (w1_data, w2_data), (v_data, np.zeros_like(w2_data)))

    # Reference: HVP via finite differences on the analytical gradient.
    eps = 1e-5
    w1_plus = Tensor(w1_data + eps * v_data)
    loss(w1_plus, Tensor(w2_data)).backward()
    g_plus = w1_plus.grad
    w1_minus = Tensor(w1_data - eps * v_data)
    loss(w1_minus, Tensor(w2_data)).backward()
    g_minus = w1_minus.grad
    assert g_plus is not None and g_minus is not None
    expected = (g_plus.data - g_minus.data) / (2.0 * eps)

    np.testing.assert_allclose(hv[0].data, expected, atol=1e-5, rtol=1e-4)


def test_hvp_returns_tuple_matching_primals_count() -> None:
    """One hvp entry per primal, regardless of which directions are zero."""
    out, hv = hvp(lambda a, b: (a * a + b * b).sum(), (1.0, 2.0), (1.0, 0.0))
    assert isinstance(hv, tuple) and len(hv) == 2
    np.testing.assert_allclose(hv[0].data, 2.0)
    np.testing.assert_allclose(hv[1].data, 0.0)


def test_hvp_rejects_non_scalar_output() -> None:
    """fn must return a 0-d Tensor for hvp to be defined."""
    with pytest.raises(ValueError, match="scalar-output"):
        hvp(lambda x: x * x, (np.array([1.0, 2.0]),), (np.array([1.0, 0.0]),))


def test_hvp_rejects_mismatched_lengths() -> None:
    """primals and vector must agree on length."""
    with pytest.raises(ValueError, match="same length"):
        hvp(lambda x, y: (x + y).sum(), (1.0, 2.0), (1.0,))


def test_hvp_rejects_mismatched_shapes() -> None:
    """Each vector entry's shape must match its primal's shape."""
    with pytest.raises(ValueError, match="primal\\[0\\] shape"):
        hvp(lambda x: (x * x).sum(), (np.zeros(3),), (np.zeros(4),))
