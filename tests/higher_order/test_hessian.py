"""M7 exit gate: tape-aware backward enables higher-order autograd.

Each test calls ``backward`` twice on the same forward graph: once on a scalar loss to
populate ``param.grad`` (a Tensor with provenance), then on a function of that gradient to
get the second derivative. The pattern is: ``loss.backward()`` → ``g = param.grad`` →
``param.zero_grad()`` → ``f(g).backward()`` → ``param.grad`` is now the second-order term.
"""

from __future__ import annotations

import numpy as np

from nanoad import Tensor, relu, tanh


def test_second_derivative_of_x_squared() -> None:
    """d²(x²)/dx² = 2 — the smallest possible HOA test."""
    x = Tensor(3.0)
    y = x * x
    y.backward()
    first_grad = x.grad
    assert first_grad is not None
    np.testing.assert_allclose(first_grad.data, 6.0)

    x.zero_grad()
    first_grad.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, 2.0)


def test_third_derivative_of_x_cubed() -> None:
    """d³(x³)/dx³ = 6. Three nested backwards through ``power``."""
    x = Tensor(2.0)
    y = x**3.0
    y.backward()
    g1 = x.grad  # 3x²
    assert g1 is not None
    np.testing.assert_allclose(g1.data, 12.0)

    x.zero_grad()
    g1.backward()
    g2 = x.grad  # 6x
    assert g2 is not None
    np.testing.assert_allclose(g2.data, 12.0)

    x.zero_grad()
    g2.backward()
    g3 = x.grad  # 6
    assert g3 is not None
    np.testing.assert_allclose(g3.data, 6.0)


def test_second_derivative_of_tanh() -> None:
    """d²(tanh x)/dx² = -2 tanh(x) (1 - tanh²(x))."""
    x = Tensor(0.5)
    y = tanh(x)
    y.backward()
    g = x.grad
    assert g is not None

    x.zero_grad()
    g.backward()
    t = float(np.tanh(0.5))
    expected = -2.0 * t * (1.0 - t * t)
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.data, expected, rtol=1e-10)


def test_hessian_of_quadratic_form() -> None:
    """Hessian of f(x) = xᵀAx is A + Aᵀ — recovered column-by-column via HVP with basis vectors."""
    n = 3
    rng = np.random.default_rng(0)
    A_data = rng.standard_normal((n, n))
    x_data = rng.standard_normal((n, 1))

    expected_H = A_data + A_data.T

    A = Tensor(A_data)
    x = Tensor(x_data)
    f = ((x.transpose() @ A) @ x).sum()
    f.backward()
    g = x.grad
    assert g is not None
    # Sanity: ∇f = (A + Aᵀ) x
    np.testing.assert_allclose(g.data.flatten(), (expected_H @ x_data).flatten(), rtol=1e-10)

    H = np.zeros((n, n))
    for j in range(n):
        x.zero_grad()
        e_j = Tensor(np.eye(n)[:, j].reshape(n, 1))
        # gᵀe_j is a scalar — backward gives ∂(g·e_j)/∂x = H[j, :]
        (g * e_j).sum().backward()
        assert x.grad is not None
        H[j, :] = x.grad.data.flatten()

    np.testing.assert_allclose(H, expected_H, rtol=1e-10)


def test_hvp_through_two_layer_mlp_matches_explicit_product() -> None:
    """Hessian-vector product on a small ReLU-free MLP loss agrees with the explicit Hv."""
    rng = np.random.default_rng(1)
    n_in, n_hidden, n_out, batch = 3, 4, 2, 2

    x_data = rng.standard_normal((batch, n_in))
    y_data = rng.standard_normal((batch, n_out))
    w1_data = rng.standard_normal((n_in, n_hidden)) * 0.3
    w2_data = rng.standard_normal((n_hidden, n_out)) * 0.3

    def loss_fn(w1: Tensor, w2: Tensor) -> Tensor:
        x = Tensor(x_data)
        y = Tensor(y_data)
        h = tanh(x @ w1)
        pred = h @ w2
        diff = pred - y
        return (diff * diff).mean()

    # Build the full forward, then take first derivative w.r.t. w1.
    w1 = Tensor(w1_data)
    w2 = Tensor(w2_data)
    loss = loss_fn(w1, w2)
    loss.backward()
    g_w1 = w1.grad
    assert g_w1 is not None

    # Pick a fixed direction v of the same shape as w1.
    v = rng.standard_normal(w1_data.shape)
    v_t = Tensor(v)

    # HVP via grad-of-grad-dot-v: the second backward populates w1.grad with H @ v.
    w1.zero_grad()
    (g_w1 * v_t).sum().backward()
    hvp_autograd = w1.grad
    assert hvp_autograd is not None

    # HVP via finite differences: H v ≈ (∇f(w1 + ε v) - ∇f(w1 - ε v)) / (2ε).
    eps = 1e-5
    w1_plus = Tensor(w1_data + eps * v)
    loss_plus = loss_fn(w1_plus, Tensor(w2_data))
    loss_plus.backward()
    g_plus = w1_plus.grad
    assert g_plus is not None

    w1_minus = Tensor(w1_data - eps * v)
    loss_minus = loss_fn(w1_minus, Tensor(w2_data))
    loss_minus.backward()
    g_minus = w1_minus.grad
    assert g_minus is not None

    hvp_finite_diff = (g_plus.data - g_minus.data) / (2.0 * eps)
    np.testing.assert_allclose(hvp_autograd.data, hvp_finite_diff, atol=1e-5, rtol=1e-4)


def test_grad_of_relu_grad_is_disconnected() -> None:
    """ReLU's mask is captured as a numpy snapshot in _fwd_ctx (it's non-differentiable),
    so the second-order path through ReLU does not reach x — the engine leaves x.grad as
    None after the second backward. This matches the mathematical fact that
    d²(relu(x))/dx² = 0 almost everywhere.
    """
    x = Tensor(np.array([-2.0, -0.5, 0.5, 2.0]))
    y = relu(x).sum()
    y.backward()
    g = x.grad
    assert g is not None
    np.testing.assert_allclose(g.data, [0.0, 0.0, 1.0, 1.0])

    x.zero_grad()
    g.sum().backward()
    # x is unreachable from g's tape (the mask was snapshotted), so the engine never visits it.
    assert x.grad is None
