"""M8: ``jvp`` — forward-mode directional derivative via the double-VJP transposition trick.

nanoad has only reverse-mode rules. ``jvp`` recovers ``J·v`` by building a fresh
leaf ``u``, taking the gradient of ``(out·u).sum()`` wrt the primals to get a
``u``-parameterized expression for ``Jᵀu``, then differentiating
``(grad · v).sum()`` wrt ``u`` — which by the symmetry of the bilinear form
yields ``J·v``. First-order JVP works through every op (including ``relu`` /
``conv2d`` / ``max_pool2d`` / ``cross_entropy``) because their VJPs multiply
the captured numpy state into ``out_grad`` via the public ``*`` op, preserving
the backward chain back to ``u``. The second-order disconnect documented in
``RISKS.md`` only bites at HVP / Hessian time.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor, relu, tanh
from nanoad.functional import jvp


def test_jvp_of_x_squared_at_three() -> None:
    """f(x) = x², J·v at x=3 with v=1 is 2x·v = 6."""
    out, jv = jvp(lambda x: x * x, (3.0,), (1.0,))
    np.testing.assert_allclose(out.data, 9.0)
    np.testing.assert_allclose(jv.data, 6.0)


def test_jvp_linear_map_is_constant_in_primal() -> None:
    """For y = A @ x, J·v = A·v independent of x."""
    rng = np.random.default_rng(0)
    A_data = rng.standard_normal((3, 4))
    x_data = rng.standard_normal((4, 1))
    v_data = rng.standard_normal((4, 1))
    A = Tensor(A_data)
    out, jv = jvp(lambda x: (A @ x).sum(), (x_data,), (v_data,))
    expected = (A_data @ v_data).sum()
    np.testing.assert_allclose(jv.data, expected, rtol=1e-12)


def test_jvp_tanh_matches_closed_form() -> None:
    """For y = tanh(x), J·v = (1 - tanh²(x)) ⊙ v."""
    rng = np.random.default_rng(1)
    x_data = rng.standard_normal(5)
    v_data = rng.standard_normal(5)
    out, jv = jvp(lambda x: tanh(x).sum(), (x_data,), (v_data,))
    expected = ((1.0 - np.tanh(x_data) ** 2) * v_data).sum()
    np.testing.assert_allclose(jv.data, expected, rtol=1e-10)


def test_jvp_linearity_in_tangent() -> None:
    """jvp(f, x, a + b) == jvp(f, x, a) + jvp(f, x, b)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(3)
    a = rng.standard_normal(3)
    b = rng.standard_normal(3)

    def f(x: Tensor) -> Tensor:
        return tanh(x).sum()

    _, jv_a = jvp(f, (x,), (a,))
    _, jv_b = jvp(f, (x,), (b,))
    _, jv_ab = jvp(f, (x,), (a + b,))
    np.testing.assert_allclose(jv_ab.data, jv_a.data + jv_b.data, rtol=1e-10)


def test_jvp_matches_finite_difference_through_mlp() -> None:
    """JVP through a small tanh-MLP forward pass agrees with central finite differences."""
    rng = np.random.default_rng(3)
    n_in, n_hidden = 3, 4
    w1_data = rng.standard_normal((n_in, n_hidden)) * 0.3
    x_data = rng.standard_normal((1, n_in))
    v_data = rng.standard_normal(w1_data.shape) * 0.1

    def fwd(w1: Tensor) -> Tensor:
        x = Tensor(x_data)
        return tanh(x @ w1).sum()

    _, jv = jvp(fwd, (w1_data,), (v_data,))

    eps = 1e-5
    f_plus = fwd(Tensor(w1_data + eps * v_data)).data
    f_minus = fwd(Tensor(w1_data - eps * v_data)).data
    expected = (f_plus - f_minus) / (2.0 * eps)
    np.testing.assert_allclose(jv.data, expected, atol=1e-6, rtol=1e-5)


def test_jvp_multi_arg_sums_directional_contributions() -> None:
    """For f(x, y) = x² + y², J·(v_x, v_y) = 2x·v_x + 2y·v_y."""
    out, jv = jvp(lambda x, y: x * x + y * y, (3.0, 4.0), (1.0, 1.0))
    # Total derivative = 2·3·1 + 2·4·1 = 14.
    np.testing.assert_allclose(jv.data, 14.0)


def test_jvp_through_relu_recovers_mask_dot_tangent() -> None:
    """First-order JVP works through relu — output is mask·v (the disconnect is HVP-only)."""
    x_data = np.array([0.5, -0.5, 2.0, -1.0])
    v_data = np.array([1.0, 1.0, 0.5, 0.25])
    out, jv = jvp(lambda x: relu(x).sum(), (x_data,), (v_data,))
    mask = (x_data > 0.0).astype(np.float64)
    np.testing.assert_allclose(jv.data, float(np.sum(mask * v_data)))


def test_jvp_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        jvp(lambda x, y: x + y, (1.0, 2.0), (1.0,))


def test_jvp_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="primal\\[0\\] shape"):
        jvp(lambda x: x.sum(), (np.zeros(3),), (np.zeros(4),))
