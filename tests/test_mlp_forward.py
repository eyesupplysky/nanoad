"""M3 exit gate: a 2-layer MLP, composed from primitives, has correct shapes and gradients."""

import numpy as np

from nanoad import Tensor, relu, tanh


def test_two_layer_mlp_forward_shapes_with_relu():
    """Forward pass: (batch, in) -> hidden via ReLU -> (batch, out)."""
    batch, in_dim, hidden, out_dim = 4, 3, 5, 2

    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((batch, in_dim)))
    w1 = Tensor(rng.standard_normal((in_dim, hidden)) * 0.1)
    b1 = Tensor(np.zeros(hidden))
    w2 = Tensor(rng.standard_normal((hidden, out_dim)) * 0.1)
    b2 = Tensor(np.zeros(out_dim))

    h = relu(x @ w1 + b1)
    y = h @ w2 + b2

    assert h.shape == (batch, hidden)
    assert y.shape == (batch, out_dim)


def test_two_layer_mlp_grads_with_tanh(grad_check):
    """Tanh-MLP gradients match finite differences for every parameter and the input.

    Tanh is used instead of ReLU for the gradient check because finite differences
    near the ReLU corner can disagree with the analytic sub-gradient.
    """
    batch, in_dim, hidden, out_dim = 2, 3, 4, 2
    rng = np.random.default_rng(0)

    x = rng.standard_normal((batch, in_dim))
    w1 = rng.standard_normal((in_dim, hidden)) * 0.5
    b1 = rng.standard_normal(hidden) * 0.1
    w2 = rng.standard_normal((hidden, out_dim)) * 0.5
    b2 = rng.standard_normal(out_dim) * 0.1

    def forward(x_t: Tensor, w1_t: Tensor, b1_t: Tensor, w2_t: Tensor, b2_t: Tensor) -> Tensor:
        h = tanh(x_t @ w1_t + b1_t)
        return h @ w2_t + b2_t

    grad_check(forward, x, w1, b1, w2, b2)


def test_mlp_with_reshape_and_transpose():
    """Compose matmul with reshape/transpose to verify they participate cleanly in autograd."""
    rng = np.random.default_rng(1)
    a = Tensor(rng.standard_normal((6, 4)))
    w = Tensor(rng.standard_normal((4, 2)))

    out = (a.reshape(2, 3, 4).transpose(1, 0, 2).reshape(6, 4) @ w).sum()
    out.backward()

    assert a.grad.shape == (6, 4)
    assert w.grad.shape == (4, 2)
    assert not np.allclose(a.grad, 0.0)
    assert not np.allclose(w.grad, 0.0)
