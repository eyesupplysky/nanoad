"""AdamW tests: convergence, decoupled-decay invariant, zero_grad behavior."""

from __future__ import annotations

import numpy as np

from nanoad import Tensor
from nanoad.nn import Linear
from nanoad.optim import Adam, AdamW


def test_adamw_step_decreases_loss_on_linear_regression() -> None:
    """Train a single Linear layer to fit y = 2x + 3 with AdamW; loss should drop sharply."""
    np.random.seed(0)
    x_arr = np.random.randn(64, 1)
    y_arr = 2.0 * x_arr + 3.0

    layer = Linear(1, 1)
    opt = AdamW(layer.parameters(), lr=0.05, weight_decay=0.0)

    initial_loss: float | None = None
    final_loss: float | None = None
    for step in range(200):
        x = Tensor(x_arr)
        y = Tensor(y_arr)
        pred = layer(x)
        diff = pred - y
        loss = (diff * diff).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = float(loss.data)
        final_loss = float(loss.data)

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.05


def test_adamw_with_zero_decay_matches_adam_step() -> None:
    """With weight_decay=0, AdamW must produce the same first-step update as Adam."""
    p_a = Tensor(np.array([1.0, -2.0, 3.0]))
    p_a.grad = Tensor(np.array([0.1, 0.2, -0.3]))
    p_b = Tensor(np.array([1.0, -2.0, 3.0]))
    p_b.grad = Tensor(np.array([0.1, 0.2, -0.3]))

    Adam([p_a], lr=0.01, betas=(0.9, 0.999), eps=1e-8).step()
    AdamW([p_b], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()

    np.testing.assert_allclose(p_a.data, p_b.data, atol=1e-12)


def test_adamw_decoupled_decay_shrinks_param_independently_of_grad() -> None:
    """With grad=0, AdamW still shrinks the parameter by lr * weight_decay each step."""
    p = Tensor(np.array([2.0, -4.0]))
    p.grad = Tensor(np.array([0.0, 0.0]))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    opt.step()
    # Decay-only step: p_new = p - lr * weight_decay * p = p * (1 - 0.05) = 0.95 * p.
    np.testing.assert_allclose(p.data, np.array([1.9, -3.8]), atol=1e-12)


def test_adamw_zero_grad_clears_all_params() -> None:
    layer = Linear(3, 2)
    for q in layer.parameters():
        q.grad = Tensor(np.ones_like(q.data))
    opt = AdamW(layer.parameters(), lr=0.01)
    opt.zero_grad()
    for q in layer.parameters():
        assert q.grad is None


def test_adamw_state_is_per_param() -> None:
    """m and v arrays match each param's shape and stay parallel to the param list."""
    layer = Linear(4, 3)
    opt = AdamW(layer.parameters(), lr=0.001)
    params = list(layer.parameters())
    assert len(opt.m) == len(params)
    for moment, p in zip(opt.m, params, strict=True):
        assert moment.shape == p.data.shape
