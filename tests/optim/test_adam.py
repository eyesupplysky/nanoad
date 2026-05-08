"""Adam tests: convergence on synthetic regression and bias-correction sanity."""

from __future__ import annotations

import numpy as np

from nanoad import Tensor
from nanoad.nn import Linear
from nanoad.optim import Adam


def test_adam_step_decreases_loss_on_linear_regression() -> None:
    """Train a single Linear layer to fit y = 2x + 3 via Adam; loss should drop."""
    np.random.seed(0)
    x_arr = np.random.randn(64, 1)
    y_arr = 2.0 * x_arr + 3.0

    layer = Linear(1, 1)
    opt = Adam(layer.parameters(), lr=0.05)

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


def test_adam_zero_grad_clears_all_params() -> None:
    layer = Linear(3, 2)
    for p in layer.parameters():
        p.grad = Tensor(np.ones_like(p.data))
    opt = Adam(layer.parameters(), lr=0.01)
    opt.zero_grad()
    for p in layer.parameters():
        assert p.grad is None


def test_adam_bias_correction_first_step_magnitude() -> None:
    """At t=1 with grad=g, the update equals lr * g / (|g| + eps) — bias correction makes it
    equal lr in magnitude regardless of beta values (when g is uniform sign)."""
    p = Tensor(np.array([1.0]))
    p.grad = Tensor(np.array([1.0]))
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    before = p.data.copy()
    opt.step()
    delta = float(np.abs(before - p.data).item())
    np.testing.assert_allclose(delta, 0.1, rtol=1e-4)


def test_adam_state_is_per_param() -> None:
    """m and v arrays match each param's shape and stay parallel to the param list."""
    layer = Linear(4, 3)
    opt = Adam(layer.parameters(), lr=0.001)
    params = list(layer.parameters())
    assert len(opt.m) == len(params)
    for moment, p in zip(opt.m, params, strict=True):
        assert moment.shape == p.data.shape
