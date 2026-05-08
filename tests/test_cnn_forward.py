"""M5 exit gate: a tiny CNN, composed from primitives, has correct shapes and gradients."""

from __future__ import annotations

import numpy as np

from nanoad import Tensor
from nanoad.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropy,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)


def test_tiny_cnn_forward_shapes() -> None:
    """Conv→BN→ReLU→Pool→Flatten→Linear stack matches expected shape per stage."""
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1),  # (N, 4, 8, 8)
        BatchNorm2d(4),
        ReLU(),
        MaxPool2d(2),  # (N, 4, 4, 4)
        Flatten(),  # (N, 64)
        Linear(64, 10),
    )
    x = Tensor(np.random.randn(3, 1, 8, 8))
    out = model(x)
    assert out.shape == (3, 10)


def test_tiny_cnn_grads_match_finite_diff(grad_check) -> None:
    """End-to-end grad check on a Conv→ReLU→Pool→Flatten→Linear stack (no BN: stateful)."""
    np.random.seed(0)
    conv_w = np.random.randn(2, 1, 3, 3) * 0.1
    conv_b = np.zeros(2)
    fc_w = np.random.randn(2 * 3 * 3, 4) * 0.1
    fc_b = np.zeros(4)

    def fn(x: Tensor, cw: Tensor, cb: Tensor, fw: Tensor, fb: Tensor) -> Tensor:
        from nanoad.ops.conv import conv2d
        from nanoad.ops.pool import max_pool2d

        h = conv2d(x, cw, cb, stride=1, padding=0)  # (N, 2, 4, 4)
        from nanoad.ops.activations import relu

        h = relu(h)
        h = max_pool2d(h, kernel=2)  # (N, 2, 2, 2)
        h = h.reshape(h.shape[0], -1)  # (N, 8)... wait, 2*2*2=8; need fw shape (8,4)
        return h @ fw + fb

    fc_w = np.random.randn(8, 4) * 0.1
    grad_check(
        fn,
        np.random.randn(1, 1, 6, 6),
        conv_w,
        conv_b,
        fc_w,
        fc_b,
    )


def test_cnn_with_cross_entropy_backward_runs() -> None:
    """Smoke: tiny CNN + CE produces finite grads on every parameter."""
    model = Sequential(
        Conv2d(1, 2, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(2 * 2 * 2, 3),
    )
    loss_fn = CrossEntropy()
    x = Tensor(np.random.randn(2, 1, 4, 4))
    targets = np.array([0, 2])
    logits = model(x)
    loss = loss_fn(logits, targets)
    loss.backward()
    for p in model.parameters():
        assert np.all(np.isfinite(p.grad))
