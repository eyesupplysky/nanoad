"""Forward shapes and grad checks for max_pool2d."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.ops.pool import max_pool2d


def test_max_pool2d_forward_shape_default_stride() -> None:
    x = Tensor(np.random.randn(2, 3, 4, 4))
    out = max_pool2d(x, kernel=2)
    assert out.shape == (2, 3, 2, 2)


def test_max_pool2d_forward_value_pinned() -> None:
    """Hand-derived 1x1x4x4 → 1x1x2x2 with kernel=2, stride=2."""
    x = Tensor(np.array([[[[1.0, 3.0, 2.0, 0.0], [4.0, 2.0, 1.0, 5.0], [0.0, 1.0, 6.0, 7.0], [2.0, 3.0, 8.0, 4.0]]]]))
    out = max_pool2d(x, kernel=2)
    expected = np.array([[[[4.0, 5.0], [3.0, 8.0]]]])
    np.testing.assert_allclose(out.data, expected)


def test_max_pool2d_overlapping_stride(grad_check) -> None:
    grad_check(
        lambda x: max_pool2d(x, kernel=3, stride=1),
        np.random.randn(1, 1, 5, 5),
    )


def test_max_pool2d_grads_no_overlap(grad_check) -> None:
    grad_check(
        lambda x: max_pool2d(x, kernel=2),
        np.random.randn(1, 2, 4, 4) + 0.5 * np.arange(32).reshape(1, 2, 4, 4),
    )


def test_max_pool2d_argmax_routing_hand_check() -> None:
    """Backward routes 1.0 of grad to the single argmax of each window."""
    x = Tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]]))
    out = max_pool2d(x, kernel=2)  # max is at (1,1) -> 4
    out.sum().backward()
    expected = np.array([[[[0.0, 0.0], [0.0, 1.0]]]])
    np.testing.assert_allclose(x.grad.data, expected)


def test_max_pool2d_rejects_3d_input() -> None:
    x = Tensor(np.random.randn(3, 4, 4))
    with pytest.raises(ValueError, match="max_pool2d expects 4-d"):
        max_pool2d(x, kernel=2)
