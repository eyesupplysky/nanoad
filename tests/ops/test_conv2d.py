"""Forward shapes and finite-difference gradient checks for conv2d."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.ops.conv import conv2d


def test_conv2d_forward_shape_no_padding() -> None:
    x = Tensor(np.random.randn(2, 3, 5, 5))
    w = Tensor(np.random.randn(4, 3, 3, 3))
    b = Tensor(np.zeros(4))
    out = conv2d(x, w, b, stride=1, padding=0)
    assert out.shape == (2, 4, 3, 3)


def test_conv2d_forward_shape_with_padding() -> None:
    x = Tensor(np.random.randn(1, 1, 4, 4))
    w = Tensor(np.random.randn(2, 1, 3, 3))
    out = conv2d(x, w, None, stride=1, padding=1)
    assert out.shape == (1, 2, 4, 4)


def test_conv2d_forward_stride_two() -> None:
    x = Tensor(np.random.randn(1, 1, 6, 6))
    w = Tensor(np.random.randn(1, 1, 3, 3))
    out = conv2d(x, w, None, stride=2, padding=0)
    assert out.shape == (1, 1, 2, 2)


def test_conv2d_forward_value_pinned() -> None:
    """Hand-derived 1x1x3x3 input convolved with a 1x1x2x2 sum kernel."""
    x = Tensor(np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]))
    w = Tensor(np.ones((1, 1, 2, 2)))
    out = conv2d(x, w, None, stride=1, padding=0)
    expected = np.array([[[[12.0, 16.0], [24.0, 28.0]]]])
    np.testing.assert_allclose(out.data, expected)


def test_conv2d_grads_no_bias_no_padding(grad_check) -> None:
    grad_check(
        lambda x, w: conv2d(x, w, None, stride=1, padding=0),
        np.random.randn(1, 2, 4, 4),
        np.random.randn(3, 2, 3, 3),
    )


def test_conv2d_grads_with_bias_and_padding(grad_check) -> None:
    grad_check(
        lambda x, w, b: conv2d(x, w, b, stride=1, padding=1),
        np.random.randn(2, 2, 4, 4),
        np.random.randn(3, 2, 3, 3),
        np.random.randn(3),
    )


def test_conv2d_grads_stride_two(grad_check) -> None:
    grad_check(
        lambda x, w: conv2d(x, w, None, stride=2, padding=0),
        np.random.randn(1, 1, 5, 5),
        np.random.randn(2, 1, 3, 3),
    )


def test_conv2d_grads_non_square_kernel(grad_check) -> None:
    grad_check(
        lambda x, w: conv2d(x, w, None, stride=1, padding=0),
        np.random.randn(1, 1, 4, 5),
        np.random.randn(2, 1, 2, 3),
    )


def test_conv2d_rejects_3d_input() -> None:
    x = Tensor(np.random.randn(2, 5, 5))
    w = Tensor(np.random.randn(1, 2, 3, 3))
    with pytest.raises(ValueError, match="conv2d expects 4-d x"):
        conv2d(x, w)


def test_conv2d_channel_mismatch_raises() -> None:
    x = Tensor(np.random.randn(1, 3, 5, 5))
    w = Tensor(np.random.randn(2, 4, 3, 3))
    with pytest.raises(ValueError, match="in_channels mismatch"):
        conv2d(x, w)
