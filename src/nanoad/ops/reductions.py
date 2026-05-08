"""Reductions: sum and mean with axis handling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanoad.tensor import Tensor


def _broadcast_grad_back(
    grad: NDArray[np.float64],
    x_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> NDArray[np.float64]:
    """Re-insert reduced axes (when keepdims=False) and broadcast to x_shape."""
    if not keepdims and axis is not None:
        axes = (axis,) if isinstance(axis, int) else axis
        axes = tuple(a % len(x_shape) for a in axes)
        for ax in sorted(axes):
            grad = np.expand_dims(grad, ax)
    return np.broadcast_to(grad, x_shape)


def sum(
    x: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum elements along axis(es). axis=None reduces over all elements."""
    out = Tensor(x.data.sum(axis=axis, keepdims=keepdims), _prev=(x,), _op="sum")

    def _backward() -> None:
        x.grad += _broadcast_grad_back(out.grad, x.shape, axis, keepdims)

    out._backward = _backward
    return out


def mean(
    x: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Mean of elements along axis(es). axis=None reduces over all elements."""
    out = Tensor(x.data.mean(axis=axis, keepdims=keepdims), _prev=(x,), _op="mean")

    if axis is None:
        n = x.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        n = 1
        for ax in axes:
            n *= x.data.shape[ax]

    def _backward() -> None:
        x.grad += _broadcast_grad_back(out.grad, x.shape, axis, keepdims) / n

    out._backward = _backward
    return out
