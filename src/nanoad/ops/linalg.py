"""Linear algebra ops: matmul, transpose, reshape."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nanoad.tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """2-D matrix multiplication. Shapes (M,K) @ (K,N) -> (M,N)."""
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError(f"matmul requires 2-d operands; got shapes {a.shape} and {b.shape}")
    out = Tensor(a.data @ b.data, _prev=(a, b), _op="@")

    def _backward() -> None:
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad

    out._backward = _backward
    return out


def transpose(x: Tensor, axes: Sequence[int] | None = None) -> Tensor:
    """Permute axes of x. Default reverses all axes (matches np.transpose)."""
    perm = tuple(axes) if axes is not None else tuple(range(x.data.ndim))[::-1]
    out = Tensor(np.transpose(x.data, perm), _prev=(x,), _op="T")

    inverse = tuple(int(i) for i in np.argsort(perm))

    def _backward() -> None:
        x.grad += np.transpose(out.grad, inverse)

    out._backward = _backward
    return out


def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape data to shape. Backward reshapes the upstream grad back to original shape."""
    original_shape = x.shape
    out = Tensor(x.data.reshape(shape), _prev=(x,), _op="reshape")

    def _backward() -> None:
        x.grad += out.grad.reshape(original_shape)

    out._backward = _backward
    return out
