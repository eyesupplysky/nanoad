"""Tensor class and reverse-mode autograd engine."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Tensor:
    """N-d array participating in a reverse-mode autograd graph; backed by float64 numpy."""

    __slots__ = ("_backward", "_op", "_prev", "data", "grad")

    def __init__(
        self,
        data: ArrayLike,
        _prev: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: NDArray[np.float64] = np.array(data, dtype=np.float64)
        self.grad: NDArray[np.float64] = np.zeros_like(self.data)
        self._prev: tuple[Tensor, ...] = _prev
        self._backward: Callable[[], None] = _noop_backward
        self._op: str = _op

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying data array."""
        return self.data.ndim

    def backward(self) -> None:
        """Populate .grad on every reachable node via reverse-mode autodiff."""
        if self.data.ndim != 0:
            raise RuntimeError(
                f"backward() requires scalar output; got shape {self.data.shape}. "
                "Reduce to a scalar (e.g., .sum()) before calling backward()."
            )
        topo = _topological_order(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Reset only this node's gradient. Other nodes are not touched."""
        self.grad = np.zeros_like(self.data)

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Sum along axis(es)."""
        from nanoad.ops.reductions import sum as _sum

        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Mean along axis(es)."""
        from nanoad.ops.reductions import mean as _mean

        return _mean(self, axis=axis, keepdims=keepdims)

    @property
    def T(self) -> Tensor:  # noqa: N802 — matches NumPy/PyTorch convention for transpose
        """Reverse all axes; shorthand for transpose() with no arguments."""
        from nanoad.ops.linalg import transpose

        return transpose(self)

    def transpose(self, *axes: int) -> Tensor:
        """Permute axes. With no args, reverses all axes."""
        from nanoad.ops.linalg import transpose

        return transpose(self, axes if axes else None)

    def reshape(self, *shape: int) -> Tensor:
        """Reshape data to shape (variadic). One dim may be -1 to be inferred."""
        from nanoad.ops.linalg import reshape

        return reshape(self, shape)

    def __add__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import add

        return add(self, _coerce(other))

    def __radd__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import add

        return add(_coerce(other), self)

    def __sub__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import sub

        return sub(self, _coerce(other))

    def __rsub__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import sub

        return sub(_coerce(other), self)

    def __mul__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import mul

        return mul(self, _coerce(other))

    def __rmul__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import mul

        return mul(_coerce(other), self)

    def __truediv__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import div

        return div(self, _coerce(other))

    def __rtruediv__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.arithmetic import div

        return div(_coerce(other), self)

    def __pow__(self, exponent: float) -> Tensor:
        from nanoad.ops.arithmetic import power

        return power(self, exponent)

    def __matmul__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.linalg import matmul

        return matmul(self, _coerce(other))

    def __rmatmul__(self, other: ArrayLike | Tensor) -> Tensor:
        from nanoad.ops.linalg import matmul

        return matmul(_coerce(other), self)

    def __neg__(self) -> Tensor:
        from nanoad.ops.arithmetic import mul

        return mul(self, _coerce(-1.0))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, data={self.data}, grad={self.grad})"


def _coerce(value: ArrayLike | Tensor) -> Tensor:
    """Wrap a Python number or array as a leaf Tensor; pass Tensors through unchanged."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


def _noop_backward() -> None:
    """Default _backward for leaf tensors — nothing to propagate to."""


def _topological_order(root: Tensor) -> list[Tensor]:
    """Iterative post-order DFS from root; each Tensor appears exactly once."""
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, emit = stack.pop()
        if emit:
            order.append(node)
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.append((node, True))
        for parent in node._prev:
            if id(parent) not in visited:
                stack.append((parent, False))
    return order
