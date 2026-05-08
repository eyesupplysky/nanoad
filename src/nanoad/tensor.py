"""Tensor class and tape-aware reverse-mode autograd engine.

A Tensor wraps a float64 NumPy array. Forward ops record the result Tensor's parents in
``_prev`` and tag it with an ``_op`` registry key (and optional ``_fwd_ctx`` for non-Tensor
state). ``backward`` walks the forward tape in reverse-topological order, dispatches each
op's registered VJP, and accumulates parent gradients via the public arithmetic ops — so
``Tensor.grad`` is itself a Tensor whose own ``_prev`` traces *how* it was computed,
enabling higher-order autograd.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nanoad._engine import get_vjp


class Tensor:
    """N-d array participating in a reverse-mode autograd graph; backed by float64 numpy.

    Two graphs share these Tensors. The *forward graph* is built by user code and is reachable
    via ``_prev``; the *gradient graph* is built by ``backward()`` and is reachable via the
    ``_prev`` of each gradient Tensor stored in ``.grad``. Calling ``backward`` on a function
    of someone's ``.grad`` walks the gradient graph and yields second-order gradients.
    """

    __slots__ = ("_fwd_ctx", "_op", "_prev", "data", "grad")

    def __init__(
        self,
        data: ArrayLike,
        _prev: tuple[Tensor, ...] = (),
        _op: str = "",
        _fwd_ctx: dict[str, Any] | None = None,
    ) -> None:
        self.data: NDArray[np.float64] = np.array(data, dtype=np.float64)
        self.grad: Tensor | None = None
        self._prev: tuple[Tensor, ...] = _prev
        self._op: str = _op
        self._fwd_ctx: dict[str, Any] | None = _fwd_ctx

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying data array."""
        return self.data.ndim

    def backward(self) -> None:
        """Build the gradient graph by walking the forward tape in reverse-topological order.

        Resets intermediate gradients so consecutive (higher-order) backward calls do not
        pollute each other, then seeds ``self.grad`` with ones and dispatches each non-leaf
        node's registered VJP. Parent contributions accumulate through the public ``+`` op,
        so the resulting gradient graph is itself a differentiable Tensor DAG.
        Leaf gradients are *not* reset — they accumulate across backwards (the training
        pattern); call ``zero_grad`` on a leaf before a fresh higher-order pass.
        """
        if self.data.ndim != 0:
            raise RuntimeError(
                f"backward() requires scalar output; got shape {self.data.shape}. "
                "Reduce to a scalar (e.g., .sum()) before calling backward()."
            )
        topo = _topological_order(self)
        for node in topo:
            if node._prev:
                node.grad = None
        self.grad = Tensor(np.ones_like(self.data))
        for node in reversed(topo):
            if not node._op:
                continue
            assert node.grad is not None  # guaranteed by topo-order accumulation
            vjp = get_vjp(node._op)
            parent_grads = vjp(node.grad, node._prev, node._fwd_ctx)
            if len(parent_grads) != len(node._prev):
                raise RuntimeError(
                    f"VJP for {node._op!r} returned {len(parent_grads)} grads "
                    f"but op has {len(node._prev)} parents"
                )
            for parent, contribution in zip(node._prev, parent_grads, strict=True):
                if parent.grad is None:
                    parent.grad = contribution
                else:
                    parent.grad = parent.grad + contribution

    def zero_grad(self) -> None:
        """Drop this node's gradient; matches PyTorch ``set_to_none=True``."""
        self.grad = None

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
