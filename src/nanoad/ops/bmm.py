"""Batched matrix multiplication.

Generalizes ``matmul`` to ≥2-d operands: contracts the last two axes
``(..., M, K) @ (..., K, N) -> (..., M, N)`` and broadcasts the leading dims per NumPy
rules. The VJPs feed gradients through public ops only — ``bmm`` itself, the existing
``transpose``, and ``unbroadcast`` — so the gradient DAG stays differentiable.
"""

from __future__ import annotations

import numpy as np

from nanoad._engine import register_vjp
from nanoad.ops._broadcast import unbroadcast
from nanoad.ops.linalg import transpose
from nanoad.tensor import Tensor


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """Batched matrix multiplication. ``a`` and ``b`` must each have ndim >= 2."""
    if a.data.ndim < 2 or b.data.ndim < 2:
        raise ValueError(f"bmm requires at least 2-d operands; got {a.shape} and {b.shape}")
    return Tensor(np.matmul(a.data, b.data), _prev=(a, b), _op="bmm")


def _swap_last_two(x: Tensor) -> Tensor:
    """Permute the last two axes of x (matrix-transpose for batched data)."""
    n = x.data.ndim
    perm = tuple(range(n - 2)) + (n - 1, n - 2)
    return transpose(x, perm)


@register_vjp("bmm")
def _vjp_bmm(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    a, b = parents
    # Same shape rules as matmul, lifted to batched: dA = dC @ B^T, dB = A^T @ dC.
    # Leading dims may have broadcast forward, so route grads through unbroadcast.
    da = unbroadcast(bmm(out_grad, _swap_last_two(b)), a.shape)
    db = unbroadcast(bmm(_swap_last_two(a), out_grad), b.shape)
    return (da, db)
