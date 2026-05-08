"""Broadcasting helpers — Tensor-level ops with mutual VJPs.

The forward and backward of broadcasting are each other's transpose: ``broadcast_to`` tiles
size-1 axes up; ``unbroadcast`` sums them back down. Lifting both to Tensor ops keeps the
backward pass tape-aware so elementwise gradients carry their full provenance through
broadcasting steps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nanoad._engine import register_vjp
from nanoad.tensor import Tensor


def _unbroadcast_array(
    grad: NDArray[np.float64],
    target_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Pure-numpy unbroadcast: sum grad along axes that broadcast to lift target up to grad."""
    extra = grad.ndim - len(target_shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes_to_sum = tuple(
        i
        for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape, strict=True))
        if t_dim == 1 and g_dim != 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)
    return grad


def unbroadcast(grad: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    """Sum ``grad`` along axes that were broadcast forward, lifting target_shape up to grad.shape.

    Two-step reduction: first collapse extra leading dims, then collapse axes
    where the target had size 1 but grad has size > 1.
    """
    out_data = _unbroadcast_array(grad.data, target_shape)
    return Tensor(
        out_data,
        _prev=(grad,),
        _op="unbroadcast",
        _fwd_ctx={"original_shape": grad.shape},
    )


def broadcast_to(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Broadcast x to ``shape`` (the inverse of unbroadcast)."""
    out_data = np.broadcast_to(x.data, shape).copy()
    return Tensor(
        out_data,
        _prev=(x,),
        _op="broadcast_to",
        _fwd_ctx={"original_shape": x.shape},
    )


@register_vjp("unbroadcast")
def _vjp_unbroadcast(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    return (broadcast_to(out_grad, fwd_ctx["original_shape"]),)


@register_vjp("broadcast_to")
def _vjp_broadcast_to(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    return (unbroadcast(out_grad, fwd_ctx["original_shape"]),)
