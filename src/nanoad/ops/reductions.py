"""Reductions: sum and mean with axis handling.

Backward of a reduction is a broadcast: insert any axes that were collapsed (when
``keepdims=False``) and tile back up to the original input shape. The VJPs use the public
``reshape`` and ``broadcast_to`` ops so the gradient graph stays differentiable.
"""

from __future__ import annotations

from nanoad._engine import register_vjp
from nanoad.ops._broadcast import broadcast_to
from nanoad.ops.linalg import reshape
from nanoad.tensor import Tensor


def _expanded_shape(
    reduced_shape: tuple[int, ...],
    x_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> tuple[int, ...]:
    """Re-insert size-1 axes at the reduced positions (if keepdims=False)."""
    if keepdims or axis is None:
        return reduced_shape
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes_resolved = tuple(a % len(x_shape) for a in axes)
    expanded = list(reduced_shape)
    for ax in sorted(axes_resolved):
        expanded.insert(ax, 1)
    return tuple(expanded)


def sum(
    x: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum elements along axis(es). axis=None reduces over all elements."""
    return Tensor(
        x.data.sum(axis=axis, keepdims=keepdims),
        _prev=(x,),
        _op="sum",
        _fwd_ctx={
            "axis": axis,
            "keepdims": keepdims,
            "x_shape": x.shape,
        },
    )


def mean(
    x: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Mean of elements along axis(es). axis=None reduces over all elements."""
    if axis is None:
        n = x.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        n = 1
        for ax in axes:
            n *= x.data.shape[ax]
    return Tensor(
        x.data.mean(axis=axis, keepdims=keepdims),
        _prev=(x,),
        _op="mean",
        _fwd_ctx={
            "axis": axis,
            "keepdims": keepdims,
            "x_shape": x.shape,
            "n": int(n),
        },
    )


@register_vjp("sum")
def _vjp_sum(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    (x,) = parents
    expanded = _expanded_shape(out_grad.shape, fwd_ctx["x_shape"], fwd_ctx["axis"], fwd_ctx["keepdims"])
    g = out_grad if expanded == out_grad.shape else reshape(out_grad, expanded)
    return (broadcast_to(g, fwd_ctx["x_shape"]),)


@register_vjp("mean")
def _vjp_mean(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    (x,) = parents
    expanded = _expanded_shape(out_grad.shape, fwd_ctx["x_shape"], fwd_ctx["axis"], fwd_ctx["keepdims"])
    g = out_grad if expanded == out_grad.shape else reshape(out_grad, expanded)
    return (broadcast_to(g / fwd_ctx["n"], fwd_ctx["x_shape"]),)
