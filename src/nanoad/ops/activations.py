"""Activation functions: relu, tanh.

``tanh``'s VJP recomputes ``tanh(x)`` via the public op so the second derivative chains
back through ``x`` (the recompute cost is one extra ``np.tanh`` per backward — accepted as
the price of HOA correctness). ``relu``'s mask is a piecewise constant; its VJP captures
it as numpy in ``_fwd_ctx`` because the mask itself is non-differentiable, which matches
the mathematical fact that ``d²relu/dx² = 0`` almost everywhere.
"""

from __future__ import annotations

import numpy as np

from nanoad._engine import register_vjp
from nanoad.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """Rectified linear unit, elementwise. Sub-gradient at 0 is taken to be 0 by convention."""
    return Tensor(
        np.maximum(x.data, 0.0),
        _prev=(x,),
        _op="relu",
        _fwd_ctx={"mask": (x.data > 0.0).astype(np.float64)},
    )


def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent, elementwise. d(tanh x)/dx = 1 - tanh^2 x."""
    return Tensor(np.tanh(x.data), _prev=(x,), _op="tanh")


@register_vjp("relu")
def _vjp_relu(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    return (Tensor(fwd_ctx["mask"]) * out_grad,)


@register_vjp("tanh")
def _vjp_tanh(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    (x,) = parents
    t = tanh(x)  # public op — keeps x in the gradient graph for second-order
    return ((1.0 - t * t) * out_grad,)
