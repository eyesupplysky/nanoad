"""Linear algebra ops: matmul, transpose, reshape.

Each op's VJP composes Tensor-valued operations on the upstream gradient so the gradient
DAG is itself differentiable. Per-op shape parameters live in ``_fwd_ctx`` so the registry
key remains a single op name.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nanoad._engine import register_vjp
from nanoad.tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """2-D matrix multiplication. Shapes (M,K) @ (K,N) -> (M,N)."""
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError(f"matmul requires 2-d operands; got shapes {a.shape} and {b.shape}")
    return Tensor(a.data @ b.data, _prev=(a, b), _op="matmul")


def transpose(x: Tensor, axes: Sequence[int] | None = None) -> Tensor:
    """Permute axes of x. Default reverses all axes (matches np.transpose)."""
    perm = tuple(axes) if axes is not None else tuple(range(x.data.ndim))[::-1]
    return Tensor(
        np.transpose(x.data, perm),
        _prev=(x,),
        _op="transpose",
        _fwd_ctx={"perm": perm},
    )


def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape data to shape. Backward reshapes the upstream grad back to the original shape."""
    return Tensor(
        x.data.reshape(shape),
        _prev=(x,),
        _op="reshape",
        _fwd_ctx={"original_shape": x.shape},
    )


@register_vjp("matmul")
def _vjp_matmul(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    a, b = parents
    # d(A@B)/dA = grad @ B^T, d(A@B)/dB = A^T @ grad — composed of public ops only.
    return (out_grad @ transpose(b), transpose(a) @ out_grad)


@register_vjp("transpose")
def _vjp_transpose(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    perm = fwd_ctx["perm"]
    inverse = tuple(int(i) for i in np.argsort(perm))
    return (transpose(out_grad, inverse),)


@register_vjp("reshape")
def _vjp_reshape(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    return (reshape(out_grad, fwd_ctx["original_shape"]),)
