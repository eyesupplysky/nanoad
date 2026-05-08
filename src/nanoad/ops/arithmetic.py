"""Forward and backward for arithmetic ops with NumPy broadcasting.

Each op's VJP returns Tensor-valued contributions built from public ops, so gradients are
themselves differentiable. Broadcasting routes through ``unbroadcast`` (a Tensor op),
preserving tape continuity through shape changes.
"""

from __future__ import annotations

from nanoad._engine import register_vjp
from nanoad.ops._broadcast import unbroadcast
from nanoad.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition with broadcasting. d(a+b)/da = 1, d(a+b)/db = 1."""
    return Tensor(a.data + b.data, _prev=(a, b), _op="add")


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction with broadcasting. d(a-b)/da = 1, d(a-b)/db = -1."""
    return Tensor(a.data - b.data, _prev=(a, b), _op="sub")


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication with broadcasting. d(a*b)/da = b, d(a*b)/db = a."""
    return Tensor(a.data * b.data, _prev=(a, b), _op="mul")


def div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division with broadcasting. d(a/b)/da = 1/b, d(a/b)/db = -a/b^2."""
    return Tensor(a.data / b.data, _prev=(a, b), _op="div")


def power(base: Tensor, exponent: float) -> Tensor:
    """Raise tensor to a fixed exponent elementwise. d(x^k)/dx = k * x^(k-1)."""
    return Tensor(
        base.data**exponent,
        _prev=(base,),
        _op="power",
        _fwd_ctx={"exponent": float(exponent)},
    )


@register_vjp("add")
def _vjp_add(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    a, b = parents
    return (unbroadcast(out_grad, a.shape), unbroadcast(out_grad, b.shape))


@register_vjp("sub")
def _vjp_sub(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    a, b = parents
    return (unbroadcast(out_grad, a.shape), unbroadcast(-out_grad, b.shape))


@register_vjp("mul")
def _vjp_mul(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    a, b = parents
    return (unbroadcast(b * out_grad, a.shape), unbroadcast(a * out_grad, b.shape))


@register_vjp("div")
def _vjp_div(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    a, b = parents
    da = unbroadcast(out_grad / b, a.shape)
    db = unbroadcast(-a * out_grad / (b * b), b.shape)
    return (da, db)


@register_vjp("power")
def _vjp_power(
    out_grad: Tensor,
    parents: tuple[Tensor, ...],
    fwd_ctx: dict | None,
) -> tuple[Tensor, ...]:
    assert fwd_ctx is not None
    (base,) = parents
    k = fwd_ctx["exponent"]
    # d(x^k)/dx = k * x^(k-1); use power op on base so the second derivative tape is intact.
    return (k * power(base, k - 1.0) * out_grad,)
