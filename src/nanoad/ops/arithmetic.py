"""Forward and backward for arithmetic ops with NumPy broadcasting."""

from __future__ import annotations

from nanoad.ops._broadcast import unbroadcast
from nanoad.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition with broadcasting. d(a+b)/da = 1, d(a+b)/db = 1."""
    out = Tensor(a.data + b.data, _prev=(a, b), _op="+")

    def _backward() -> None:
        a.grad += unbroadcast(out.grad, a.shape)
        b.grad += unbroadcast(out.grad, b.shape)

    out._backward = _backward
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction with broadcasting. d(a-b)/da = 1, d(a-b)/db = -1."""
    out = Tensor(a.data - b.data, _prev=(a, b), _op="-")

    def _backward() -> None:
        a.grad += unbroadcast(out.grad, a.shape)
        b.grad -= unbroadcast(out.grad, b.shape)

    out._backward = _backward
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication with broadcasting. d(a*b)/da = b, d(a*b)/db = a."""
    out = Tensor(a.data * b.data, _prev=(a, b), _op="*")

    def _backward() -> None:
        a.grad += unbroadcast(b.data * out.grad, a.shape)
        b.grad += unbroadcast(a.data * out.grad, b.shape)

    out._backward = _backward
    return out


def div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division with broadcasting. d(a/b)/da = 1/b, d(a/b)/db = -a/b^2."""
    out = Tensor(a.data / b.data, _prev=(a, b), _op="/")

    def _backward() -> None:
        a.grad += unbroadcast(out.grad / b.data, a.shape)
        b.grad += unbroadcast(-a.data / (b.data * b.data) * out.grad, b.shape)

    out._backward = _backward
    return out


def power(base: Tensor, exponent: float) -> Tensor:
    """Raise tensor to a fixed exponent elementwise. d(x^k)/dx = k * x^(k-1)."""
    out = Tensor(base.data**exponent, _prev=(base,), _op=f"**{exponent}")

    def _backward() -> None:
        base.grad += exponent * (base.data ** (exponent - 1)) * out.grad

    out._backward = _backward
    return out
