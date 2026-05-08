"""Tape-aware reverse-mode autograd engine.

Each op registers a vector-Jacobian product (VJP) under its op name. ``Tensor.backward``
walks the forward tape in reverse-topological order, calls each node's VJP, and accumulates
gradients into parents *through the public arithmetic ops* — so every gradient itself becomes
a Tensor with its own provenance, enabling higher-order autograd.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoad.tensor import Tensor


# A VJP receives the upstream gradient (as a Tensor), the op's parent tensors,
# and a forward-context dict of any non-Tensor state captured at forward time.
# It returns one gradient Tensor per parent, in the same order as ``parents``.
VjpFn = Callable[
    ["Tensor", tuple["Tensor", ...], "dict | None"],
    tuple["Tensor", ...],
]


_VJP_REGISTRY: dict[str, VjpFn] = {}


def register_vjp(op_name: str) -> Callable[[VjpFn], VjpFn]:
    """Register a VJP under op_name; duplicate registrations are a programming error."""

    def decorator(fn: VjpFn) -> VjpFn:
        if op_name in _VJP_REGISTRY:
            raise RuntimeError(f"VJP already registered for op {op_name!r}")
        _VJP_REGISTRY[op_name] = fn
        return fn

    return decorator


def get_vjp(op_name: str) -> VjpFn:
    """Resolve a VJP by op name; raise if the op forgot to register."""
    try:
        return _VJP_REGISTRY[op_name]
    except KeyError as exc:
        raise RuntimeError(
            f"No VJP registered for op {op_name!r}; the op forward is missing register_vjp."
        ) from exc
