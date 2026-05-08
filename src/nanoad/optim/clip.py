"""Gradient clipping by global L2 norm.

Standard transformer hygiene: between ``loss.backward()`` and ``optimizer.step()``, scale
every parameter's ``.grad`` so the global L2 norm is at most ``max_norm``. This mutates the
gradients' underlying ``.data`` arrays in-place — it does not register a new graph node,
because clipping is conceptually downstream of backward.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from nanoad.tensor import Tensor


def clip_grad_norm(params: Iterable[Tensor], max_norm: float) -> float:
    """Rescale gradients in-place so their joint L2 norm is at most ``max_norm``.

    Returns the pre-clip total norm so callers can log it.
    """
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive; got {max_norm}")
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_sq = sum(float((g.data * g.data).sum()) for g in grads)
    total_norm = float(np.sqrt(total_sq))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g.data = g.data * scale
    return total_norm
