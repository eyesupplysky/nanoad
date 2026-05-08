"""Broadcasting helpers for backward-pass gradient reduction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def unbroadcast(grad: NDArray[np.float64], target_shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Sum grad along axes that were broadcast to lift target_shape up to grad.shape.

    Two-step reduction: first collapse extra leading dims, then collapse axes
    where the target had size 1 but grad has size > 1.
    """
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
