"""AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

The decay term is applied directly to the parameter, not folded into the gradient — so the
moment estimates ``m`` and ``v`` track only the loss gradient, and the regularization
strength stays independent of the adaptive learning rate.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from nanoad.tensor import Tensor


class AdamW:
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params: list[Tensor] = list(params)
        self.lr: float = lr
        self.b1: float = betas[0]
        self.b2: float = betas[1]
        self.eps: float = eps
        self.weight_decay: float = weight_decay
        self.t: int = 0
        self.m: list[NDArray[np.float64]] = [np.zeros_like(p.data) for p in self.params]
        self.v: list[NDArray[np.float64]] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Update every parameter by one AdamW step using its current .grad."""
        self.t += 1
        b1_correction = 1.0 - self.b1**self.t
        b2_correction = 1.0 - self.b2**self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            # Decoupled weight decay: shrink the parameter directly, before the Adam update.
            if self.weight_decay != 0.0:
                p.data = p.data - self.lr * self.weight_decay * p.data
            g = p.grad.data
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[i] / b1_correction
            v_hat = self.v[i] / b2_correction
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset .grad on every managed parameter."""
        for p in self.params:
            p.zero_grad()
