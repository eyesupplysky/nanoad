"""Learning-rate schedulers.

A scheduler holds a reference to an optimizer and rewrites its ``lr`` attribute over time.
Call ``scheduler.step()`` once per training step, after ``optimizer.step()`` — the same
convention as PyTorch's ``torch.optim.lr_scheduler``.
"""

from __future__ import annotations

import math
from typing import Protocol


class _HasLr(Protocol):
    lr: float


#   first non-zero warmup value (or the cosine schedule's t=1 sample if warmup_steps == 0)
class CosineWarmupLR:
    """Linear warmup, then cosine decay from base_lr down to min_lr."""

    def __init__(
        self,
        optimizer: _HasLr,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative; got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must exceed warmup_steps ({warmup_steps})"
            )
        self.optimizer = optimizer
        self.warmup_steps: int = warmup_steps
        self.total_steps: int = total_steps
        self.min_lr: float = min_lr
        self.base_lr: float = optimizer.lr
        self.t: int = 0

    def step(self) -> None:
        """Advance one step and rewrite ``optimizer.lr``."""
        self.t += 1
        self.optimizer.lr = self._lr_at(self.t)

    def _lr_at(self, step: int) -> float:
        if step <= self.warmup_steps:
            # Warmup uses max(1, ...) so warmup_steps == 0 falls through cleanly.
            return self.base_lr * step / max(self.warmup_steps, 1)
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
