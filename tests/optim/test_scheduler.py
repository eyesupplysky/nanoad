"""CosineWarmupLR: schedule shape, warmup ramp, decay floor, edge cases."""

from __future__ import annotations

import math

import pytest

from nanoad.optim import Adam
from nanoad.optim.scheduler import CosineWarmupLR
from nanoad.tensor import Tensor


class _DummyOpt:
    def __init__(self, lr: float) -> None:
        self.lr = lr


def test_scheduler_warmup_ramps_linearly_from_zero() -> None:
    opt = _DummyOpt(lr=0.1)
    sched = CosineWarmupLR(opt, warmup_steps=4, total_steps=20)
    lrs = []
    for _ in range(4):
        sched.step()
        lrs.append(opt.lr)
    assert lrs == pytest.approx([0.025, 0.050, 0.075, 0.100])


def test_scheduler_post_warmup_starts_at_base_lr_and_descends() -> None:
    opt = _DummyOpt(lr=0.1)
    sched = CosineWarmupLR(opt, warmup_steps=2, total_steps=10)
    for _ in range(2):
        sched.step()
    assert opt.lr == pytest.approx(0.1)
    last = opt.lr
    for _ in range(8):
        sched.step()
        assert opt.lr <= last + 1e-12
        last = opt.lr


def test_scheduler_reaches_min_lr_at_total_steps() -> None:
    opt = _DummyOpt(lr=0.1)
    sched = CosineWarmupLR(opt, warmup_steps=2, total_steps=10, min_lr=0.001)
    for _ in range(10):
        sched.step()
    assert opt.lr == pytest.approx(0.001, abs=1e-12)


def test_scheduler_clamps_at_min_lr_past_total_steps() -> None:
    opt = _DummyOpt(lr=0.1)
    sched = CosineWarmupLR(opt, warmup_steps=2, total_steps=10, min_lr=0.001)
    for _ in range(15):
        sched.step()
    assert opt.lr == pytest.approx(0.001, abs=1e-12)


def test_scheduler_zero_warmup_starts_directly_at_base_lr_first_sample() -> None:
    """warmup_steps=0 means step 1 is already the first cosine-schedule sample."""
    opt = _DummyOpt(lr=0.1)
    sched = CosineWarmupLR(opt, warmup_steps=0, total_steps=4, min_lr=0.0)
    sched.step()
    expected = 0.5 * 0.1 * (1.0 + math.cos(math.pi * 0.25))
    assert opt.lr == pytest.approx(expected, abs=1e-12)


def test_scheduler_rejects_total_le_warmup() -> None:
    opt = _DummyOpt(lr=0.1)
    with pytest.raises(ValueError, match="total_steps"):
        CosineWarmupLR(opt, warmup_steps=10, total_steps=10)


def test_scheduler_rejects_negative_warmup() -> None:
    opt = _DummyOpt(lr=0.1)
    with pytest.raises(ValueError, match="warmup_steps"):
        CosineWarmupLR(opt, warmup_steps=-1, total_steps=10)


def test_scheduler_drives_real_optimizer_lr() -> None:
    """Integration: CosineWarmupLR mutates a real Adam optimizer's lr."""
    p = Tensor([1.0])
    opt = Adam([p], lr=0.05)
    sched = CosineWarmupLR(opt, warmup_steps=1, total_steps=4)
    sched.step()
    assert opt.lr == pytest.approx(0.05)  # end of 1-step warmup
    sched.step()
    assert opt.lr < 0.05
