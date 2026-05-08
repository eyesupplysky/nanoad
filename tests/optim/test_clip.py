"""clip_grad_norm tests: scale-down trigger, no-op when under threshold, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.optim.clip import clip_grad_norm


def test_clip_no_op_when_norm_under_threshold() -> None:
    p = Tensor(np.zeros(3))
    p.grad = Tensor(np.array([0.1, 0.2, 0.3]))  # norm ~ 0.374
    pre = clip_grad_norm([p], max_norm=1.0)
    np.testing.assert_allclose(p.grad.data, np.array([0.1, 0.2, 0.3]))
    assert pre == pytest.approx(np.linalg.norm([0.1, 0.2, 0.3]))


def test_clip_scales_down_when_norm_exceeds_threshold() -> None:
    p = Tensor(np.zeros(3))
    p.grad = Tensor(np.array([3.0, 4.0, 0.0]))  # norm = 5.0
    pre = clip_grad_norm([p], max_norm=1.0)
    assert pre == pytest.approx(5.0)
    post_norm = float(np.linalg.norm(p.grad.data))
    assert post_norm == pytest.approx(1.0, rel=1e-5)


def test_clip_uses_global_norm_across_params() -> None:
    """Two params sharing the global norm must scale by the same factor."""
    p1 = Tensor(np.zeros(2))
    p1.grad = Tensor(np.array([3.0, 0.0]))  # contributes 9 to norm**2
    p2 = Tensor(np.zeros(1))
    p2.grad = Tensor(np.array([4.0]))  # contributes 16 — together norm = 5
    pre = clip_grad_norm([p1, p2], max_norm=1.0)
    assert pre == pytest.approx(5.0)
    # Both grads should be scaled by ~1/5.
    np.testing.assert_allclose(p1.grad.data, np.array([0.6, 0.0]), rtol=1e-5)
    np.testing.assert_allclose(p2.grad.data, np.array([0.8]), rtol=1e-5)


def test_clip_skips_none_grad_params() -> None:
    """Params with grad=None should not be touched and not affect the norm."""
    p_with = Tensor(np.zeros(2))
    p_with.grad = Tensor(np.array([3.0, 4.0]))
    p_without = Tensor(np.zeros(2))
    pre = clip_grad_norm([p_with, p_without], max_norm=10.0)
    assert pre == pytest.approx(5.0)
    assert p_without.grad is None


def test_clip_returns_zero_when_no_grads_present() -> None:
    p = Tensor(np.zeros(3))  # grad still None
    assert clip_grad_norm([p], max_norm=1.0) == 0.0


def test_clip_rejects_non_positive_max_norm() -> None:
    p = Tensor(np.zeros(2))
    p.grad = Tensor(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="max_norm"):
        clip_grad_norm([p], max_norm=0.0)


def test_clip_preserves_grad_tensor_identity() -> None:
    """Clipping should mutate .data in-place; the .grad Tensor object stays the same."""
    p = Tensor(np.zeros(2))
    p.grad = Tensor(np.array([3.0, 4.0]))
    grad_before = p.grad
    clip_grad_norm([p], max_norm=1.0)
    assert p.grad is grad_before
