"""Forward shapes and finite-difference grad checks for batched matmul."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.ops.bmm import bmm


def test_bmm_two_d_matches_matmul() -> None:
    a = Tensor(np.random.randn(3, 4))
    b = Tensor(np.random.randn(4, 5))
    out = bmm(a, b)
    assert out.shape == (3, 5)
    np.testing.assert_allclose(out.data, a.data @ b.data)


def test_bmm_three_d_batched() -> None:
    a = Tensor(np.random.randn(2, 3, 4))
    b = Tensor(np.random.randn(2, 4, 5))
    out = bmm(a, b)
    assert out.shape == (2, 3, 5)


def test_bmm_four_d_batched() -> None:
    a = Tensor(np.random.randn(2, 3, 4, 5))
    b = Tensor(np.random.randn(2, 3, 5, 6))
    out = bmm(a, b)
    assert out.shape == (2, 3, 4, 6)


def test_bmm_broadcast_three_d_with_two_d() -> None:
    """When b is 2-d, numpy broadcasts a's leading dims through (matrix-batched-by-mat)."""
    a = Tensor(np.random.randn(2, 3, 4))
    b = Tensor(np.random.randn(4, 5))
    out = bmm(a, b)
    assert out.shape == (2, 3, 5)


def test_bmm_broadcast_leading_dims() -> None:
    a = Tensor(np.random.randn(1, 4, 5))
    b = Tensor(np.random.randn(3, 5, 6))
    out = bmm(a, b)
    assert out.shape == (3, 4, 6)


def test_bmm_grads_two_d(grad_check) -> None:
    grad_check(bmm, np.random.randn(3, 4), np.random.randn(4, 5))


def test_bmm_grads_three_d_batched(grad_check) -> None:
    grad_check(bmm, np.random.randn(2, 3, 4), np.random.randn(2, 4, 5))


def test_bmm_grads_four_d_batched(grad_check) -> None:
    grad_check(bmm, np.random.randn(2, 2, 3, 4), np.random.randn(2, 2, 4, 3))


def test_bmm_grads_with_leading_broadcast(grad_check) -> None:
    """Broadcasting leading dim in a (size 1) requires unbroadcast in the VJP to sum over it."""
    grad_check(bmm, np.random.randn(1, 3, 4), np.random.randn(2, 4, 5))


def test_bmm_grads_three_d_against_two_d(grad_check) -> None:
    grad_check(bmm, np.random.randn(2, 3, 4), np.random.randn(4, 5))


def test_bmm_rejects_one_d() -> None:
    with pytest.raises(ValueError, match="at least 2-d"):
        bmm(Tensor(np.random.randn(3)), Tensor(np.random.randn(3, 4)))
