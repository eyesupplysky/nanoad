"""Forward shapes and finite-difference grad check for embedding gather/scatter."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor
from nanoad.ops.embedding import _embedding_scatter, embedding


def test_embedding_forward_shape_one_d_indices() -> None:
    weight = Tensor(np.random.randn(10, 4))
    out = embedding(weight, np.array([3, 1, 7]))
    assert out.shape == (3, 4)


def test_embedding_forward_shape_two_d_indices() -> None:
    weight = Tensor(np.random.randn(10, 4))
    out = embedding(weight, np.array([[3, 1], [7, 0], [2, 9]]))
    assert out.shape == (3, 2, 4)


def test_embedding_forward_value_pinned() -> None:
    weight = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    out = embedding(weight, np.array([2, 0, 1]))
    np.testing.assert_array_equal(out.data, np.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]]))


def test_embedding_grad_check(grad_check) -> None:
    """Gradient w.r.t. the weight matches central differences (with index repetition)."""
    indices = np.array([2, 0, 5, 2])  # row 2 repeated — should accumulate

    def fn(weight: Tensor) -> Tensor:
        return embedding(weight, indices)

    grad_check(fn, np.random.randn(7, 3))


def test_embedding_repeated_indices_accumulate() -> None:
    """Backward into a row picked twice should equal twice the row's contribution."""
    weight = Tensor(np.zeros((4, 2)))
    indices = np.array([1, 1, 3])
    out = embedding(weight, indices)
    out.sum().backward()
    # Each of three lookups contributes ones(2) to its row's grad.
    expected = np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(weight.grad.data, expected)


def test_embedding_rejects_non_2d_weight() -> None:
    weight = Tensor(np.zeros((4, 2, 3)))
    with pytest.raises(ValueError, match="2-d"):
        embedding(weight, np.array([0, 1]))


def test_embedding_scatter_round_trip() -> None:
    """gather then scatter on the same indices is the identity for rows that appear."""
    weight = Tensor(np.random.randn(6, 3))
    indices = np.array([0, 1, 2, 3, 4, 5])  # cover every row exactly once
    gathered = embedding(weight, indices)
    scattered = _embedding_scatter(gathered, indices, 6)
    np.testing.assert_allclose(scattered.data, weight.data)
