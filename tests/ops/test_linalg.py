"""Forward and backward tests for matmul, transpose, reshape."""

import numpy as np
import pytest

from nanoad import Tensor


def test_matmul_basic_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    assert c.shape == (2, 2)
    assert np.allclose(c.data, [[19.0, 22.0], [43.0, 50.0]])


def test_matmul_grads(grad_check):
    grad_check(
        lambda a, b: a @ b,
        np.arange(6, dtype=np.float64).reshape(2, 3) * 0.1,
        np.arange(12, dtype=np.float64).reshape(3, 4) * 0.1,
    )


def test_matmul_non_square(grad_check):
    grad_check(
        lambda a, b: a @ b,
        [[1.0, 2.0, 3.0]],
        [[4.0], [5.0], [6.0]],
    )


def test_matmul_rejects_one_d():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="matmul"):
        _ = a @ b


def test_matmul_rejects_three_d():
    a = Tensor(np.zeros((2, 3, 4)))
    b = Tensor(np.zeros((4, 5)))
    with pytest.raises(ValueError, match="matmul"):
        _ = a @ b


def test_transpose_property_t():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x.T.shape == (3, 2)
    assert np.allclose(x.T.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


def test_transpose_grads(grad_check):
    grad_check(lambda x: x.T, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_transpose_with_axes_grads(grad_check):
    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    grad_check(lambda x: x.transpose(2, 0, 1), arr)


def test_transpose_three_d_forward():
    x = Tensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
    y = x.transpose(2, 0, 1)
    assert y.shape == (4, 2, 3)


def test_reshape_grads(grad_check):
    grad_check(lambda x: x.reshape(3, 2), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_reshape_to_flat(grad_check):
    grad_check(lambda x: x.reshape(6), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_reshape_with_inferred_dim(grad_check):
    grad_check(lambda x: x.reshape(2, -1), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_reshape_forward_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = x.reshape(2, 3)
    assert y.shape == (2, 3)
    assert np.allclose(y.data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
