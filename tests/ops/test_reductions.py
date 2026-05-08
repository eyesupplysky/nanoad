"""Forward and backward tests for sum and mean reductions."""

import numpy as np

from nanoad import Tensor


def test_sum_all(grad_check):
    grad_check(lambda x: x.sum(), [[1.0, 2.0], [3.0, 4.0]])


def test_sum_axis_zero(grad_check):
    grad_check(lambda x: x.sum(axis=0), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_sum_axis_one(grad_check):
    grad_check(lambda x: x.sum(axis=1), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_sum_keepdims(grad_check):
    grad_check(lambda x: x.sum(axis=1, keepdims=True), [[1.0, 2.0], [3.0, 4.0]])


def test_sum_negative_axis(grad_check):
    grad_check(lambda x: x.sum(axis=-1), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_sum_multi_axis(grad_check):
    grad_check(lambda x: x.sum(axis=(0, 2)), np.arange(24, dtype=np.float64).reshape(2, 3, 4))


def test_sum_forward_value():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    s = x.sum()
    assert s.data == 10.0
    assert s.shape == ()


def test_sum_forward_axis_value():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    s = x.sum(axis=0)
    assert s.shape == (3,)
    assert np.allclose(s.data, [5.0, 7.0, 9.0])


def test_mean_all(grad_check):
    grad_check(lambda x: x.mean(), [[1.0, 2.0], [3.0, 4.0]])


def test_mean_axis_zero(grad_check):
    grad_check(lambda x: x.mean(axis=0), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_mean_keepdims(grad_check):
    grad_check(lambda x: x.mean(axis=1, keepdims=True), [[1.0, 2.0], [3.0, 4.0]])


def test_mean_forward_value():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    m = x.mean()
    assert m.data == 2.5
