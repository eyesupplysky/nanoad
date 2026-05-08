"""Direct unit tests for the unbroadcast helper."""

import numpy as np

from nanoad.ops._broadcast import unbroadcast


def test_unbroadcast_no_op_when_shapes_match():
    grad = np.ones((2, 3))
    out = unbroadcast(grad, (2, 3))
    assert out.shape == (2, 3)
    assert np.all(out == 1.0)


def test_unbroadcast_to_scalar_sums_everything():
    grad = np.ones((2, 3))
    out = unbroadcast(grad, ())
    assert out.shape == ()
    assert out == 6.0


def test_unbroadcast_drops_leading_dims():
    grad = np.ones((2, 3))
    out = unbroadcast(grad, (3,))
    assert out.shape == (3,)
    assert np.all(out == 2.0)


def test_unbroadcast_collapses_size_one_dim_with_keepdims():
    grad = np.ones((2, 3))
    out = unbroadcast(grad, (1, 3))
    assert out.shape == (1, 3)
    assert np.all(out == 2.0)


def test_unbroadcast_collapses_inner_size_one_dim():
    grad = np.ones((2, 3))
    out = unbroadcast(grad, (2, 1))
    assert out.shape == (2, 1)
    assert np.all(out == 3.0)


def test_unbroadcast_combined_leading_and_size_one():
    grad = np.ones((4, 5, 3))
    out = unbroadcast(grad, (1, 3))
    assert out.shape == (1, 3)
    assert np.all(out == 20.0)


def test_unbroadcast_three_d_target_with_size_one_middle():
    grad = np.ones((2, 5, 3))
    out = unbroadcast(grad, (2, 1, 3))
    assert out.shape == (2, 1, 3)
    assert np.all(out == 5.0)
