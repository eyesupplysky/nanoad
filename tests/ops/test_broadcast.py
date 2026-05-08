"""Direct unit tests for the unbroadcast Tensor op (the elementwise-backward chokepoint)."""

import numpy as np

from nanoad import Tensor
from nanoad.ops._broadcast import broadcast_to, unbroadcast


def test_unbroadcast_no_op_when_shapes_match():
    grad = Tensor(np.ones((2, 3)))
    out = unbroadcast(grad, (2, 3))
    assert out.data.shape == (2, 3)
    assert np.all(out.data == 1.0)


def test_unbroadcast_to_scalar_sums_everything():
    grad = Tensor(np.ones((2, 3)))
    out = unbroadcast(grad, ())
    assert out.data.shape == ()
    assert out.data == 6.0


def test_unbroadcast_drops_leading_dims():
    grad = Tensor(np.ones((2, 3)))
    out = unbroadcast(grad, (3,))
    assert out.data.shape == (3,)
    assert np.all(out.data == 2.0)


def test_unbroadcast_collapses_size_one_dim_with_keepdims():
    grad = Tensor(np.ones((2, 3)))
    out = unbroadcast(grad, (1, 3))
    assert out.data.shape == (1, 3)
    assert np.all(out.data == 2.0)


def test_unbroadcast_collapses_inner_size_one_dim():
    grad = Tensor(np.ones((2, 3)))
    out = unbroadcast(grad, (2, 1))
    assert out.data.shape == (2, 1)
    assert np.all(out.data == 3.0)


def test_unbroadcast_combined_leading_and_size_one():
    grad = Tensor(np.ones((4, 5, 3)))
    out = unbroadcast(grad, (1, 3))
    assert out.data.shape == (1, 3)
    assert np.all(out.data == 20.0)


def test_unbroadcast_three_d_target_with_size_one_middle():
    grad = Tensor(np.ones((2, 5, 3)))
    out = unbroadcast(grad, (2, 1, 3))
    assert out.data.shape == (2, 1, 3)
    assert np.all(out.data == 5.0)


def test_broadcast_to_round_trips_through_unbroadcast():
    """broadcast_to and unbroadcast are mutual VJPs — composition preserves the smaller shape."""
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))  # (1, 3)
    expanded = broadcast_to(x, (4, 3))
    assert expanded.data.shape == (4, 3)
    assert np.all(expanded.data[0] == [1.0, 2.0, 3.0])

    back = unbroadcast(expanded, (1, 3))
    assert back.data.shape == (1, 3)
    np.testing.assert_allclose(back.data, [[4.0, 8.0, 12.0]])
