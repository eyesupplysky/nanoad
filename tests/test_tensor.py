"""M1 exit gate: hand-derived gradient on a small graph."""

from nanoad import Tensor


def test_three_node_graph():
    """f(a, b, c) = (a + b) * c. df/da = c, df/db = c, df/dc = a + b."""
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = Tensor(4.0)

    out = (a + b) * c
    out.backward()

    assert out.data == 20.0
    assert a.grad == 4.0
    assert b.grad == 4.0
    assert c.grad == 5.0


def test_grad_accumulates_across_paths():
    """When a node feeds two paths, gradients sum: d(x*x)/dx = 2x."""
    x = Tensor(3.0)
    y = x * x
    y.backward()

    assert y.data == 9.0
    assert x.grad == 6.0


def test_zero_grad_resets_local():
    x = Tensor(1.0)
    x.grad = 5.0
    x.zero_grad()
    assert x.grad == 0.0


def test_backward_on_leaf_does_not_explode():
    x = Tensor(7.0)
    x.backward()
    assert x.grad == 1.0


def test_repr_is_readable():
    x = Tensor(1.5)
    r = repr(x)
    assert "Tensor" in r
    assert "1.5" in r
    assert "shape=()" in r


def test_tensor_accepts_array_likes():
    """Tensor constructor coerces lists, tuples, scalars, and numpy arrays into float64."""
    import numpy as np

    a = Tensor([1, 2, 3])
    assert a.shape == (3,)
    assert a.data.dtype == np.float64

    b = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert b.shape == (2, 2)

    c = Tensor(np.arange(6).reshape(2, 3))
    assert c.shape == (2, 3)
    assert c.data.dtype == np.float64


def test_backward_rejects_non_scalar_output():
    """backward() requires the output to be 0-d; reduce first."""
    import pytest

    x = Tensor([1.0, 2.0, 3.0])
    y = x * 2.0
    with pytest.raises(RuntimeError, match="scalar"):
        y.backward()


def test_backward_after_sum_works_on_vector_input():
    x = Tensor([1.0, 2.0, 3.0])
    y = (x * 2.0).sum()
    y.backward()
    import numpy as np

    assert np.allclose(x.grad, [2.0, 2.0, 2.0])
