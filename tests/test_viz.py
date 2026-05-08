"""Visualization tests. Skipped automatically when graphviz is not installed."""

import pytest

graphviz = pytest.importorskip("graphviz")

from nanoad import Tensor  # noqa: E402
from nanoad.viz import draw  # noqa: E402


def test_draw_returns_digraph():
    x = Tensor(2.0)
    g = draw(x)
    assert isinstance(g, graphviz.Digraph)


def test_draw_leaf_has_no_op_node():
    x = Tensor(2.0)
    g = draw(x)
    src = g.source
    assert "shape=record" in src
    assert "shape=oval" not in src


def test_draw_includes_op_label_for_addition():
    a = Tensor(1.0)
    b = Tensor(2.0)
    c = a + b
    g = draw(c)
    src = g.source
    assert "+" in src
    assert "shape=oval" in src


def test_draw_records_all_reachable_tensors_with_shared_parent():
    """When a node feeds two paths, it appears once in the graph."""
    a = Tensor(1.0)
    b = Tensor(2.0)
    c = (a + b) * a
    g = draw(c)
    # Tensors: a, b, (a+b), c → 4 unique record nodes
    n_records = g.source.count("shape=record")
    assert n_records == 4


def test_draw_with_n_d_tensor_shows_shape_label():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    g = draw(x)
    assert "shape=(2, 2)" in g.source


def test_draw_after_backward_includes_grad():
    a = Tensor(3.0)
    b = a * 2.0
    b.backward()
    g = draw(b)
    src = g.source
    assert "grad: 1" in src


def test_draw_rankdir_argument_threads_through():
    a = Tensor(1.0)
    b = Tensor(2.0)
    g = draw(a + b, rankdir="TB")
    assert "rankdir=TB" in g.source
