"""Graph visualization for autograd tapes via graphviz.

Renders a Tensor's computation graph as a DOT diagram. Each tensor becomes a
record node showing data and grad; each op becomes an oval between its operands
and its output. Optional dependency: install with ``pip install nanoad[viz]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from nanoad.tensor import Tensor

if TYPE_CHECKING:
    import graphviz


def _fmt_value(arr: NDArray[np.float64]) -> str:
    """Compact label for a tensor's data or grad: scalar value or shape tag."""
    if arr.ndim == 0:
        return f"{float(arr):.4g}"
    return f"shape={arr.shape}"


def _trace(root: Tensor) -> tuple[list[Tensor], list[tuple[Tensor, Tensor]]]:
    """Iterative DFS from root; return reachable tensors and parent->child edges."""
    nodes: list[Tensor] = []
    edges: list[tuple[Tensor, Tensor]] = []
    visited: set[int] = set()
    stack: list[Tensor] = [root]
    while stack:
        t = stack.pop()
        if id(t) in visited:
            continue
        visited.add(id(t))
        nodes.append(t)
        for parent in t._prev:
            edges.append((parent, t))
            stack.append(parent)
    return nodes, edges


def draw(root: Tensor, rankdir: str = "LR") -> graphviz.Digraph:
    """Render the autograd graph rooted at root as a graphviz Digraph.

    Each tensor renders as a record node showing data and grad. Each op renders
    as an oval between its operands and its output tensor. The returned Digraph
    can be saved with .render(path) or piped with .pipe(format="svg").
    """
    try:
        import graphviz as _gv
    except ImportError as e:
        raise ImportError(
            "draw() requires the graphviz package. Install with: pip install nanoad[viz]"
        ) from e

    dot = _gv.Digraph(format="svg")
    dot.attr(rankdir=rankdir)

    nodes, edges = _trace(root)

    for t in nodes:
        node_id = str(id(t))
        label = f"{{ data: {_fmt_value(t.data)} | grad: {_fmt_value(t.grad)} }}"
        dot.node(node_id, label=label, shape="record")
        if t._op:
            op_id = node_id + "_op"
            dot.node(op_id, label=t._op, shape="oval")
            dot.edge(op_id, node_id)

    for parent, child in edges:
        dot.edge(str(id(parent)), str(id(child)) + "_op")

    return dot
