"""Render a small autograd graph to DOT.

Run from project root after `pip install -e .[dev,viz]`:
    python examples/visualize_graph.py

Writes graph.dot. Render to SVG with the system Graphviz:
    dot -Tsvg graph.dot -o graph.svg
"""

from __future__ import annotations

from pathlib import Path

from nanoad import Tensor, relu
from nanoad.viz import draw


def main() -> None:
    a = Tensor(2.0)
    b = Tensor(-3.0)
    c = Tensor(10.0)

    e = a * b
    d = e + c
    f = relu(d) ** 2
    f.backward()

    g = draw(f)
    out = Path("graph.dot")
    out.write_text(g.source, encoding="utf-8")
    print(f"Wrote {out} — render with: dot -Tsvg {out} -o graph.svg")


if __name__ == "__main__":
    main()
