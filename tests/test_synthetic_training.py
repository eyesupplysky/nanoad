"""End-to-end smoke test: a 2-layer MLP learns a 2-class synthetic problem.

Exercises the full M4 stack — Tensor, Linear, ReLU, Sequential, CrossEntropy, SGD.
Stays small enough to run in CI in well under a second.
"""

import numpy as np

from nanoad import Tensor
from nanoad.nn import CrossEntropy, Linear, ReLU, Sequential
from nanoad.optim import SGD


def test_mlp_learns_xor_quadrant_pattern():
    """Classify points by XOR of sign(x), sign(y) — non-linearly separable."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal((n, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int64)

    np.random.seed(0)
    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 2),
    )
    loss_fn = CrossEntropy()
    opt = SGD(model.parameters(), lr=0.1)

    initial_loss = float(loss_fn(model(Tensor(x)), y).data)

    for _ in range(1000):
        logits = model(Tensor(x))
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = float(loss_fn(model(Tensor(x)), y).data)
    assert final_loss < 0.1 * initial_loss

    preds = model(Tensor(x)).data.argmax(axis=-1)
    accuracy = float((preds == y).mean())
    assert accuracy > 0.95
