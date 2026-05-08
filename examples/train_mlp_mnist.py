"""Train a 2-layer MLP on MNIST. Run from project root after `pip install -e .[dev]`."""

from __future__ import annotations

import time

import numpy as np

from nanoad import Tensor
from nanoad.data import DataLoader, load_mnist
from nanoad.nn import CrossEntropy, Linear, ReLU, Sequential
from nanoad.optim import SGD


def main() -> None:
    np.random.seed(0)

    print("Loading MNIST...")
    x_train, y_train = load_mnist("train")
    x_test, y_test = load_mnist("test")
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 10),
    )
    loss_fn = CrossEntropy()
    opt = SGD(model.parameters(), lr=0.1)

    train_loader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)

    n_epochs = 5
    for epoch in range(n_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            logits = model(Tensor(x_batch))
            loss = loss_fn(logits, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.data)
            n_batches += 1

        preds = model(Tensor(x_test)).data.argmax(axis=-1)
        acc = float((preds == y_test).mean())
        print(
            f"epoch {epoch + 1}/{n_epochs}  "
            f"loss {epoch_loss / n_batches:.4f}  "
            f"test_acc {acc * 100:.2f}%  "
            f"elapsed {time.time() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()
