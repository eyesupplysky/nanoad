"""Train a tiny CNN on MNIST. Run from project root after `pip install -e .[dev]`."""

from __future__ import annotations

import time

import numpy as np

from nanoad import Tensor
from nanoad.data import DataLoader, load_mnist
from nanoad.nn import (
    BatchNorm2d,
    Conv2d,
    CrossEntropy,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from nanoad.optim import Adam


def main() -> None:
    np.random.seed(0)

    print("Loading MNIST (4-d images for CNN)...")
    x_train, y_train = load_mnist("train", flatten=False)
    x_test, y_test = load_mnist("test", flatten=False)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    model = Sequential(
        Conv2d(1, 8, kernel_size=3, padding=1),  # (N, 8, 28, 28)
        BatchNorm2d(8),
        ReLU(),
        MaxPool2d(2),  # (N, 8, 14, 14)
        Conv2d(8, 16, kernel_size=3, padding=1),  # (N, 16, 14, 14)
        BatchNorm2d(16),
        ReLU(),
        MaxPool2d(2),  # (N, 16, 7, 7)
        Flatten(),  # (N, 784)
        Linear(16 * 7 * 7, 10),
    )
    loss_fn = CrossEntropy()
    opt = Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(x_train, y_train, batch_size=128, shuffle=True)

    n_epochs = 1
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
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

        model.eval()
        # Test in chunks to keep peak memory bounded.
        n_correct = 0
        chunk = 512
        for start in range(0, x_test.shape[0], chunk):
            xb = x_test[start : start + chunk]
            yb = y_test[start : start + chunk]
            preds = model(Tensor(xb)).data.argmax(axis=-1)
            n_correct += int((preds == yb).sum())
        acc = n_correct / x_test.shape[0]
        print(
            f"epoch {epoch + 1}/{n_epochs}  "
            f"loss {epoch_loss / n_batches:.4f}  "
            f"test_acc {acc * 100:.2f}%  "
            f"elapsed {time.time() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()
