"""Train a small CNN on CIFAR-10. Run from project root after `pip install -e .[dev]`.

Architecture: 2-block VGG-tiny (3->16->16, pool, ->32->32, pool, FC 2048->64->10).
With random-crop and horizontal-flip augmentation, ~10 epochs reach 65-70% test accuracy.
A single epoch on numpy/CPU is slow (an order of magnitude beyond the MNIST example);
expect tens of minutes per epoch. Reduce to N_EPOCHS=1 or shrink the network for a fast smoke test.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from nanoad import Tensor
from nanoad.data import DataLoader, load_cifar10
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


def random_crop_pad4(
    x: NDArray[np.float64], y: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Pad each image with 4 zero pixels on each side, then crop a random 32x32 window per sample."""
    n = x.shape[0]
    padded = np.pad(x, ((0, 0), (0, 0), (4, 4), (4, 4)))
    h_off = np.random.randint(0, 9, size=n)
    w_off = np.random.randint(0, 9, size=n)
    out = np.empty_like(x)
    for i in range(n):
        out[i] = padded[i, :, h_off[i] : h_off[i] + 32, w_off[i] : w_off[i] + 32]
    return out, y


def random_hflip(
    x: NDArray[np.float64], y: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Flip each image along the width axis independently with probability 0.5."""
    flip_mask = np.random.rand(x.shape[0]) < 0.5
    out = x.copy()
    out[flip_mask] = out[flip_mask, :, :, ::-1]
    return out, y


def augment(
    x: NDArray[np.float64], y: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compose random crop + horizontal flip."""
    x, y = random_crop_pad4(x, y)
    x, y = random_hflip(x, y)
    return x, y


def main() -> None:
    np.random.seed(0)

    print("Loading CIFAR-10 (channel-normalized, NCHW)...")
    x_train, y_train = load_cifar10("train")
    x_test, y_test = load_cifar10("test")
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    model = Sequential(
        Conv2d(3, 16, kernel_size=3, padding=1),  # (N, 16, 32, 32)
        BatchNorm2d(16),
        ReLU(),
        Conv2d(16, 16, kernel_size=3, padding=1),  # (N, 16, 32, 32)
        BatchNorm2d(16),
        ReLU(),
        MaxPool2d(2),  # (N, 16, 16, 16)
        Conv2d(16, 32, kernel_size=3, padding=1),  # (N, 32, 16, 16)
        BatchNorm2d(32),
        ReLU(),
        Conv2d(32, 32, kernel_size=3, padding=1),  # (N, 32, 16, 16)
        BatchNorm2d(32),
        ReLU(),
        MaxPool2d(2),  # (N, 32, 8, 8)
        Flatten(),  # (N, 2048)
        Linear(32 * 8 * 8, 64),
        ReLU(),
        Linear(64, 10),
    )
    loss_fn = CrossEntropy()
    opt = Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(x_train, y_train, batch_size=64, shuffle=True, transform=augment)

    n_epochs = 5
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
        chunk = 256
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
