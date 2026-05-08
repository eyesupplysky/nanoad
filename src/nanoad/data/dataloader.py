"""Simple iterating DataLoader."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


class DataLoader:
    """Yields (x_batch, y_batch) numpy arrays. Reshuffles per __iter__ when shuffle=True."""

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same first dim; got {x.shape[0]} and {y.shape[0]}")
        self.x: NDArray[np.float64] = x
        self.y: NDArray[np.int64] = y
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self._n: int = x.shape[0]

    def __iter__(self) -> Iterator[tuple[NDArray[np.float64], NDArray[np.int64]]]:
        idx = np.arange(self._n)
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, self._n, self.batch_size):
            chunk = idx[start : start + self.batch_size]
            yield self.x[chunk], self.y[chunk]

    def __len__(self) -> int:
        return (self._n + self.batch_size - 1) // self.batch_size
