"""DataLoader tests."""

import numpy as np
import pytest

from nanoad.data import DataLoader


def test_dataloader_yields_correct_batch_shapes():
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.int64)

    loader = DataLoader(x, y, batch_size=3, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4
    assert batches[0][0].shape == (3, 2)
    assert batches[-1][0].shape == (1, 2)


def test_dataloader_no_shuffle_preserves_order():
    x = np.arange(8, dtype=np.float64).reshape(4, 2)
    y = np.array([10, 20, 30, 40], dtype=np.int64)
    loader = DataLoader(x, y, batch_size=2, shuffle=False)
    batches = list(loader)
    assert np.array_equal(batches[0][1], [10, 20])
    assert np.array_equal(batches[1][1], [30, 40])


def test_dataloader_covers_all_samples_with_shuffle():
    x = np.arange(30, dtype=np.float64).reshape(10, 3)
    y = np.arange(10, dtype=np.int64)
    np.random.seed(0)
    loader = DataLoader(x, y, batch_size=4, shuffle=True)

    seen_labels: list[int] = []
    for _, y_batch in loader:
        seen_labels.extend(int(v) for v in y_batch)

    assert sorted(seen_labels) == list(range(10))


def test_dataloader_reshuffles_per_iter():
    x = np.arange(10, dtype=np.float64).reshape(5, 2)
    y = np.arange(5, dtype=np.int64)
    loader = DataLoader(x, y, batch_size=5, shuffle=True)

    np.random.seed(0)
    batch_one = next(iter(loader))[1].copy()
    np.random.seed(1)
    batch_two = next(iter(loader))[1].copy()

    assert not np.array_equal(batch_one, batch_two)


def test_dataloader_size_mismatch_raises():
    x = np.zeros((5, 2))
    y = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="same first dim"):
        DataLoader(x, y, batch_size=2)


def test_dataloader_len_rounds_up():
    x = np.zeros((10, 2))
    y = np.zeros(10, dtype=np.int64)
    loader = DataLoader(x, y, batch_size=3, shuffle=False)
    assert len(loader) == 4
