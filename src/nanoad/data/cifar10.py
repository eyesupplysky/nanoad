"""CIFAR-10 loader: download (cached), unpickle batches, channel-normalize, return numpy arrays."""

from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_ARCHIVE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_ARCHIVE_NAME = "cifar-10-python.tar.gz"
_EXTRACTED_DIR = "cifar-10-batches-py"

_TRAIN_BATCHES = ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5")
_TEST_BATCH = "test_batch"

_CHANNEL_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float64).reshape(1, 3, 1, 1)
_CHANNEL_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float64).reshape(1, 3, 1, 1)


def _cache_dir() -> Path:
    """Where the downloaded archive and extracted batches live across runs."""
    return Path.home() / ".cache" / "nanoad" / "cifar10"


def _download_archive() -> Path:
    """Fetch the tarball into the cache, skipping the network if already present."""
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / _ARCHIVE_NAME
    if target.exists():
        return target
    with urllib.request.urlopen(_ARCHIVE_URL) as resp:
        target.write_bytes(resp.read())
    return target


def _extract_archive(archive: Path) -> Path:
    """Untar archive into the cache once; subsequent calls are no-ops. Return the batch dir."""
    batch_dir = archive.parent / _EXTRACTED_DIR
    if batch_dir.exists():
        return batch_dir
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(archive.parent)
    if not batch_dir.exists():
        raise RuntimeError(f"expected {batch_dir} after extraction")
    return batch_dir


def _load_batch(path: Path) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Unpickle one CIFAR-10 batch file. Returns (data uint8 (n, 3072), labels int64 (n,))."""
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    data = np.asarray(obj["data"], dtype=np.uint8)
    labels = np.asarray(obj["labels"], dtype=np.int64)
    if data.shape[1] != 3072:
        raise RuntimeError(f"bad CIFAR-10 row size: {data.shape[1]}")
    return data, labels


def _stack_batches(batch_dir: Path, names: tuple[str, ...]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Concatenate named batches and reshape to (n, 3, 32, 32) float64 in [0,1]."""
    raw_data: list[NDArray[np.uint8]] = []
    raw_labels: list[NDArray[np.int64]] = []
    for name in names:
        d, l = _load_batch(batch_dir / name)
        raw_data.append(d)
        raw_labels.append(l)
    flat = np.concatenate(raw_data, axis=0)
    images = flat.reshape(-1, 3, 32, 32).astype(np.float64) / 255.0
    labels = np.concatenate(raw_labels, axis=0)
    return images, labels


def load_cifar10(split: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return (images, labels). float64 (n, 3, 32, 32) channel-normalized; labels int64 (n,)."""
    if split == "train":
        names = _TRAIN_BATCHES
    elif split == "test":
        names = (_TEST_BATCH,)
    else:
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    archive = _download_archive()
    batch_dir = _extract_archive(archive)
    images, labels = _stack_batches(batch_dir, names)
    images = (images - _CHANNEL_MEAN) / _CHANNEL_STD
    return images, labels
