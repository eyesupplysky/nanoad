"""MNIST loader: download (cached), decode IDX-format, return numpy arrays."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

_FILES: dict[str, tuple[str, str]] = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _cache_dir() -> Path:
    """Where downloaded MNIST archives live across runs."""
    return Path.home() / ".cache" / "nanoad" / "mnist"


def _download(name: str) -> Path:
    """Fetch name from _BASE_URL into the cache, skipping the network if already present."""
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if target.exists():
        return target
    url = _BASE_URL + name
    with urllib.request.urlopen(url) as resp:
        target.write_bytes(resp.read())
    return target


def _read_idx_images(path: Path) -> NDArray[np.float64]:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise RuntimeError(f"bad IDX image magic: {magic}")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def _read_idx_labels(path: Path) -> NDArray[np.int64]:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise RuntimeError(f"bad IDX label magic: {magic}")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(split: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return (images, labels). images shape (n, 784), float64 in [0,1]. labels shape (n,) int64."""
    if split not in _FILES:
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    img_name, lbl_name = _FILES[split]
    img_path = _download(img_name)
    lbl_path = _download(lbl_name)
    return _read_idx_images(img_path), _read_idx_labels(lbl_path)
