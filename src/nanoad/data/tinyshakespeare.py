"""Tiny-Shakespeare loader: download (cached + sha256-verified), return as str."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
_FILE_NAME = "input.txt"
_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def _cache_dir() -> Path:
    """Where the downloaded corpus lives across runs."""
    return Path.home() / ".cache" / "nanoad" / "tinyshakespeare"


def _download() -> Path:
    """Fetch the corpus into the cache, verifying sha256; re-fetch once if a cached copy is corrupt."""
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / _FILE_NAME
    if target.exists() and _sha256(target) == _SHA256:
        return target
    with urllib.request.urlopen(_URL) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != _SHA256:
        raise RuntimeError(
            f"tinyshakespeare sha256 mismatch: expected {_SHA256}, got {digest}"
        )
    target.write_bytes(data)
    return target


def _sha256(path: Path) -> str:
    """Hex sha256 of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tinyshakespeare() -> str:
    """Return the tiny-Shakespeare corpus as a single str (~1.1MB, 65 unique chars)."""
    path = _download()
    return path.read_text(encoding="utf-8")
