"""Char-level tokenizer with a deterministic sorted-unique-char vocab."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


class CharTokenizer:
    """Char-level tokenizer: vocab is sorted unique chars from the training corpus."""

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("CharTokenizer requires non-empty text to build a vocab")
        chars = sorted(set(text))
        self._chars: tuple[str, ...] = tuple(chars)
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self._itos: tuple[str, ...] = self._chars

    @property
    def vocab_size(self) -> int:
        """Number of unique chars in the vocab."""
        return len(self._chars)

    @property
    def chars(self) -> tuple[str, ...]:
        """The vocab itself, in id order. ``chars[i]`` is the char with token id ``i``."""
        return self._chars

    def encode(self, s: str) -> NDArray[np.int64]:
        """Map ``s`` to a 1-d int64 array of token ids; raise KeyError on unknown char."""
        try:
            ids = [self._stoi[ch] for ch in s]
        except KeyError as e:
            raise KeyError(f"char {e.args[0]!r} not in vocab") from None
        return np.array(ids, dtype=np.int64)

    def decode(self, ids: Iterable[int] | NDArray[np.int64]) -> str:
        """Map a 1-d sequence of token ids back to a string."""
        arr = np.asarray(ids, dtype=np.int64).ravel()
        if arr.size and (arr.min() < 0 or arr.max() >= len(self._chars)):
            raise IndexError(
                f"token id out of range [0, {len(self._chars)}); got min={arr.min()} max={arr.max()}"
            )
        return "".join(self._itos[int(i)] for i in arr)
