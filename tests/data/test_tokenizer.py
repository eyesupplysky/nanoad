"""CharTokenizer tests."""

import numpy as np
import pytest

from nanoad.data import CharTokenizer


def test_vocab_is_sorted_unique_chars():
    tok = CharTokenizer("banana")
    assert tok.chars == ("a", "b", "n")
    assert tok.vocab_size == 3


def test_encode_returns_int64_array():
    tok = CharTokenizer("ab")
    ids = tok.encode("aabb")
    assert ids.dtype == np.int64
    assert ids.shape == (4,)
    assert ids.tolist() == [0, 0, 1, 1]


def test_encode_decode_round_trip():
    text = "to be or not to be, that is the question"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text


def test_decode_accepts_python_list():
    tok = CharTokenizer("abc")
    assert tok.decode([0, 1, 2, 1, 0]) == "abcba"


def test_encode_raises_on_unknown_char():
    tok = CharTokenizer("ab")
    with pytest.raises(KeyError, match="'z'"):
        tok.encode("zoo")


def test_decode_raises_on_out_of_range_id():
    tok = CharTokenizer("ab")
    with pytest.raises(IndexError, match="out of range"):
        tok.decode([0, 1, 99])


def test_vocab_is_deterministic_across_instances():
    text = "the quick brown fox jumps over the lazy dog"
    a = CharTokenizer(text)
    b = CharTokenizer(text)
    assert a.chars == b.chars
    assert np.array_equal(a.encode(text), b.encode(text))


def test_empty_text_raises():
    with pytest.raises(ValueError, match="non-empty text"):
        CharTokenizer("")


def test_decode_empty_returns_empty_string():
    tok = CharTokenizer("ab")
    assert tok.decode([]) == ""
    assert tok.decode(np.array([], dtype=np.int64)) == ""
