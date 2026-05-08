"""GPT: forward shape, parameter discovery, causal smoke, and end-to-end gradient flow."""

from __future__ import annotations

import numpy as np
import pytest

from nanoad import Tensor, cross_entropy
from nanoad.nn import GELU, GPT


def _tiny_gpt() -> GPT:
    """A toy config used across most tests."""
    return GPT(vocab_size=16, embed_dim=8, num_heads=2, num_layers=2, block_size=10)


def test_gpt_forward_shape() -> None:
    model = _tiny_gpt()
    idx = np.random.randint(0, 16, size=(3, 5))
    logits = model(idx)
    assert logits.shape == (3, 5, 16)


def test_gpt_rejects_indivisible_num_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GPT(vocab_size=16, embed_dim=8, num_heads=3, num_layers=1, block_size=4)


def test_gpt_rejects_non_2d_idx() -> None:
    model = _tiny_gpt()
    with pytest.raises(ValueError, match="2-d"):
        model(np.array([1, 2, 3]))


def test_gpt_rejects_sequence_longer_than_block_size() -> None:
    model = _tiny_gpt()  # block_size=10
    idx = np.zeros((1, 11), dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)


def test_gpt_parameters_cover_every_submodule() -> None:
    """Every layer in the architecture must contribute to parameters()."""
    model = _tiny_gpt()
    n_params = sum(1 for _ in model.parameters())
    # tok_emb(1) + pos_emb(1) + ln_f(2: gamma, beta) + lm_head(2: W, b)
    # per block: ln1(2) + attn(4) + ln2(2) + mlp(2 + 2) = 12
    expected = 1 + 1 + 2 + 2 + 2 * 12
    assert n_params == expected


def test_gpt_backward_fills_grad_on_every_parameter() -> None:
    """End-to-end smoke: a CE loss on (B, T, V) logits flows grads to every param."""
    model = _tiny_gpt()
    idx = np.random.randint(0, 16, size=(2, 4))
    targets = np.random.randint(0, 16, size=(2 * 4,))

    logits = model(idx)
    b, t, v = logits.shape
    loss = cross_entropy(logits.reshape(b * t, v), targets)
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None, "every parameter should receive a gradient"
        assert p.grad.shape == p.shape


def test_gpt_causal_mask_propagates_through_full_stack() -> None:
    """Output position t depends only on input positions [:t+1] (full-model causality)."""
    np.random.seed(0)
    model = _tiny_gpt()  # block_size=10
    idx_a = np.random.randint(0, 16, size=(1, 6))
    out_a = model(idx_a)

    idx_b = idx_a.copy()
    idx_b[0, 3:] = (idx_b[0, 3:] + 7) % 16  # perturb positions 3..5
    out_b = model(idx_b)

    np.testing.assert_allclose(out_a.data[0, 0], out_b.data[0, 0], atol=1e-10)
    np.testing.assert_allclose(out_a.data[0, 2], out_b.data[0, 2], atol=1e-10)
    assert not np.allclose(out_a.data[0, 3], out_b.data[0, 3], atol=1e-6)


def test_gelu_matches_reference_values() -> None:
    """GELU tanh-approximation should match a reference numpy implementation."""
    x = np.linspace(-3.0, 3.0, 7)
    coeff = float(np.sqrt(2.0 / np.pi))
    expected = 0.5 * x * (1.0 + np.tanh(coeff * (x + 0.044715 * x**3)))

    out = GELU()(Tensor(x))
    np.testing.assert_allclose(out.data, expected, atol=1e-12)


def test_gelu_grad_check(grad_check) -> None:
    """GELU is composed from public ops — gradients must match finite differences."""
    grad_check(lambda x: GELU()(x), np.random.randn(4, 5))


def test_generate_returns_correct_shape_and_dtype() -> None:
    model = _tiny_gpt()  # block_size=10, vocab=16
    prompt = np.array([[0, 1, 2], [3, 4, 5]])
    out = model.generate(prompt, max_new_tokens=4)
    assert out.shape == (2, 7)
    assert out.dtype == np.int64


def test_generate_token_ids_are_in_vocab_range() -> None:
    model = _tiny_gpt()  # vocab=16
    prompt = np.array([[0]])
    out = model.generate(prompt, max_new_tokens=20)
    assert ((out >= 0) & (out < 16)).all()


def test_generate_preserves_prompt_prefix() -> None:
    model = _tiny_gpt()
    prompt = np.array([[5, 7, 9]])
    out = model.generate(prompt, max_new_tokens=5)
    np.testing.assert_array_equal(out[:, :3], prompt)


def test_generate_is_deterministic_for_a_fixed_seed() -> None:
    model = _tiny_gpt()
    prompt = np.array([[1, 2, 3]])

    np.random.seed(42)
    a = model.generate(prompt, max_new_tokens=8)
    np.random.seed(42)
    b = model.generate(prompt, max_new_tokens=8)
    np.testing.assert_array_equal(a, b)


def test_generate_zero_new_tokens_returns_prompt_unchanged() -> None:
    model = _tiny_gpt()
    prompt = np.array([[1, 2, 3]])
    out = model.generate(prompt, max_new_tokens=0)
    np.testing.assert_array_equal(out, prompt)


def test_generate_handles_prompt_longer_than_block_size() -> None:
    """Prompts > block_size should generate by conditioning on the trailing window only."""
    model = _tiny_gpt()  # block_size=10
    prompt = np.random.randint(0, 16, size=(1, 15))
    out = model.generate(prompt, max_new_tokens=3)
    assert out.shape == (1, 18)


def test_generate_top_k_restricts_support_set() -> None:
    """With top_k=1 generation is greedy (argmax of logits) — no sampling randomness."""
    model = _tiny_gpt()
    prompt = np.array([[1, 2, 3]])

    np.random.seed(0)
    a = model.generate(prompt, max_new_tokens=4, top_k=1)
    np.random.seed(999)
    b = model.generate(prompt, max_new_tokens=4, top_k=1)
    np.testing.assert_array_equal(a, b)  # seed-independent under top_k=1


def test_generate_rejects_non_positive_temperature() -> None:
    model = _tiny_gpt()
    with pytest.raises(ValueError, match="temperature"):
        model.generate(np.array([[0]]), max_new_tokens=1, temperature=0.0)


def test_generate_rejects_negative_max_new_tokens() -> None:
    model = _tiny_gpt()
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(np.array([[0]]), max_new_tokens=-1)


def test_generate_rejects_non_positive_top_k() -> None:
    model = _tiny_gpt()
    with pytest.raises(ValueError, match="top_k"):
        model.generate(np.array([[0]]), max_new_tokens=1, top_k=0)


def test_generate_rejects_non_2d_idx() -> None:
    model = _tiny_gpt()
    with pytest.raises(ValueError, match="2-d"):
        model.generate(np.array([0, 1, 2]), max_new_tokens=1)
