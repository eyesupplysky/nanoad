"""Train a tiny GPT on the Karpathy tiny-Shakespeare corpus. Run from project root.

Pure NumPy, so the default config is shrunk well below GPT-2 scale to keep wall-clock
within the budget of a teaching example (~25 min on a modern CPU). Increase ``embed_dim``,
``num_layers``, ``block_size``, or ``max_iters`` if you have more time and want better
samples — the architecture scales straight through.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from nanoad import cross_entropy
from nanoad.data import CharTokenizer, load_tinyshakespeare
from nanoad.nn import GPT
from nanoad.optim import AdamW, CosineWarmupLR, clip_grad_norm


def get_batch(
    ids: NDArray[np.int64], batch_size: int, block_size: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Sample ``batch_size`` random windows. Returns input and shifted-by-one targets."""
    n = ids.shape[0]
    starts = np.random.randint(0, n - block_size, size=batch_size)
    x = np.stack([ids[s : s + block_size] for s in starts])
    y = np.stack([ids[s + 1 : s + 1 + block_size] for s in starts])
    return x, y


def eval_loss(
    model: GPT,
    ids: NDArray[np.int64],
    batch_size: int,
    block_size: int,
    n_batches: int,
) -> float:
    """Mean cross-entropy on ``n_batches`` random windows. Caller toggles eval mode."""
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(ids, batch_size, block_size)
        logits = model(x)
        b, t, v = logits.shape
        losses.append(float(cross_entropy(logits.reshape(b * t, v), y.reshape(b * t)).data))
    return float(np.mean(losses))


def main() -> None:
    # Config — tune these if you have more wall-clock budget.
    block_size = 32
    embed_dim = 64
    num_heads = 4
    num_layers = 4
    batch_size = 32
    max_iters = 2000
    eval_every = 200
    eval_batches = 20
    lr = 3e-4
    min_lr = 3e-5
    weight_decay = 0.01
    warmup_steps = 100
    grad_clip = 1.0
    seed = 0xC0FFEE

    np.random.seed(seed)

    print("Loading tiny-Shakespeare...")
    text = load_tinyshakespeare()
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    n_train = int(0.9 * ids.shape[0])
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    print(
        f"  corpus {ids.shape[0]:,} chars  vocab {tok.vocab_size}  "
        f"train {train_ids.shape[0]:,}  val {val_ids.shape[0]:,}"
    )

    model = GPT(
        vocab_size=tok.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
    )
    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    print(
        f"Model: {num_layers}L x {num_heads}H x {embed_dim}D, block_size={block_size}, "
        f"{n_params:,} parameters"
    )

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineWarmupLR(
        opt, warmup_steps=warmup_steps, total_steps=max_iters, min_lr=min_lr
    )

    print(f"Training: {max_iters} iters, batch={batch_size}, lr={lr} -> {min_lr} cosine")
    t_start = time.time()
    for step in range(1, max_iters + 1):
        x_batch, y_batch = get_batch(train_ids, batch_size, block_size)
        logits = model(x_batch)
        b, t, v = logits.shape
        loss = cross_entropy(logits.reshape(b * t, v), y_batch.reshape(b * t))
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=grad_clip)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == 1:
            model.eval()
            val = eval_loss(model, val_ids, batch_size, block_size, n_batches=eval_batches)
            model.train()
            elapsed = time.time() - t_start
            print(
                f"step {step:>5}/{max_iters}  "
                f"train {float(loss.data):.4f}  val {val:.4f}  "
                f"lr {opt.lr:.2e}  elapsed {elapsed:.0f}s"
            )

    print(f"\nTraining done in {time.time() - t_start:.0f}s.")

    # Sample
    print("\n--- Sample (temperature=0.8, top_k=20) ---")
    seed_text = "ROMEO:"
    seed_ids = tok.encode(seed_text)[None, :]
    model.eval()
    completion = model.generate(seed_ids, max_new_tokens=400, temperature=0.8, top_k=20)
    print(tok.decode(completion[0]))


if __name__ == "__main__":
    main()
