# nanoad

A from-scratch reverse-mode automatic differentiation engine and small neural-net framework in pure NumPy. No PyTorch, no JAX, no Autograd — just the tape, the ops, the optimizers, and the training loops, small enough to read in one sitting.

What sets it apart from `micrograd` and early `tinygrad`:

- **Full n-d tensors** with broadcasting, not scalar-only.
- **Higher-order autograd.** Gradients are themselves differentiable. `grad`, `vjp`, `jvp`, `hvp`, and `hessian` live in `nanoad.functional`, JAX-shaped, built on the dynamic tape rather than a tracing JIT.
- **A complete transformer.** `GPT` with causal multi-head attention, `LayerNorm`, `GELU`, `AdamW`, and cosine warmup, training end-to-end on tiny-Shakespeare in pure NumPy.

The `nano` signals "small reference implementation" in the [nanoGPT](https://github.com/karpathy/nanoGPT) / [micrograd](https://github.com/karpathy/micrograd) lineage. The `ad` is automatic differentiation.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.12+ and NumPy 1.26+. Optional `graphviz` for tape visualisation.

## A 30-second tour

```python
from nanoad import Tensor

x = Tensor([2.0, -1.0, 3.0])
y = Tensor([0.5,  4.0, 1.0])

z = (x * y + 1).sum()   # forward: build a tape as ops execute
z.backward()            # reverse-mode pass: fill .grad on every leaf

print(x.grad)  # [0.5  4.   1. ]   == d(z)/dx == y
print(y.grad)  # [ 2. -1.  3. ]    == d(z)/dy == x
```

Every op (`+`, `*`, `@`, `relu`, `softmax`, `conv2d`, ...) registers itself on the tape during forward. `backward()` walks the tape in reverse topological order and accumulates gradients into each leaf's `.grad`.

## Higher-order autograd

Most reference autograd libraries stop at first-order gradients. nanoad doesn't. Forward and backward both build differentiable tape, so a gradient is itself a Tensor whose own gradient can be taken — the second derivative falls out for free.

```python
from nanoad import Tensor
from nanoad.functional import grad, hessian

f = lambda x: (x ** 2).sum()             # f(x) = xᵀx

print(grad(f)(Tensor([1.0, 2.0])))       # [2. 4.]   == d(xᵀx)/dx == 2x
print(hessian(f)([1.0, 2.0]))            # [[2 0]    == d²(xᵀx)/dxdx == 2I
                                         #  [0 2]]
```

`grad`, `vjp`, `jvp`, `hvp`, and `hessian` form the functional API in `nanoad.functional`. `hvp` (Hessian-vector product) is grad-of-(grad·v) — the canonical reverse-over-reverse pattern. See `examples/cg_newton.py` for a Newton step solved via conjugate gradient using only `hvp`.

## Train a CNN on MNIST

```python
from nanoad import Tensor
from nanoad.data import DataLoader, load_mnist
from nanoad.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear, Sequential, CrossEntropy
from nanoad.optim import Adam

x_train, y_train = load_mnist("train", flatten=False)

model = Sequential(
    Conv2d(1, 8, kernel_size=3, padding=1), BatchNorm2d(8), ReLU(), MaxPool2d(2),
    Conv2d(8, 16, kernel_size=3, padding=1), BatchNorm2d(16), ReLU(), MaxPool2d(2),
    Flatten(), Linear(16 * 7 * 7, 10),
)
loss_fn = CrossEntropy()
opt = Adam(model.parameters(), lr=1e-3)

for x_batch, y_batch in DataLoader(x_train, y_train, batch_size=128, shuffle=True):
    loss = loss_fn(model(Tensor(x_batch)), y_batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Reaches **96.84% test accuracy in 1 epoch** (~138 s pure-NumPy CPU on a Ryzen 7). See `examples/train_cnn_mnist.py` for the full script.

## Train a tiny GPT on Shakespeare

```python
from nanoad import cross_entropy
from nanoad.data import CharTokenizer, load_tinyshakespeare
from nanoad.nn import GPT
from nanoad.optim import AdamW, CosineWarmupLR, clip_grad_norm

text = load_tinyshakespeare()                         # ~1.1 MB, sha256-verified
tok  = CharTokenizer(text)                            # 65-char vocab, deterministic
ids  = tok.encode(text)

model = GPT(vocab_size=tok.vocab_size, embed_dim=64, num_heads=4,
            num_layers=4, block_size=32)
opt   = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = CosineWarmupLR(opt, warmup_steps=100, total_steps=2000, min_lr=3e-5)

# per step: sample window -> forward -> CE -> backward -> clip -> AdamW -> sched
# (full loop in examples/train_tinygpt.py)

prompt = tok.encode("ROMEO:")[None, :]
print(tok.decode(model.generate(prompt, max_new_tokens=400, top_k=20)[0]))
```

After 2,000 AdamW iterations (~42 minutes pure-NumPy CPU), val loss reaches **2.27** (perplexity ~9.7) and the model produces:

```
ROMEO:
Yat wich four sumove, the heat bet to tho whe thak
Thar come thake hart
There de solecheshgo netlaing oimeds te houe me be so ingheand
And my nof sthere to thee has ton mawar dund.

MHary iss gerit the wath theanoug fere ilige buth mows tousse
...
```

Word boundaries and capitalization are right; longer "words" are pronounceable but invented; sentence-level meaning is gibberish — the model is clearly past unigram structure but undertrained at this size. Scaling up `embed_dim` / `num_layers` / `max_iters` is a knob, not a code change. Full sample at `examples/tinygpt_sample.txt`.

## What's implemented

| Area | Items |
| --- | --- |
| Tensor / autograd | `Tensor`, reverse-mode dynamic tape, scalar-seeded `backward()`, gradients are themselves differentiable (HOA-capable) |
| Functional autograd | `grad`, `vjp`, `jvp`, `hvp`, `hessian` |
| Elementwise ops | `+`, `-`, `*`, `/`, `**`, `relu`, `tanh`, `softmax` |
| Linear algebra | `matmul` (`@`), `bmm` (batched, leading-dim broadcast), `transpose`, `reshape`, `sum`, `mean` |
| Conv / pool | `conv2d` (im2col), `max_pool2d`, `BatchNorm2d` |
| Embedding | `embedding` op (gather/scatter mutual VJP), `CharTokenizer` |
| Transformer modules | `LayerNorm`, `Embedding`, `MultiHeadAttention` (causal), `GELU`, `GPT` (with `generate`) |
| MLP / CNN modules | `Linear` (N-d input), `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten`, `ReLU`, `Tanh`, `Sequential` |
| Loss | `CrossEntropy` (fused log-softmax + NLL) |
| Optimizers / schedulers | `SGD`, `Adam`, `AdamW`, `CosineWarmupLR`, `clip_grad_norm` |
| Data | `load_mnist`, `load_cifar10`, `load_tinyshakespeare`, `DataLoader` (with optional batch transforms) |
| Misc | autograd graph DOT export (`nanoad.viz.draw`) |

Every op has a finite-difference gradient check in the test suite. The HOA layer adds a separate suite of higher-order checks (e.g. `d²(x²)/dx² = 2`, the Hessian of `xᵀAx` is `A + Aᵀ`, `hvp` matches finite differences end-to-end through a tanh-MLP loss).

## Op benchmarks

Forward / backward time and peak Python-heap memory for each op, batch 128, on a single core:

| op            | shape                                  |     fwd us |     bwd us |   peak KiB |
| ------------- | -------------------------------------- | ---------- | ---------- | ---------- |
| add           | (128, 64, 14, 14)                      |     7873.0 |     5492.8 |    87809.5 |
| sub           | (128, 64, 14, 14)                      |     8349.5 |     5649.8 |    87809.5 |
| mul           | (128, 64, 14, 14)                      |     9265.5 |    14014.9 |    87811.0 |
| div           | (128, 64, 14, 14)                      |    10128.0 |    27912.4 |   112898.6 |
| power         | (128, 64, 14, 14)                      |    36799.4 |    18492.9 |    75266.3 |
| matmul        | (128, 784) @ (784, 128)                |    16969.3 |    21695.5 |     4178.7 |
| transpose     | (128, 16, 7, 7)                        |      287.2 |      168.4 |     3202.5 |
| reshape       | (128, 784) -> (128, 1, 28, 28)         |      236.8 |      139.0 |     3202.1 |
| sum           | (128, 64, 14, 14) axis=(2,3)           |     1247.2 |     1859.1 |    37632.6 |
| mean          | (128, 64, 14, 14) axis=(2,3)           |     1864.3 |     7291.9 |    37891.2 |
| relu          | (128, 64, 14, 14)                      |     9537.2 |     9531.9 |    64354.9 |
| tanh          | (128, 64, 14, 14)                      |    16585.7 |    18426.3 |    87810.1 |
| softmax       | (128, 10) axis=1                       |       42.6 |       34.6 |       83.9 |
| cross_entropy | logits (128, 10), targets (128,)       |       67.7 |       26.3 |       74.0 |
| conv2d        | x(128, 8, 14, 14) w(16, 8, 3, 3) pad=1 |    23720.2 |    60441.7 |    42070.6 |
| max_pool2d    | (128, 16, 14, 14) kernel=2             |     6473.8 |     3302.6 |    13332.1 |
| layer_norm    | (32, 32, 64) last-axis                 |      803.5 |     3010.6 |    14458.0 |
| mha_causal    | (32, 32, 64) heads=4                   |    16656.4 |    33814.5 |    44879.6 |
| gpt_step      | 2L × 2H × 32D, B=8 T=16 V=64           |     5738.9 |    12016.0 |    21884.1 |

Reproduce with `python bench/run.py`. Pre-M10 rows come from a Ryzen 7 5800X, NumPy 2.0 OpenBLAS; the three M10 rows (`layer_norm`, `mha_causal`, `gpt_step`) were measured separately and absolute times will not align between sections. Relative ordering is the durable signal.

## Documentation

- [Reverse-mode autodiff in 200 words](docs/autodiff.md)
- [The tape: how `_prev` records the graph](docs/tape.md)
- [Broadcasting in the backward pass](docs/broadcasting.md)
- [How to add a new op](docs/adding-an-op.md)

## Scope and non-goals

nanoad is a teaching reference, not a PyTorch replacement.

- **No GPU.** NumPy CPU only.
- **No fused kernels.** Each op is a small NumPy call plus a registered VJP — readable, not fast.
- **No autodiff through every captured value.** A handful of ops snapshot non-differentiable state in their forward context — `relu`'s mask, `conv2d`'s im2col indices, `max_pool2d`'s argmax, `cross_entropy`'s softmax probabilities, `Embedding`'s integer indices. Second derivatives through them are zero (`relu`, mathematically) or unsupported (the rest). The HOA-capable activations (`tanh`, `softmax`) recompute their forward via the public op so `x` stays in the gradient graph.
- **No streaming dataset API.** Loaders return eager NumPy arrays; `DataLoader` shuffles and slices in memory.

If you want to *use* deep learning, use PyTorch. If you want to *understand* an autograd engine end-to-end — including how higher-order gradients fall out of a tape that's differentiable in both directions, and how a transformer composes from twelve-ish primitives — this is one. The library lives in the same niche as `micrograd` (smaller and more readable than nanoad, but scalar-only) and `tinygrad` (more capable than nanoad, but less linear to read).

## License

MIT. See [LICENSE](LICENSE).
