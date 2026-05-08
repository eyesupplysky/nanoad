# nanoad

A from-scratch reverse-mode automatic differentiation engine and small neural-net framework, written in Python on top of NumPy. The whole library is meant to be readable in one sitting — the tape, the ops, the optimizer, the training loop. No PyTorch, no JAX, no Autograd. Just NumPy and roughly 1,500 lines of Python.

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

## What's implemented

| Area | Items |
| --- | --- |
| Tensor / autograd | `Tensor`, reverse-mode tape, scalar-seeded `backward()` |
| Elementwise | `+`, `-`, `*`, `/`, `**`, `relu`, `tanh`, `softmax` |
| Linear algebra | `matmul` (`@`), `transpose`, `reshape`, `sum`, `mean` |
| Convolutional | `conv2d` (im2col), `max_pool2d`, `BatchNorm2d` |
| Layers | `Linear`, `Conv2d`, `ReLU`, `Tanh`, `BatchNorm2d`, `MaxPool2d`, `Flatten`, `Sequential` |
| Loss | `CrossEntropy` (fused log-softmax + NLL) |
| Optimizers | `SGD`, `Adam` |
| Data | `load_mnist`, `load_cifar10`, `DataLoader` (with optional batch transforms) |
| Misc | autograd graph DOT export (`nanoad.viz.draw`) |

Every op has a finite-difference gradient check in the test suite.

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

Reproduce with `python bench/run.py`. Numbers come from a Ryzen 7 5800X, NumPy 2.0 OpenBLAS; relative ordering is the durable signal — absolute times will vary across machines.

## Documentation

- [Reverse-mode autodiff in 200 words](docs/autodiff.md)
- [The tape: how `_prev` records the graph](docs/tape.md)
- [Broadcasting in the backward pass](docs/broadcasting.md)
- [How to add a new op](docs/adding-an-op.md)

## Scope and non-goals

nanoad is a teaching reference, not a PyTorch replacement.

- **No GPU.** NumPy CPU only.
- **No batched / 1-d matmul.** `matmul` requires both operands 2-d.
- **No higher-order gradients.** `backward()` populates `.grad` from a scalar; differentiating through a backward pass is out of scope.
- **No fused kernels.** Each op is one or two NumPy calls plus a Python closure for backward.

If you want to *use* deep learning, use PyTorch. If you want to *understand* an autograd engine end-to-end, this is one. The library exists in the same niche as `micrograd` (more readable than nanoad, but scalar-only) and `tinygrad` (more capable than nanoad, but less linear to read).

## License

MIT. See [LICENSE](LICENSE).
