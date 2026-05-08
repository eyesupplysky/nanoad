# The tape

nanoad's autograd graph is a *dynamic tape*: it gets built as ops execute on the forward pass, with no separate compilation step.

## Per-tensor parent links

Every `Tensor` carries four fields that matter for autograd:

```python
class Tensor:
    data       # NDArray, float64
    grad       # NDArray, same shape as data
    _prev      # tuple[Tensor, ...]  — the tensors this one was computed from
    _backward  # Callable[[], None]  — closure that propagates this node's grad
```

`_prev` is the edge list: it points *upstream* to the parents in the computation. There is no global "graph" object — every Tensor stores its own incoming edges, and the graph is whatever you can reach by walking `_prev` from a root.

A *leaf* tensor has `_prev = ()` and a no-op `_backward`. Any tensor returned by an op has both fields populated.

## A forward op records itself

Look at `add` (in `src/nanoad/ops/arithmetic.py`):

```python
def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, _prev=(a, b), _op="+")

    def _backward() -> None:
        a.grad += unbroadcast(out.grad, a.shape)
        b.grad += unbroadcast(out.grad, b.shape)

    out._backward = _backward
    return out
```

Three things happen:

1. The forward computation runs (`a.data + b.data`).
2. A new `Tensor` is constructed with `_prev=(a, b)` — the parents.
3. A `_backward` closure is captured. It reads `out.grad` (set later by `backward()`) and accumulates contributions into the parents' `.grad`. **It must use `+=`, never `=`** — a parent can be used by several ops, and their grads must combine.

## `backward()` walks the tape

```python
def backward(self) -> None:
    if self.data.ndim != 0:
        raise RuntimeError("backward() requires scalar output")
    topo = _topological_order(self)
    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
        node._backward()
```

Steps:

1. **Require a scalar root.** Without that, the seed gradient `dL/dL = 1` is ill-defined. If your output is multi-dimensional, reduce it first (`.sum()`, `.mean()`).
2. **Topological order.** `_topological_order` is an iterative post-order DFS over `_prev`. Each Tensor appears exactly once, and every parent appears before any child that uses it.
3. **Seed.** `self.grad = ones_like(self.data)` — `dL/dL = 1`.
4. **Run closures in reverse.** Walking `reversed(topo)` guarantees that when we call `node._backward()`, every consumer of `node` has already deposited its contribution into `node.grad`. The closure can now read a complete upstream gradient and pass it to its own parents.

That ordering invariant is the reason topo sort matters. Skip it and you'll silently propagate partial gradients without an error.

## Grad accumulation, not assignment

Because closures use `+=`, calling `backward()` a second time without zeroing leaves first will *double* the gradients. This is by design — it lets you accumulate gradients across micro-batches if you want. In a normal training loop, call `optimizer.zero_grad()` (or `tensor.zero_grad()` directly) before each backward.

## Why per-tensor `_prev` rather than a global tape?

A global tape (one append-only list of ops, indexed by integers) is what most production frameworks use. It's faster (locality, no closure overhead) and easier to serialise. The tradeoff is readability — `_prev` makes the graph self-describing and lets you recover the structure with a single DFS, no separate index. nanoad is a teaching reference, so per-tensor wins.

## Cycles

There aren't any. A new Tensor's `_prev` only points to tensors that already existed; you can't add edges to existing tensors retroactively. The graph is a DAG by construction, and the iterative DFS in `_topological_order` doesn't need cycle detection.
