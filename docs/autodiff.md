# Reverse-mode autodiff in 200 words

Automatic differentiation is a way to compute exact gradients of any function expressed as a composition of differentiable primitives. It is not symbolic differentiation (which manipulates expressions) and it is not numerical differentiation (which subtracts finite differences). Each primitive op knows its own local derivative; AD chains those derivatives together to recover the gradient of the whole.

**Forward mode** evaluates the function and propagates a derivative *with respect to one input* alongside it. Cost: one full pass per input you want a derivative for. Useful when there are few inputs and many outputs.

**Reverse mode** evaluates the function, records every intermediate, then walks the graph *backwards* propagating a derivative *of one output* with respect to every node it passes. Cost: one forward pass plus one backward pass, regardless of how many inputs there are. Useful when there are many inputs (e.g. millions of model parameters) and one output (e.g. a scalar loss). This is what every neural-net library implements.

The backward pass is structured around the *vector–Jacobian product* (VJP): for each op `y = f(x)`, the backward closure receives `dL/dy` and contributes `dL/dy · ∂y/∂x` to `dL/dx`. nanoad implements one VJP per primitive op.
