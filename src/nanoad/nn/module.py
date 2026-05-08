"""Module base class for layers and composite networks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from nanoad.tensor import Tensor


class Module:
    """Base class for layers and composite networks. Subclasses override forward()."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the layer output. Must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must override forward()")

    def parameters(self) -> Iterator[Tensor]:
        """Yield every Tensor parameter held directly or transitively."""
        for value in vars(self).values():
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, list | tuple):
                for item in value:
                    if isinstance(item, Tensor):
                        yield item
                    elif isinstance(item, Module):
                        yield from item.parameters()

    def zero_grad(self) -> None:
        """Reset .grad on every parameter held directly or transitively."""
        for p in self.parameters():
            p.zero_grad()
