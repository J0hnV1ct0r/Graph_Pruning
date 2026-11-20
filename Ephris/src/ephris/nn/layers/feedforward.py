from __future__ import annotations

from torch import Tensor, nn

from .weights import Linear


class FeedForward(nn.Module):
    """Transformer-style feed-forward block."""

    def __init__(self, width: int, expansion: int = 4):
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        hidden_width = int(width) * int(expansion)
        self.linear1 = Linear(int(width), hidden_width)
        self.activation = nn.GELU()
        self.linear2 = Linear(hidden_width, int(width))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.activation(self.linear1(x)))
