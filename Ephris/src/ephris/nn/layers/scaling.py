from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .weights import Linear


def _log_key_count(
    key_count: int | Tensor, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if isinstance(key_count, Tensor):
        return key_count.to(device=device, dtype=dtype).clamp_min(1).log()
    return torch.tensor(math.log(max(1, int(key_count))), device=device, dtype=dtype)


class QueryScale(nn.Module):
    """query * f(log key_count) * [1 + tanh(g(query))]; key_count is neighborhood or key-set size."""

    def __init__(
        self,
        heads: int,
        head_width: int,
        *,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        if int(hidden_width) <= 0:
            raise ValueError(f"hidden_width must be positive, got {hidden_width}")
        self.heads = int(heads)
        self.head_width = int(head_width)
        size_scale_dim = self.heads * self.head_width
        query_out_dim = self.head_width
        self.length = nn.Sequential(
            Linear(1, int(hidden_width)),
            nn.GELU(),
            Linear(int(hidden_width), size_scale_dim),
        )
        self.content = nn.Sequential(
            Linear(self.head_width, int(hidden_width)),
            nn.GELU(),
            Linear(int(hidden_width), query_out_dim),
        )

    def forward(self, query: Tensor, key_count: int | Tensor) -> Tensor:
        log_key_count = _log_key_count(key_count, query.device, query.dtype)
        log_key_count = (
            log_key_count.reshape(1, 1)
            if log_key_count.dim() == 0
            else log_key_count.reshape(-1, 1)
        )
        size_batch = int(log_key_count.size(0))
        if size_batch != 1 and size_batch != int(query.size(0)):
            raise ValueError(
                f"QueryScale length batch must be 1 or {int(query.size(0))}, got {size_batch}"
            )
        size_scales = self.length(log_key_count).view(
            size_batch, self.heads, 1, self.head_width
        )
        modulation = 1 + torch.tanh(self.content(query))
        return query * size_scales * modulation
