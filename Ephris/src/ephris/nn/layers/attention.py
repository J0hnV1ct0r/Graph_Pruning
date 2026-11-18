from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .feedforward import FeedForward
from .scaling import QueryScale
from .weights import LayerNorm, Linear, linear, parameter

_LOW_PRECISION_CUDA_SDPA_BACKENDS = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def split_attention_heads(x: Tensor, *, heads: int, head_width: int) -> Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected [batch, seq, hidden] tensor, got {tuple(x.shape)}")
    batch_size = int(x.size(0))
    seq_len = int(x.size(1))
    return x.view(batch_size, seq_len, int(heads), int(head_width)).transpose(1, 2)


def merge_attention_heads(x: Tensor, *, width: int) -> Tensor:
    if x.dim() != 4:
        raise ValueError(
            f"Expected [batch, heads, seq, head_width] tensor, got {tuple(x.shape)}"
        )
    batch_size = int(x.size(0))
    seq_len = int(x.size(2))
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, int(width))


def _run_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    heads: int,
    head_width: int,
    width: int,
    scaling: nn.Module | None = None,
) -> Tensor:
    q = split_attention_heads(query, heads=heads, head_width=head_width)
    k = split_attention_heads(key, heads=heads, head_width=head_width)
    v = split_attention_heads(value, heads=heads, head_width=head_width)
    if scaling is not None:
        q = scaling(q, int(key.size(1)))
    if q.device.type == "cuda" and q.dtype in {torch.float16, torch.bfloat16}:
        with sdpa_kernel(list(_LOW_PRECISION_CUDA_SDPA_BACKENDS)):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    else:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    return merge_attention_heads(out, width=width)


class Attention(nn.Module):
    """Batch-first dense multi-head attention backed by PyTorch SDPA."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        scaling: nn.Module | None = None,
    ):
        super().__init__()
        if int(width) <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if int(heads) <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if int(width) % int(heads) != 0:
            raise ValueError(
                f"width must be divisible by heads: {width} % {heads} != 0"
            )
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = self.width // self.heads
        self.in_proj_weight = parameter(3 * self.width, self.width)
        self.in_proj_bias = parameter(3 * self.width)
        self.out_proj = Linear(self.width, self.width)
        self.scaling = scaling

    def _project(self, x: Tensor, offset: int) -> Tensor:
        start = int(offset) * self.width
        end = start + self.width
        return linear(x, self.in_proj_weight[start:end], self.in_proj_bias[start:end])

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("Attention expects [batch, seq, hidden] tensors")
        if key.shape != value.shape:
            raise ValueError(
                f"key/value shape mismatch: {tuple(key.shape)} != {tuple(value.shape)}"
            )
        if int(query.size(0)) != int(key.size(0)):
            raise ValueError("query/key batch dimensions must match")
        if int(query.size(-1)) != self.width or int(key.size(-1)) != self.width:
            raise ValueError(
                f"Expected width={self.width}, got query={int(query.size(-1))}, key={int(key.size(-1))}"
            )
        scaling = self.scaling
        if query is key and key is value:
            projected = linear(query, self.in_proj_weight, self.in_proj_bias)
            projected_query, projected_key, projected_value = projected.split(
                self.width, dim=-1
            )
        elif key is value:
            projected_query = self._project(query, 0)
            projected_kv = linear(
                key,
                self.in_proj_weight[self.width :],
                self.in_proj_bias[self.width :],
            )
            projected_key, projected_value = projected_kv.split(self.width, dim=-1)
        else:
            projected_query = self._project(query, 0)
            projected_key = self._project(key, 1)
            projected_value = self._project(value, 2)
        return self.out_proj(
            _run_sdpa(
                projected_query,
                projected_key,
                projected_value,
                heads=self.heads,
                head_width=self.head_width,
                width=self.width,
                scaling=scaling,
            )
        )


class AttentionLayer(nn.Module):
    """Post-norm attention and feed-forward residual block."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        scaling_hidden_width: int = 64,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        width = int(width)
        heads = int(heads)
        if width <= 0 or heads <= 0:
            raise ValueError("width and heads must be positive")
        if width % heads != 0:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.attention = Attention(
            width=width,
            heads=heads,
            scaling=QueryScale(
                heads=heads,
                head_width=width // heads,
                hidden_width=int(scaling_hidden_width),
            ),
        )
        self.attention_norm = LayerNorm(width)
        self.feedforward = FeedForward(width=width, expansion=int(expansion))
        self.feedforward_norm = LayerNorm(width)

    def _finish(self, query: Tensor, attention_delta: Tensor) -> Tensor:
        state = self.attention_norm(query + attention_delta)
        return self.feedforward_norm(state + self.feedforward(state))

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        if query.dim() != 3 or context.dim() != 3:
            raise ValueError("AttentionLayer expects [batch, tokens, hidden] tensors")
        if int(query.size(0)) != int(context.size(0)):
            raise ValueError("query and context batch dimensions must match")
        if int(query.size(-1)) != self.width or int(context.size(-1)) != self.width:
            raise ValueError(f"AttentionLayer expects width={self.width}")
        return self._finish(query, self.attention(query, context, context))

    @torch.no_grad()
    def forward_context_chunks(
        self,
        query: Tensor,
        context_chunk_factory: Callable[[], Iterator[Tensor]],
        *,
        context_length: int,
    ) -> Tensor:
        """Evaluate exact attention while materializing one context chunk at a time."""
        if query.dim() != 3 or int(query.size(-1)) != self.width:
            raise ValueError(
                f"query must have shape [batch, tokens, {self.width}], got {tuple(query.shape)}"
            )
        if int(context_length) <= 0:
            raise ValueError("context_length must be positive")
        attention = self.attention
        head_width = self.width // self.heads
        projected_query = split_attention_heads(
            attention._project(query, 0), heads=self.heads, head_width=head_width
        )
        scaling = attention.scaling
        if scaling is not None:
            projected_query = scaling(projected_query, int(context_length))
        batch_size = int(query.size(0))
        query_tokens = int(query.size(1))
        running_max = torch.full(
            (batch_size, self.heads, query_tokens),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        weight_sum = torch.zeros_like(running_max)
        weighted_value_sum = torch.zeros(
            (batch_size, self.heads, query_tokens, head_width),
            dtype=torch.float32,
            device=query.device,
        )
        seen = 0
        # Online softmax: rescale previous sums whenever a larger logit arrives.
        # Accumulators remain FP32 even when the attention projections use BF16.
        for context in context_chunk_factory():
            if context.dim() != 3 or int(context.size(0)) != batch_size:
                raise ValueError(
                    f"context chunks must have shape [batch, tokens, hidden], got {tuple(context.shape)}"
                )
            if int(context.size(-1)) != self.width:
                raise ValueError(f"context chunks must use width={self.width}")
            context = context.to(device=query.device, dtype=query.dtype)
            seen += int(context.size(1))
            key = split_attention_heads(
                attention._project(context, 1),
                heads=self.heads,
                head_width=head_width,
            )
            value = split_attention_heads(
                attention._project(context, 2),
                heads=self.heads,
                head_width=head_width,
            )
            scores = torch.matmul(
                projected_query, key.transpose(-2, -1)
            ).float() / math.sqrt(head_width)
            chunk_max = scores.amax(dim=-1)
            next_max = torch.maximum(running_max, chunk_max)
            previous_scale = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - next_max),
                torch.zeros_like(running_max),
            )
            weights = torch.exp(scores - next_max.unsqueeze(-1))
            weight_sum = weight_sum * previous_scale + weights.sum(dim=-1)
            weighted_value_sum = weighted_value_sum * previous_scale.unsqueeze(
                -1
            ) + torch.matmul(weights, value.float())
            running_max = next_max
        if seen != int(context_length):
            raise ValueError(
                f"Expected {int(context_length)} context tokens, received {seen}"
            )
        attention_delta = merge_attention_heads(
            (
                weighted_value_sum
                / weight_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
            ).to(query.dtype),
            width=self.width,
        )
        return self._finish(query, attention.out_proj(attention_delta))
