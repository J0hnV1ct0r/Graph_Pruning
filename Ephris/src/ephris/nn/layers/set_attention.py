from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from .attention import AttentionLayer
from .weights import parameter


class InducedSetAttentionBlock(nn.Module):
    """ISAB with block-local learned inducing tokens and explicit read/write APIs."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        num_latents: int,
        scaling_hidden_width: int = 64,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if int(num_latents) <= 0:
            raise ValueError("num_latents must be positive")
        self.width = int(width)
        self.num_latents = int(num_latents)
        self.latents = parameter(self.num_latents, self.width)
        block_kwargs = dict(
            width=self.width,
            heads=int(heads),
            scaling_hidden_width=int(scaling_hidden_width),
            expansion=int(expansion),
        )
        self.gather_attention = AttentionLayer(**block_kwargs)
        self.broadcast_attention = AttentionLayer(**block_kwargs)

    def initial_inducing(self, set_state: Tensor) -> Tensor:
        if set_state.dim() != 3 or int(set_state.size(-1)) != self.width:
            raise ValueError(
                f"set_state must have shape [batch, tokens, {self.width}], got {tuple(set_state.shape)}"
            )
        return (
            self.latents.to(device=set_state.device, dtype=set_state.dtype)
            .unsqueeze(0)
            .expand(int(set_state.size(0)), -1, -1)
        )

    def gather(self, set_state: Tensor) -> Tensor:
        return self.gather_attention(self.initial_inducing(set_state), set_state)

    @torch.no_grad()
    def gather_streamed(
        self,
        reference: Tensor,
        set_chunk_factory: Callable[[], Iterator[Tensor]],
        *,
        set_size: int,
    ) -> Tensor:
        return self.gather_attention.forward_context_chunks(
            self.initial_inducing(reference),
            set_chunk_factory,
            context_length=int(set_size),
        )

    def broadcast(self, set_state: Tensor, inducing_state: Tensor) -> Tensor:
        return self.broadcast_attention(set_state, inducing_state)

    def forward(self, set_state: Tensor) -> Tensor:
        inducing_state = self.gather(set_state)
        return self.broadcast(set_state, inducing_state)
