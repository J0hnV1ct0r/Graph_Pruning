from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers.feedforward import FeedForward
from .layers.graph_attention import GraphAttention
from .layers.weights import LayerNorm, parameter


class GraphLayer(nn.Module):
    """MP_ICL: neighborhood-aware attention and FFN, each with residual + LN."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        scaling_hidden_width: int = 64,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if int(width) <= 0 or int(heads) <= 0:
            raise ValueError("width and heads must be positive")
        if int(width) % int(heads) != 0:
            raise ValueError("width must be divisible by graph attention heads")
        self.attention = GraphAttention(
            width=int(width),
            head_width=int(width) // int(heads),
            heads=int(heads),
            scaling_hidden_width=int(scaling_hidden_width),
        )
        self.attention_norm = LayerNorm(int(width))
        self.feedforward = FeedForward(width=int(width), expansion=int(expansion))
        self.feedforward_norm = LayerNorm(int(width))

    def forward(
        self,
        state: Tensor,
        edge_index: Tensor,
        *,
        num_real_nodes: int,
        num_globals: int,
    ) -> Tensor:
        delta = self.attention(
            state,
            edge_index,
            num_real_nodes=num_real_nodes,
            num_globals=int(num_globals),
        )
        state = self.attention_norm(state + delta)
        return self.feedforward_norm(state + self.feedforward(state))


class GraphBlock(nn.Module):
    """Successive MP_ICL blocks sharing stage-local learned global nodes."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        num_layers: int,
        num_globals: int,
        scaling_hidden_width: int = 64,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if int(num_layers) < 2:
            raise ValueError("GraphBlock requires at least two layers")
        if int(num_globals) <= 0:
            raise ValueError("GraphBlock requires a positive global node count")
        self.width = int(width)
        self.num_globals = int(num_globals)
        self.global_nodes = parameter(self.num_globals, self.width)
        self.layers = nn.ModuleList(
            (
                GraphLayer(
                    width=self.width,
                    heads=int(heads),
                    scaling_hidden_width=int(scaling_hidden_width),
                    expansion=int(expansion),
                )
                for _ in range(int(num_layers))
            )
        )

    def forward(
        self,
        node_state: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        if node_state.dim() != 2 or int(node_state.size(1)) != self.width:
            raise ValueError(
                f"node_state must have shape [num_nodes, {self.width}], got {tuple(node_state.shape)}"
            )
        num_real_nodes = int(node_state.size(0))
        global_state = self.global_nodes.to(
            device=node_state.device, dtype=node_state.dtype
        )
        graph_state = torch.cat((node_state, global_state), dim=0)
        edge_index = edge_index.to(device=node_state.device)
        for layer in self.layers:
            graph_state = layer(
                graph_state,
                edge_index,
                num_real_nodes=num_real_nodes,
                num_globals=self.num_globals,
            )
        return graph_state[:num_real_nodes]
