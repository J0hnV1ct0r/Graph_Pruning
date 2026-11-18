from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from .graph import GraphBlock
from .layers.set_attention import AttentionLayer, InducedSetAttentionBlock
from .layers.weights import parameter

_MAX_RESIDENT_EVAL_NODES = 4096
_EVAL_NODE_CHUNK_SIZE = 4096
_MAX_RESIDENT_EVAL_FEATURES = 64
_EVAL_FEATURE_CHUNK_SIZE = 64


def _eval_node_slices(num_nodes: int) -> Iterator[slice]:
    for start in range(0, int(num_nodes), _EVAL_NODE_CHUNK_SIZE):
        yield slice(start, min(start + _EVAL_NODE_CHUNK_SIZE, int(num_nodes)))


def _eval_feature_slices(num_features: int) -> Iterator[slice]:
    for start in range(0, int(num_features), _EVAL_FEATURE_CHUNK_SIZE):
        yield slice(start, min(start + _EVAL_FEATURE_CHUNK_SIZE, int(num_features)))


def _requires_eval_node_chunking(num_nodes: int) -> bool:
    return int(num_nodes) > _MAX_RESIDENT_EVAL_NODES


def _requires_eval_feature_chunking(num_features: int) -> bool:
    return int(num_features) > _MAX_RESIDENT_EVAL_FEATURES


class FeatureRefiner(nn.Module):
    """(a) Symmetric refinement: gather/broadcast over all nodes, per feature.

    Input and output: feature tokens [N, F, d]. Query labels are never used.
    """

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        num_latents: int,
        scaling_hidden_width: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.pool = InducedSetAttentionBlock(
            width=self.width,
            heads=int(heads),
            num_latents=int(num_latents),
            scaling_hidden_width=int(scaling_hidden_width),
            expansion=int(expansion),
        )

    def forward(self, feature_state: Tensor) -> Tensor:
        if feature_state.dim() != 3 or int(feature_state.size(-1)) != self.width:
            raise ValueError(
                f"feature_state must have shape [num_nodes, num_features, width], got {tuple(feature_state.shape)}"
            )
        by_feature = feature_state.transpose(0, 1).contiguous()
        num_nodes = int(by_feature.size(1))
        num_features = int(by_feature.size(0))
        chunk_nodes = _requires_eval_node_chunking(num_nodes)
        chunk_features = _requires_eval_feature_chunking(num_features)
        if chunk_nodes or chunk_features:
            output = torch.empty_like(by_feature)
            for feature_slice in _eval_feature_slices(num_features):
                feature_batch = by_feature[feature_slice]
                if not chunk_nodes:
                    output[feature_slice] = self.pool(feature_batch)
                    continue
                inducing_state = self.pool.gather_streamed(
                    feature_batch[:, :1],
                    lambda batch=feature_batch: (
                        batch[:, node_slice]
                        for node_slice in _eval_node_slices(num_nodes)
                    ),
                    set_size=num_nodes,
                )
                for node_slice in _eval_node_slices(num_nodes):
                    output[feature_slice, node_slice] = self.pool.broadcast(
                        feature_batch[:, node_slice], inducing_state
                    )
            return output.transpose(0, 1).contiguous()
        return self.pool(by_feature).transpose(0, 1).contiguous()


class GraphRefiner(nn.Module):
    """(b) Graph-aware refinement: gather -> MP_ICL -> broadcast.

    Gather maps [N, F, d] to [N, K_N, d]; MP uses flattened [N, K_N*d].
    Broadcast returns the propagated information to [N, F, d] tokens.
    """

    def __init__(
        self,
        *,
        width: int,
        set_heads: int,
        node_latents: int,
        graph_heads: int,
        num_layers: int,
        num_globals: int,
        scaling_hidden_width: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.node_latents = int(node_latents)
        self.node_width = self.width * self.node_latents
        self.pool = InducedSetAttentionBlock(
            width=self.width,
            heads=int(set_heads),
            num_latents=self.node_latents,
            scaling_hidden_width=int(scaling_hidden_width),
            expansion=int(expansion),
        )
        self.mp = GraphBlock(
            width=self.node_width,
            heads=int(graph_heads),
            num_layers=int(num_layers),
            num_globals=int(num_globals),
            scaling_hidden_width=int(scaling_hidden_width),
            expansion=int(expansion),
        )

    def gather(self, feature_state: Tensor) -> Tensor:
        num_nodes = int(feature_state.size(0))
        if _requires_eval_node_chunking(num_nodes):
            node_state = feature_state.new_empty(
                (num_nodes, self.node_latents, self.width)
            )
            for node_slice in _eval_node_slices(num_nodes):
                node_state[node_slice] = self.pool.gather(feature_state[node_slice])
            return node_state
        return self.pool.gather(feature_state)

    @torch.no_grad()
    def gather_streamed(
        self,
        reference: Tensor,
        feature_chunk_factory: Callable[[], Iterator[Tensor]],
        *,
        num_features: int,
    ) -> Tensor:
        return self.pool.gather_streamed(
            reference, feature_chunk_factory, set_size=int(num_features)
        )

    def propagate(self, node_state: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = int(node_state.size(0))
        flat_state = node_state.reshape(num_nodes, self.node_width)
        flat_state = self.mp(
            flat_state,
            edge_index,
        )
        return flat_state.reshape(num_nodes, self.node_latents, self.width)

    def broadcast(self, feature_state: Tensor, node_state: Tensor) -> Tensor:
        num_nodes = int(feature_state.size(0))
        if _requires_eval_node_chunking(num_nodes):
            output = torch.empty_like(feature_state)
            for node_slice in _eval_node_slices(num_nodes):
                output[node_slice] = self.pool.broadcast(
                    feature_state[node_slice], node_state[node_slice]
                )
            return output
        return self.pool.broadcast(feature_state, node_state)

    def forward(self, feature_state: Tensor, edge_index: Tensor) -> Tensor:
        node_state = self.gather(feature_state)
        node_state = self.propagate(node_state, edge_index)
        return self.broadcast(feature_state, node_state)


class RefineBlock(nn.Module):
    """One (a) symmetric -> (b) graph-aware refinement cycle."""

    def __init__(
        self, *, feature_refiner: FeatureRefiner, graph_refiner: GraphRefiner
    ) -> None:
        super().__init__()
        self.feature_refiner = feature_refiner
        self.graph_refiner = graph_refiner

    def forward(self, feature_state: Tensor, edge_index: Tensor) -> Tensor:
        feature_state = self.feature_refiner(feature_state)
        return self.graph_refiner(feature_state, edge_index)


class Compressor(nn.Module):
    """Final (a) refinement and gather: [N, F, d] -> [N, K_N, d].

    Unlike RefineBlock, this stage has no message passing or broadcast.
    """

    def __init__(
        self,
        *,
        feature_refiner: FeatureRefiner,
        width: int,
        set_heads: int,
        node_latents: int,
        scaling_hidden_width: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.feature_refiner = feature_refiner
        self.width = int(width)
        self.node_latents = int(node_latents)
        self.latents = parameter(self.node_latents, self.width)
        self.gather_attention = AttentionLayer(
            width=self.width,
            heads=int(set_heads),
            scaling_hidden_width=int(scaling_hidden_width),
            expansion=int(expansion),
        )

    def initial_nodes(self, reference: Tensor) -> Tensor:
        return (
            self.latents.to(device=reference.device, dtype=reference.dtype)
            .unsqueeze(0)
            .expand(int(reference.size(0)), -1, -1)
        )

    def gather(self, feature_state: Tensor) -> Tensor:
        num_nodes = int(feature_state.size(0))
        if _requires_eval_node_chunking(num_nodes):
            node_state = feature_state.new_empty(
                (num_nodes, self.node_latents, self.width)
            )
            for node_slice in _eval_node_slices(num_nodes):
                node_state[node_slice] = self.gather_attention(
                    self.initial_nodes(feature_state[node_slice]),
                    feature_state[node_slice],
                )
            return node_state
        return self.gather_attention(self.initial_nodes(feature_state), feature_state)

    @torch.no_grad()
    def gather_streamed(
        self,
        reference: Tensor,
        feature_chunk_factory: Callable[[], Iterator[Tensor]],
        *,
        num_features: int,
    ) -> Tensor:
        return self.gather_attention.forward_context_chunks(
            self.initial_nodes(reference),
            feature_chunk_factory,
            context_length=int(num_features),
        )

    def forward(self, feature_state: Tensor) -> Tensor:
        return self.gather(self.feature_refiner(feature_state))
