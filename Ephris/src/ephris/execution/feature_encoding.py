"""Bounded-memory feature encoding: cached tokens or streamed replay."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .memory import ChunkBudget

if TYPE_CHECKING:
    from ..model import EphrisModel

logger = logging.getLogger(__name__)

_CACHE_BUDGET_FRACTION = 0.3
_WORKSPACE_BUDGET_FRACTION = 0.4
_MAX_CACHED_FEATURE_CHUNK = 16
_FEATURE_WORKSPACE_FACTOR = 20


@torch.no_grad()
def encode_cached(
    model: EphrisModel,
    x: Tensor,
    labels: Tensor,
    context_mask: Tensor,
    feature_type_ids: Tensor,
    edge_index: Tensor,
    *,
    chunk_features: int,
    feature_order: Tensor | None = None,
) -> Tensor:
    """Cache [N, F, d] tokens; finish each graph stage before feature write-back."""
    if chunk_features < 1:
        raise ValueError("chunk_features must be positive.")
    num_nodes, num_features = x.shape
    tokenizer = model.feature_tokenizer
    node_context, _ = tokenizer.prepare_node_context(
        x,
        feature_type_ids,
        chunk_budget=None,
        feature_order=feature_order,
        feature_chunk_size=chunk_features,
    )
    feature_state: Tensor | None = None
    start = 0
    for chunk in tokenizer.iter_feature_token_chunks(
        x,
        labels,
        context_mask,
        feature_type_ids=feature_type_ids,
        node_context=node_context,
        feature_order=feature_order,
        feature_chunk_size=chunk_features,
    ):
        chunk = model.refiners[0].feature_refiner(chunk)
        if feature_state is None:
            feature_state = chunk.new_empty((num_nodes, num_features, chunk.size(-1)))
        end = start + chunk.size(1)
        feature_state[:, start:end].copy_(chunk)
        start = end
    if feature_state is None:
        raise ValueError("Cached encoding requires nonempty node features.")
    del chunk, node_context
    edge_index = edge_index.to(feature_state.device)
    shape_reference = torch.empty(
        (1, 1, model.width),
        device=feature_state.device,
        dtype=model._feature_compute_dtype(),
    ).expand(num_nodes, -1, -1)

    def iter_cached_features() -> Iterator[Tensor]:
        for offset in range(0, num_features, chunk_features):
            yield feature_state[:, offset : offset + chunk_features]

    def apply_feature_chunks(function: Callable[[Tensor], Tensor]) -> None:
        nonlocal feature_state
        input_dtype = feature_state.dtype
        for offset in range(0, num_features, chunk_features):
            end = min(num_features, offset + chunk_features)
            result = function(feature_state[:, offset:end].to(input_dtype))
            if result.dtype != feature_state.dtype:
                feature_state = feature_state.to(result.dtype)
            feature_state[:, offset:end].copy_(result)
            del result

    for index, block in enumerate(model.refiners):
        if index:
            apply_feature_chunks(block.feature_refiner)
        nodes = block.graph_refiner.gather_streamed(
            shape_reference, iter_cached_features, num_features=num_features
        )
        propagated = block.graph_refiner.propagate(nodes, edge_index)
        del nodes
        apply_feature_chunks(
            lambda chunk, nodes=propagated: block.graph_refiner.broadcast(chunk, nodes)
        )
        del propagated
    apply_feature_chunks(model.compressor.feature_refiner)
    return model.compressor.gather_streamed(
        shape_reference, iter_cached_features, num_features=num_features
    )


@torch.no_grad()
def encode_cached_or_replay(
    model: EphrisModel,
    x: Tensor,
    labels: Tensor,
    context_mask: Tensor,
    feature_type_ids: Tensor,
    edge_index: Tensor,
    *,
    chunk_budget: ChunkBudget | None,
    feature_order: Tensor | None = None,
) -> Tensor:
    """Try a conservative cache allocation, shrinking feature chunks after OOM."""
    chunk = 0
    if chunk_budget is not None and chunk_budget.enabled:
        chunk_budget.refresh()
        weight = model.feature_tokenizer.cell_encoder.skip_proj.weight
        bytes_per_feature = x.size(0) * model.width * max(4, weight.element_size())
        cache_bytes = bytes_per_feature * x.size(1)
        if cache_bytes <= int(chunk_budget.target_bytes * _CACHE_BUDGET_FRACTION):
            workspace_bytes = int(
                chunk_budget.target_bytes * _WORKSPACE_BUDGET_FRACTION
            )
            chunk = min(
                _MAX_CACHED_FEATURE_CHUNK,
                x.size(1),
                workspace_bytes
                // max(1, bytes_per_feature * _FEATURE_WORKSPACE_FACTOR),
            )
    while chunk >= 1:
        logger.info("Feature model route=cached chunk_features=%d", chunk)
        try:
            return encode_cached(
                model,
                x,
                labels,
                context_mask,
                feature_type_ids,
                edge_index,
                chunk_features=chunk,
                feature_order=feature_order,
            )
        except torch.OutOfMemoryError:
            logger.warning("Cached feature OOM; retrying with smaller feature chunks")
        gc.collect()
        device = model.feature_tokenizer.cell_encoder.skip_proj.weight.device
        if device.type == "cuda":
            torch.cuda.empty_cache()
        chunk //= 2
    logger.info("Feature model route=replay")
    return encode_replayed(
        model,
        x,
        labels,
        context_mask,
        feature_type_ids,
        edge_index,
        chunk_budget=chunk_budget,
        feature_order=feature_order,
    )


@torch.no_grad()
def encode_replayed(
    model: EphrisModel,
    x_preprocessed: Tensor,
    y: Tensor,
    context_mask: Tensor,
    feature_type_ids: Tensor,
    edge_index: Tensor,
    *,
    chunk_budget: ChunkBudget | None,
    feature_order: Tensor | None = None,
) -> Tensor:
    """Exact bounded-memory model; prior blocks are replayed per feature chunk."""
    num_nodes = int(x_preprocessed.size(0))
    num_features = int(x_preprocessed.size(1))
    device = model._model_device()
    dtype = model._feature_compute_dtype()
    labels = y.to(device=device)
    observed = context_mask.to(device=device)
    edge_index = edge_index.to(device=device)
    graph_node_cache: list[Tensor] = []
    node_context, chunk_features = model.feature_tokenizer.prepare_node_context(
        x_preprocessed,
        feature_type_ids,
        chunk_budget=chunk_budget,
        feature_order=feature_order,
    )

    def base_chunks() -> Iterator[Tensor]:
        yield from model.feature_tokenizer.iter_feature_token_chunks(
            x_preprocessed,
            labels,
            observed,
            feature_type_ids=feature_type_ids,
            chunk_budget=chunk_budget,
            feature_order=feature_order,
            node_context=node_context,
            feature_chunk_size=chunk_features,
        )

    def features_at(block_index: int) -> Iterator[Tensor]:
        for chunk in base_chunks():
            state = chunk
            for prior_index in range(block_index):
                prior = model.refiners[prior_index]
                state = prior.feature_refiner(state)
                state = prior.graph_refiner.broadcast(
                    state, graph_node_cache[prior_index]
                )
            if block_index < len(model.refiners):
                state = model.refiners[block_index].feature_refiner(state)
            else:
                state = model.compressor.feature_refiner(state)
            yield state

    reference = torch.empty((1, 1, model.width), device=device, dtype=dtype).expand(
        num_nodes, -1, -1
    )
    for block_index, block in enumerate(model.refiners):
        node_state = block.graph_refiner.gather_streamed(
            reference,
            lambda index=block_index: features_at(index),
            num_features=num_features,
        )
        graph_node_cache.append(block.graph_refiner.propagate(node_state, edge_index))
    return model.compressor.gather_streamed(
        reference,
        lambda: features_at(len(model.refiners)),
        num_features=num_features,
    )
