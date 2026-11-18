"""Single-pass inference: permute, encode, propagate, and restore class order."""

from __future__ import annotations

import gc
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.context import GraphContext
from ..data.permutation import make_inference_permutation
from ..model import EphrisModel
from .feature_encoding import encode_cached_or_replay
from .memory import ChunkBudget, empty_cuda_cache
from .precision import bf16_autocast, configure_cuda_matmul_precision
from .readout import (
    _cuda_illegal_memory_help,
    _is_cuda_illegal_memory_access,
    project_logits_in_chunks,
)


def _make_chunk_budget(model: EphrisModel, device: torch.device) -> ChunkBudget | None:
    return ChunkBudget.for_device(
        device=device,
        enabled=model.feature_dynamic_chunking,
        memory_fraction=model.feature_dynamic_memory_fraction,
    )


@torch.inference_mode()
def infer_node_logits(
    model: EphrisModel,
    context: GraphContext,
    x_all: Tensor,
    labels: Tensor,
    context_mask: Tensor,
    device: torch.device,
    feature_type_ids: Tensor | None = None,
    feature_order: Tensor | None = None,
    chunk_budget: ChunkBudget | None = None,
) -> Tensor:
    num_nodes = int(x_all.size(0))
    configure_cuda_matmul_precision(device)
    if chunk_budget is None:
        chunk_budget = _make_chunk_budget(model, device)
    with bf16_autocast(device):

        def _run_once(*, stream_feature_tokens: bool) -> Tensor:
            if stream_feature_tokens:
                x_preprocessed, resolved_feature_type_ids = (
                    model.prepare_streaming_features(
                        x=x_all,
                        x_is_preprocessed=True,
                        feature_type_ids=feature_type_ids,
                    )
                )
            else:
                feature_state = model.tokenize_features(
                    x=x_all,
                    y=labels,
                    context_mask=context_mask,
                    x_is_preprocessed=True,
                    feature_type_ids=feature_type_ids,
                    chunk_budget=chunk_budget,
                    feature_order=feature_order,
                )
            edge_index_undir = model.edge_cache.prepare(
                context.data.edge_index,
                num_nodes=num_nodes,
                device=device,
            )
            if stream_feature_tokens:
                node_token_state = encode_cached_or_replay(
                    model,
                    x_preprocessed,
                    labels,
                    context_mask,
                    resolved_feature_type_ids,
                    edge_index_undir,
                    chunk_budget=chunk_budget,
                    feature_order=feature_order,
                )
            else:
                node_token_state = model.encode_features(
                    feature_state, edge_index_undir
                )
            node_state = model.flatten_node_state(node_token_state)
            node_state = model.inject_labels(node_state, labels, context_mask)
            node_state = model.propagate(
                node_state,
                edge_index_undir,
            )
            return project_logits_in_chunks(
                model=model,
                node_state=node_state,
                device=device,
                chunk_budget=chunk_budget,
            )

        stream_feature_tokens = model.should_stream_feature_tokens(
            x_all, chunk_budget=chunk_budget
        )
        try:
            return _run_once(stream_feature_tokens=stream_feature_tokens)
        except torch.OutOfMemoryError:
            if stream_feature_tokens:
                raise
        gc.collect()
        empty_cuda_cache(device)
        return _run_once(stream_feature_tokens=True)


@torch.inference_mode()
def _infer_logits_with_fallback(
    *,
    model: EphrisModel,
    context: GraphContext,
    x_all: Tensor,
    labels: Tensor,
    context_mask: Tensor,
    device: torch.device,
    chunk_budget: ChunkBudget | None,
    feature_type_ids: Tensor | None = None,
    feature_order: Tensor | None = None,
) -> Tensor:
    try:
        return infer_node_logits(
            model=model,
            context=context,
            x_all=x_all,
            labels=labels,
            context_mask=context_mask,
            device=device,
            feature_type_ids=feature_type_ids,
            feature_order=feature_order,
            chunk_budget=chunk_budget,
        )
    except torch.OutOfMemoryError as exc:
        empty_cuda_cache(device)
        raise RuntimeError(
            "Full-batch evaluation exhausted device memory after streamed-feature retry."
        ) from exc
    except RuntimeError as exc:
        if _is_cuda_illegal_memory_access(exc):
            raise RuntimeError(
                "CUDA illegal memory access during full-batch evaluation. "
                + _cuda_illegal_memory_help()
            ) from exc
        raise


def _feature_type_ids(model: EphrisModel, *, feature_dim: int) -> Tensor | None:
    try:
        feature_type_ids = model.feature_preprocessor.feature_metadata(
            feature_dim=int(feature_dim)
        )
    except (RuntimeError, ValueError):
        return None
    return feature_type_ids


def _permute_labels(
    labels: Tensor, class_permutation: Tensor | None, *, num_outputs: int
) -> Tensor:
    if class_permutation is None:
        return labels
    permutation = class_permutation.to(device=labels.device, dtype=torch.long)
    permuted = labels.clone()
    valid = (permuted >= 0) & (permuted < int(num_outputs))
    if bool(valid.any()):
        permuted[valid] = permutation.index_select(
            0, permuted[valid].to(dtype=torch.long)
        )
    return permuted


def _align_logits(logits: Tensor, class_permutation: Tensor | None) -> Tensor:
    if class_permutation is None:
        return logits
    permutation = class_permutation.to(device=logits.device, dtype=torch.long)
    return logits.index_select(1, permutation)


@torch.inference_mode()
def predict_log_probs(
    model: EphrisModel,
    context: GraphContext,
    x_all: Tensor,
    labels: Tensor,
    context_mask: Tensor,
    device: torch.device,
) -> Tensor:
    model.eval()
    permutation = make_inference_permutation(
        x=x_all,
        num_outputs=int(context.num_outputs),
    )
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        logits_all = _infer_logits_with_fallback(
            model=model,
            context=context,
            x_all=x_all,
            labels=_permute_labels(
                labels,
                permutation.class_permutation,
                num_outputs=int(context.num_outputs),
            ),
            context_mask=context_mask,
            device=device,
            chunk_budget=_make_chunk_budget(model, device),
            feature_type_ids=_feature_type_ids(model, feature_dim=int(x_all.size(1))),
            feature_order=permutation.feature_order,
        )
        logits = _align_logits(
            logits_all[:, : int(context.num_outputs)].float(),
            permutation.class_permutation,
        )
        log_probs = F.softmax(logits.detach().cpu(), dim=-1).clamp_min(1e-12).log()
    return log_probs
