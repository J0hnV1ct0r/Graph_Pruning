from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ...execution.memory import empty_cuda_cache
from .fused_linear import (
    FusedLinearCache,
    fused_eval_linear,
    fused_linear,
)
from .scaling import QueryScale
from .weights import Linear

_EDGE_WORKSPACE_BUDGET_FRACTION = 0.3


class GraphAttention(nn.Module):
    """Multi-head graph attention over one flattened vector per graph node."""

    def __init__(
        self,
        *,
        width: int,
        head_width: int,
        heads: int = 1,
        bias: bool = True,
        scaling_hidden_width: int = 64,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.head_width = int(head_width)
        self.heads = int(heads)
        if self.width <= 0 or self.head_width <= 0 or self.heads <= 0:
            raise ValueError("width, head_width, and heads must be positive")
        projection_dim = self.heads * self.head_width
        self.q_proj = Linear(self.width, projection_dim, bias=bias)
        self.k_proj = Linear(self.width, projection_dim, bias=bias)
        self.v_proj = Linear(self.width, projection_dim, bias=bias)
        self.scaling = QueryScale(
            self.heads,
            self.head_width,
            hidden_width=int(scaling_hidden_width),
        )
        self._fused_qkv_cache: FusedLinearCache | None = None

    def _project_all_nodes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.device.type == "cuda":
            projected, self._fused_qkv_cache = fused_eval_linear(
                x, (self.q_proj, self.k_proj, self.v_proj), self._fused_qkv_cache
            )
        else:
            projected = fused_linear(x, (self.q_proj, self.k_proj, self.v_proj))
        query, key, value = projected.split(self.heads * self.head_width, dim=-1)
        return tuple(
            (
                item.view(int(x.size(0)), self.heads, self.head_width)
                for item in (query, key, value)
            )
        )

    def _scale_query_by_lengths(self, query: Tensor, lengths: Tensor) -> Tensor:
        return self.scaling(query.unsqueeze(2), lengths).squeeze(2)

    def _automatic_edge_chunk_size(self, *, num_edges: int, x: Tensor) -> int | None:
        """Return a bounded eval chunk size only when full edge messages are unsafe."""
        if x.device.type != "cuda" or int(num_edges) <= 0:
            return None
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(x.device)
        except RuntimeError:
            return None
        dtype_bytes = max(1, x.element_size())
        bytes_per_edge = max(
            1,
            int(self.heads * self.head_width) * dtype_bytes * 8 + int(self.heads) * 16,
        )
        target_bytes = max(1, int(int(free_bytes) * _EDGE_WORKSPACE_BUDGET_FRACTION))
        estimated_bytes = int(num_edges) * bytes_per_edge
        if estimated_bytes <= target_bytes:
            return None
        return max(1, min(int(num_edges), target_bytes // bytes_per_edge))

    @staticmethod
    def _validate_edge_index(edge_index: Tensor, *, num_nodes: int) -> None:
        if edge_index.numel() == 0:
            return
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())
        if min_index < 0 or max_index >= int(num_nodes):
            raise ValueError(
                f"edge_index contains node ids outside the token tensor range: min={min_index}, max={max_index}, num_nodes={int(num_nodes)}"
            )

    @staticmethod
    def _scatter_max(values: Tensor, index: Tensor, *, size: int) -> Tensor:
        result = values.new_full((int(size), int(values.size(1))), -torch.inf)
        expanded_index = index.view(-1, 1).expand_as(values)
        result.scatter_reduce_(
            0, expanded_index, values, reduce="amax", include_self=True
        )
        return result

    @staticmethod
    def _scatter_sum(values: Tensor, index: Tensor, *, size: int) -> Tensor:
        result = values.new_zeros((int(size),) + tuple(values.shape[1:]))
        expanded_index = index.view(-1, *[1] * (values.dim() - 1)).expand_as(values)
        result.scatter_add_(0, expanded_index, values)
        return result

    def _iter_edge_chunks(self, edge_index: Tensor, *, chunk_size: int):
        num_edges = int(edge_index.size(1))
        for start in range(0, num_edges, int(chunk_size)):
            end = min(num_edges, start + int(chunk_size))
            yield (
                edge_index[0, start:end].long(),
                edge_index[1, start:end].long(),
            )

    def _accumulate_factorized_global_edges(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        running_max: Tensor,
        weight_sum: Tensor,
        weighted_value_sum: Tensor,
        num_real_nodes: int,
        num_globals: int,
        chunk_size: int,
    ) -> None:
        """Accumulate complete real/global attention without edge-index scatter."""
        global_start = int(num_real_nodes)
        global_end = global_start + int(num_globals)
        key_global = key[global_start:global_end]
        value_global = value[global_start:global_end]
        scale = math.sqrt(self.head_width)
        real_chunk = max(1, int(chunk_size) // max(1, int(num_globals)))
        for start in range(0, int(num_real_nodes), real_chunk):
            end = min(int(num_real_nodes), start + real_chunk)
            scores = (query[start:end].unsqueeze(1) * key_global.unsqueeze(0)).sum(
                dim=-1
            ).float() / scale
            chunk_max = scores.amax(dim=1)
            previous_max = running_max[start:end]
            next_max = torch.maximum(previous_max, chunk_max)
            previous_scale = torch.where(
                torch.isfinite(previous_max),
                torch.exp(previous_max - next_max),
                torch.zeros_like(previous_max),
            )
            weights = torch.exp(scores - next_max.unsqueeze(1))
            running_max[start:end] = next_max
            weight_sum[start:end] = weight_sum[
                start:end
            ] * previous_scale + weights.sum(dim=1)
            weighted_value_sum[start:end] = weighted_value_sum[
                start:end
            ] * previous_scale.unsqueeze(-1) + (
                weights.unsqueeze(-1) * value_global.unsqueeze(0).float()
            ).sum(dim=1)
        query_global = query[global_start:global_end]
        self_scores = (query_global * key_global).sum(dim=-1).float() / scale
        running_max[global_start:global_end] = self_scores
        weight_sum[global_start:global_end] = torch.ones_like(self_scores)
        weighted_value_sum[global_start:global_end] = value_global.float()
        for start in range(0, int(num_real_nodes), real_chunk):
            end = min(int(num_real_nodes), start + real_chunk)
            key_real = key[start:end]
            value_real = value[start:end]
            scores = (query_global.unsqueeze(1) * key_real.unsqueeze(0)).sum(
                dim=-1
            ).float() / scale
            chunk_max = scores.amax(dim=1)
            previous_max = running_max[global_start:global_end]
            next_max = torch.maximum(previous_max, chunk_max)
            previous_scale = torch.exp(previous_max - next_max)
            weights = torch.exp(scores - next_max.unsqueeze(1))
            running_max[global_start:global_end] = next_max
            weight_sum[global_start:global_end] = weight_sum[
                global_start:global_end
            ] * previous_scale + weights.sum(dim=1)
            weighted_value_sum[global_start:global_end] = weighted_value_sum[
                global_start:global_end
            ] * previous_scale.unsqueeze(-1) + (
                weights.unsqueeze(-1) * value_real.unsqueeze(0).float()
            ).sum(dim=1)

    def _forward_chunked(
        self,
        *,
        x: Tensor,
        edge_index: Tensor,
        chunk_size: int,
        num_real_nodes: int,
        num_globals: int,
    ) -> Tensor:
        """Evaluate graph attention with bounded edge activations.

        The running log-sum-exp reduction is algebraically equivalent to a
        destination-wise attention softmax, while keeping only one edge chunk
        live at a time.
        """
        num_nodes = int(x.size(0))
        if edge_index.numel() > 0 and int(edge_index.max().item()) >= int(
            num_real_nodes
        ):
            raise ValueError(
                "Virtual global attention expects base edges between real nodes only"
            )
        query, key, value = self._project_all_nodes(x)
        degree = torch.bincount(edge_index[1].long(), minlength=num_nodes)
        degree[: int(num_real_nodes)] += int(num_globals)
        degree[int(num_real_nodes) :] += int(num_real_nodes) + 1
        query = self._scale_query_by_lengths(query, degree.clamp_min(1))
        running_max = torch.full(
            (num_nodes, self.heads),
            -torch.inf,
            device=x.device,
            dtype=torch.float32,
        )
        weight_sum = torch.zeros(
            (num_nodes, self.heads), device=x.device, dtype=torch.float32
        )
        weighted_value_sum = torch.zeros(
            (num_nodes, self.heads, self.head_width),
            device=x.device,
            dtype=torch.float32,
        )
        # Destination-wise online softmax; global edges join the same accumulators.
        for src_index, dst_index in self._iter_edge_chunks(
            edge_index, chunk_size=int(chunk_size)
        ):
            target_query = query.index_select(0, dst_index)
            source_key = key.index_select(0, src_index)
            source_value = value.index_select(0, src_index)
            scores = (target_query * source_key).sum(dim=-1).float() / math.sqrt(
                self.head_width
            )
            unique_dst, inverse = torch.unique(
                dst_index, sorted=True, return_inverse=True
            )
            chunk_max = self._scatter_max(scores, inverse, size=int(unique_dst.numel()))
            previous_max = running_max.index_select(0, unique_dst)
            next_max = torch.maximum(previous_max, chunk_max)
            previous_scale = torch.where(
                torch.isfinite(previous_max),
                torch.exp(previous_max - next_max),
                torch.zeros_like(previous_max),
            )
            weights = torch.exp(scores - next_max.index_select(0, inverse))
            chunk_denom = self._scatter_sum(
                weights, inverse, size=int(unique_dst.numel())
            )
            chunk_out = self._scatter_sum(
                source_value.float() * weights.unsqueeze(-1),
                inverse,
                size=int(unique_dst.numel()),
            )
            running_max[unique_dst] = next_max
            weight_sum[unique_dst] = (
                weight_sum.index_select(0, unique_dst) * previous_scale + chunk_denom
            )
            weighted_value_sum[unique_dst] = (
                weighted_value_sum.index_select(0, unique_dst)
                * previous_scale.unsqueeze(-1)
                + chunk_out
            )
        self._accumulate_factorized_global_edges(
            query=query,
            key=key,
            value=value,
            running_max=running_max,
            weight_sum=weight_sum,
            weighted_value_sum=weighted_value_sum,
            num_real_nodes=int(num_real_nodes),
            num_globals=int(num_globals),
            chunk_size=int(chunk_size),
        )
        out = weighted_value_sum / weight_sum.clamp_min(
            torch.finfo(torch.float32).tiny
        ).unsqueeze(-1)
        out = out.to(dtype=x.dtype)
        return out.reshape(num_nodes, self.heads * self.head_width)

    def _forward_chunked_with_retry(
        self,
        *,
        x: Tensor,
        edge_index: Tensor,
        chunk_size: int,
        num_real_nodes: int,
        num_globals: int,
    ) -> Tensor:
        while True:
            try:
                return self._forward_chunked(
                    x=x,
                    edge_index=edge_index,
                    chunk_size=max(1, int(chunk_size)),
                    num_real_nodes=num_real_nodes,
                    num_globals=int(num_globals),
                )
            except torch.OutOfMemoryError:
                if int(chunk_size) <= 1:
                    raise
                next_chunk_size = max(1, int(chunk_size) // 2)
            empty_cuda_cache(x.device)
            chunk_size = next_chunk_size

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_chunk_size: int | None = None,
        num_real_nodes: int,
        num_globals: int,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(
                f"GraphAttention expects [num_nodes, node_width], got {tuple(x.shape)}"
            )
        if int(x.size(-1)) != self.width:
            raise ValueError(f"Expected width={self.width}, got {int(x.size(-1))}")
        if edge_index.dim() != 2 or int(edge_index.size(0)) != 2:
            raise ValueError(
                f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
            )
        num_nodes = int(x.size(0))
        self._validate_edge_index(edge_index, num_nodes=num_nodes)
        if int(num_globals) <= 0:
            raise ValueError(f"num_globals must be positive, got {num_globals}")
        expected_nodes = int(num_real_nodes) + int(num_globals)
        if expected_nodes != num_nodes:
            raise ValueError(
                f"Virtual global-node count must match the token tensor: expected {expected_nodes} nodes, got {num_nodes}"
            )
        total_edges = int(edge_index.size(1)) + (
            2 * int(num_real_nodes) * int(num_globals) + int(num_globals)
        )
        selected_chunk_size = edge_chunk_size
        if selected_chunk_size is None:
            selected_chunk_size = self._automatic_edge_chunk_size(
                num_edges=total_edges, x=x
            )
        if selected_chunk_size is None:
            selected_chunk_size = max(1, min(total_edges, 262144))
        return self._forward_chunked_with_retry(
            x=x,
            edge_index=edge_index,
            chunk_size=max(1, int(selected_chunk_size)),
            num_real_nodes=num_real_nodes,
            num_globals=int(num_globals),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(width={self.width}, head_width={self.head_width}, heads={self.heads})"
