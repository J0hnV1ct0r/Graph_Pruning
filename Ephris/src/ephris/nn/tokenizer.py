from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..execution.memory import ChunkBudget
from .embeddings import LabelEmbedding, embed_observed_labels
from .layers.weights import Embedding, Linear, RMSNorm, parameter

FeatureTokenChunks = tuple[Tensor, ...]
_EVAL_FEATURE_WORKSPACE_FACTOR = 20


class RobustSetCellEncoder(nn.Module):
    """Embed bounded scalar bases with permutation-equivariant node context."""

    _NUM_FEATURE_TYPES = 3

    def __init__(self, *, basis_size: int, width: int, input_clip: float = 8.0) -> None:
        super().__init__()
        if int(basis_size) <= 0:
            raise ValueError(f"feature_basis_size must be positive, got {basis_size}")
        if int(width) <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if float(input_clip) <= 0.0:
            raise ValueError(f"input_clip must be positive, got {input_clip}")
        self.basis_size = int(basis_size)
        self.width = int(width)
        self.input_clip = float(input_clip)
        self.basis_dim = self.basis_size + 3
        self.register_buffer("rbf_centers", torch.empty(self.basis_size, device="meta"))
        self.register_buffer("rbf_inv_width", torch.empty((), device="meta"))
        self.skip_proj = Linear(self.basis_dim, self.width)
        self.gate_proj = Linear(self.basis_dim, self.width)
        self.up_proj = Linear(self.basis_dim, self.width)
        self.down_proj = Linear(self.width, self.width)
        self.node_context_proj = Linear(
            self._NUM_FEATURE_TYPES * 2 * self.basis_dim, self.width
        )
        self.context_gate = Linear(self.width, 1)
        self.context_scale = parameter()
        self.output_norm = RMSNorm(self.width)

    def _validate_features(self, x: Tensor, *, start: int, end: int) -> tuple[int, int]:
        if x.dim() != 2:
            raise ValueError(
                f"RobustSetCellEncoder expects [num_nodes, feature_dim], got {tuple(x.shape)}"
            )
        feature_dim = int(x.size(1))
        if feature_dim <= 0:
            raise ValueError("feature_width must be positive")
        start = max(0, int(start))
        end = min(feature_dim, int(end))
        if end <= start:
            raise ValueError(f"empty feature range: start={start}, end={end}")
        return (start, end)

    def _basis(self, x: Tensor) -> Tensor:
        work = x.to(device=self.rbf_centers.device, dtype=torch.float32)
        clipped = work.clamp(-self.input_clip, self.input_clip)
        signed_log = clipped.sign() * torch.log1p(clipped.abs())
        distance = (
            clipped.unsqueeze(-1) - self.rbf_centers.float()
        ) * self.rbf_inv_width.float()
        rbf = torch.exp(-0.5 * distance.square())
        return torch.cat(
            (
                clipped.unsqueeze(-1),
                clipped.tanh().unsqueeze(-1),
                signed_log.unsqueeze(-1),
                rbf,
            ),
            dim=-1,
        )

    def _local_tokens(self, x: Tensor) -> Tensor:
        weight = self.skip_proj.weight
        basis = self._basis(x).to(dtype=weight.dtype)
        gated = F.silu(self.gate_proj(basis)) * self.up_proj(basis)
        return self.skip_proj(basis) + self.down_proj(gated)

    def compute_node_context(
        self,
        x: Tensor,
        feature_type_ids: Tensor | None = None,
        *,
        feature_chunk_size: int | None = None,
        feature_order: Tensor | None = None,
    ) -> Tensor:
        _, feature_dim = self._validate_features(x, start=0, end=int(x.size(1)))
        device = self.rbf_centers.device
        if feature_type_ids is None:
            feature_types = torch.zeros(feature_dim, dtype=torch.long, device=device)
        else:
            if int(feature_type_ids.numel()) != feature_dim:
                raise ValueError(
                    f"feature_type_ids must have length {feature_dim}, got {int(feature_type_ids.numel())}"
                )
            feature_types = feature_type_ids.to(device=device, dtype=torch.long).clamp(
                0, self._NUM_FEATURE_TYPES - 1
            )
        if feature_order is not None:
            if int(feature_order.numel()) != feature_dim:
                raise ValueError(
                    f"feature_order must have length {feature_dim}, got {int(feature_order.numel())}"
                )
            order = feature_order.view(-1).to(device=x.device, dtype=torch.long)
            feature_types = feature_types.index_select(0, order.to(device=device))
        else:
            order = None
        num_nodes = int(x.size(0))
        sums = torch.zeros(
            num_nodes,
            self._NUM_FEATURE_TYPES,
            self.basis_dim,
            device=device,
            dtype=torch.float32,
        )
        square_sums = torch.zeros_like(sums)
        counts = torch.zeros(
            self._NUM_FEATURE_TYPES, device=device, dtype=torch.float32
        )
        chunk_size = min(
            feature_dim,
            max(
                1,
                feature_dim if feature_chunk_size is None else int(feature_chunk_size),
            ),
        )
        for chunk_start in range(0, feature_dim, chunk_size):
            chunk_end = min(feature_dim, chunk_start + chunk_size)
            if order is None:
                feature_values = x[:, chunk_start:chunk_end]
            else:
                feature_values = x.index_select(1, order[chunk_start:chunk_end])
            basis = self._basis(feature_values)
            chunk_types = feature_types[chunk_start:chunk_end]
            type_weights = F.one_hot(
                chunk_types, num_classes=self._NUM_FEATURE_TYPES
            ).to(dtype=torch.float32)
            sums += torch.einsum("nfd,ft->ntd", basis, type_weights)
            square_sums += torch.einsum("nfd,ft->ntd", basis.square(), type_weights)
            counts += type_weights.sum(dim=0)
        denom = counts.clamp_min(1.0).view(1, self._NUM_FEATURE_TYPES, 1)
        mean = sums / denom
        rms = torch.sqrt((square_sums / denom).clamp_min(0.0))
        summary = torch.cat((mean, rms), dim=-1).flatten(1)
        weight = self.node_context_proj.weight
        return self.node_context_proj(summary.to(dtype=weight.dtype))

    def encode_features(
        self,
        x: Tensor,
        *,
        start: int,
        end: int,
        node_context: Tensor | None = None,
        feature_type_ids: Tensor | None = None,
        feature_order: Tensor | None = None,
    ) -> Tensor:
        start, end = self._validate_features(x, start=start, end=end)
        if feature_order is None:
            feature_values = x[:, start:end]
        else:
            if int(feature_order.numel()) != int(x.size(1)):
                raise ValueError(
                    f"feature_order must have length {int(x.size(1))}, got {int(feature_order.numel())}"
                )
            order = feature_order.view(-1).to(device=x.device, dtype=torch.long)
            feature_values = x.index_select(1, order[start:end])
        local = self._local_tokens(feature_values)
        if node_context is None:
            node_context = self.compute_node_context(
                x, feature_type_ids, feature_order=feature_order
            )
        context = node_context.to(device=local.device, dtype=local.dtype).unsqueeze(1)
        gate = torch.sigmoid(self.context_gate(local))
        output = local + self.context_scale.to(dtype=local.dtype) * gate * context
        return self.output_norm(output)


class FeatureTokenizer(nn.Module):
    """Create the initial label-aware state for every node feature."""

    def __init__(
        self,
        *,
        max_classes: int,
        width: int,
        basis_size: int,
        input_clip: float = 8.0,
    ):
        super().__init__()
        if int(max_classes) <= 0:
            raise ValueError(f"max_classes must be positive, got {max_classes}")
        self.width = int(width)
        self.cell_encoder = RobustSetCellEncoder(
            basis_size=int(basis_size),
            width=self.width,
            input_clip=float(input_clip),
        )
        self.feature_type_embedding = Embedding(3, self.width)
        self.feature_label_embedding = LabelEmbedding(
            num_classes=int(max_classes), embedding_dim=self.width
        )

    def embed_feature_labels(
        self,
        y: Tensor,
        context_mask: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return embed_observed_labels(
            self.feature_label_embedding, y, context_mask, dtype=dtype, device=device
        )

    def feature_chunk_size(
        self,
        *,
        num_nodes: int,
        feature_dim: int,
        dtype: torch.dtype,
        chunk_budget: ChunkBudget | None,
    ) -> int:
        feature_dim = max(1, int(feature_dim))
        if chunk_budget is None:
            return feature_dim
        bytes_per_feature = self.estimated_feature_workspace_bytes(
            num_nodes=int(num_nodes), num_features=1, dtype=dtype
        )
        return chunk_budget.choose_chunk(
            stage="feature_tokenizer_features",
            shape_key=(int(num_nodes), self.width, str(dtype)),
            max_items=feature_dim,
            bytes_per_item=bytes_per_feature,
        )

    def estimated_feature_workspace_bytes(
        self, *, num_nodes: int, num_features: int, dtype: torch.dtype
    ) -> int:
        """Conservatively estimate resident Feature-ISAB evaluation workspace."""
        dtype_size = torch.empty((), dtype=dtype).element_size()
        return max(
            1,
            int(num_nodes)
            * int(num_features)
            * self.width
            * dtype_size
            * _EVAL_FEATURE_WORKSPACE_FACTOR,
        )

    def prepare_node_context(
        self,
        x: Tensor,
        feature_type_ids: Tensor,
        *,
        chunk_budget: ChunkBudget | None,
        feature_order: Tensor | None = None,
        feature_chunk_size: int | None = None,
    ) -> tuple[Tensor, int]:
        feature_dim = int(x.size(1))
        token_device = self.cell_encoder.skip_proj.weight.device
        token_dtype = self.cell_encoder.skip_proj.weight.dtype
        chunk_features = (
            self.feature_chunk_size(
                num_nodes=int(x.size(0)),
                feature_dim=feature_dim,
                dtype=token_dtype,
                chunk_budget=chunk_budget,
            )
            if feature_chunk_size is None
            else min(feature_dim, max(1, int(feature_chunk_size)))
        )
        budget_shape_key = (int(x.size(0)), self.width, str(token_dtype))
        while True:
            try:
                node_context = self.cell_encoder.compute_node_context(
                    x,
                    feature_type_ids,
                    feature_chunk_size=chunk_features,
                    feature_order=feature_order,
                )
                return (node_context, chunk_features)
            except torch.OutOfMemoryError:
                if chunk_features <= 1:
                    raise
                if token_device.type == "cuda":
                    torch.cuda.empty_cache()
                if chunk_budget is not None:
                    chunk_features = chunk_budget.shrink_chunk(
                        stage="feature_tokenizer_features",
                        shape_key=budget_shape_key,
                        current_chunk=chunk_features,
                    )
                else:
                    chunk_features = max(1, int(chunk_features) // 2)

    def encode_feature_token_chunk(
        self,
        x: Tensor,
        feature_label_embedding: Tensor,
        context_mask: Tensor,
        *,
        start: int,
        end: int,
        feature_type_ids: Tensor | None = None,
        node_context: Tensor | None = None,
        feature_order: Tensor | None = None,
    ) -> Tensor:
        feature_tokens = self.cell_encoder.encode_features(
            x,
            start=start,
            end=end,
            node_context=node_context,
            feature_type_ids=feature_type_ids,
            feature_order=feature_order,
        )
        token_count = int(feature_tokens.size(1))
        if feature_type_ids is None:
            feature_type_chunk = torch.zeros(
                (token_count,), dtype=torch.long, device=feature_tokens.device
            )
        elif feature_order is not None:
            order = feature_order.view(-1).to(
                device=feature_type_ids.device, dtype=torch.long
            )
            feature_type_chunk = feature_type_ids.index_select(0, order[start:end]).to(
                device=feature_tokens.device, dtype=torch.long
            )
        else:
            feature_type_chunk = feature_type_ids[start:end].to(
                device=feature_tokens.device, dtype=torch.long
            )
        feature_type_chunk = feature_type_chunk.clamp(
            0, self.feature_type_embedding.num_embeddings - 1
        )
        metadata = self.feature_type_embedding(feature_type_chunk).to(
            device=feature_tokens.device, dtype=feature_tokens.dtype
        )
        feature_tokens = feature_tokens + metadata.view(1, token_count, self.width)
        observed = context_mask.view(-1).to(
            device=feature_tokens.device, dtype=torch.bool
        )
        if int(observed.numel()) != int(feature_tokens.size(0)):
            raise ValueError(
                f"context_mask must have length {int(feature_tokens.size(0))}, got {int(observed.numel())}"
            )
        if bool(observed.any()):
            context = feature_label_embedding.to(
                device=feature_tokens.device, dtype=feature_tokens.dtype
            )
            feature_tokens = feature_tokens.clone()
            feature_tokens[observed] = feature_tokens[observed] + context[
                observed
            ].unsqueeze(1)
        return feature_tokens

    def iter_feature_token_chunks(
        self,
        x: Tensor,
        y: Tensor,
        context_mask: Tensor,
        *,
        feature_type_ids: Tensor | None = None,
        chunk_budget: ChunkBudget | None = None,
        feature_order: Tensor | None = None,
        node_context: Tensor | None = None,
        feature_chunk_size: int | None = None,
    ):
        if x.dim() != 2:
            raise ValueError(
                f"FeatureTokenizer expects x with shape [num_nodes, feature_dim], got {tuple(x.shape)}"
            )
        feature_dim = int(x.size(1))
        if (
            feature_type_ids is not None
            and int(feature_type_ids.numel()) != feature_dim
        ):
            raise ValueError(
                f"feature_type_ids must have length {feature_dim}, got {int(feature_type_ids.numel())}"
            )
        if feature_order is not None and int(feature_order.numel()) != feature_dim:
            raise ValueError(
                f"feature_order must have length {feature_dim}, got {int(feature_order.numel())}"
            )
        model_weight = self.cell_encoder.skip_proj.weight
        token_device = model_weight.device
        token_dtype = model_weight.dtype
        resolved_feature_type_ids = (
            torch.zeros((feature_dim,), dtype=torch.long)
            if feature_type_ids is None
            else feature_type_ids
        )
        feature_label_embedding = self.embed_feature_labels(
            y, context_mask, dtype=token_dtype, device=token_device
        )
        if node_context is None:
            node_context, chunk_features = self.prepare_node_context(
                x,
                resolved_feature_type_ids,
                chunk_budget=chunk_budget,
                feature_order=feature_order,
                feature_chunk_size=feature_chunk_size,
            )
        else:
            chunk_features = (
                self.feature_chunk_size(
                    num_nodes=int(x.size(0)),
                    feature_dim=feature_dim,
                    dtype=token_dtype,
                    chunk_budget=chunk_budget,
                )
                if feature_chunk_size is None
                else min(feature_dim, max(1, int(feature_chunk_size)))
            )
        budget_shape_key = (int(x.size(0)), self.width, str(token_dtype))
        start = 0
        while start < feature_dim:
            end = min(start + chunk_features, feature_dim)
            try:
                yield self.encode_feature_token_chunk(
                    x,
                    feature_label_embedding,
                    context_mask,
                    start=start,
                    end=end,
                    feature_type_ids=resolved_feature_type_ids,
                    node_context=node_context,
                    feature_order=feature_order,
                )
            except torch.OutOfMemoryError:
                if chunk_features <= 1:
                    raise
                if token_device.type == "cuda":
                    torch.cuda.empty_cache()
                if chunk_budget is not None:
                    chunk_features = chunk_budget.shrink_chunk(
                        stage="feature_tokenizer_features",
                        shape_key=budget_shape_key,
                        current_chunk=chunk_features,
                    )
                else:
                    chunk_features = max(1, int(chunk_features) // 2)
                continue
            start = end

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        context_mask: Tensor,
        *,
        feature_type_ids: Tensor | None = None,
        chunk_budget: ChunkBudget | None = None,
        feature_order: Tensor | None = None,
    ) -> FeatureTokenChunks:
        feature_chunks = tuple(
            self.iter_feature_token_chunks(
                x,
                y,
                context_mask,
                feature_type_ids=feature_type_ids,
                chunk_budget=chunk_budget,
                feature_order=feature_order,
            )
        )
        if not feature_chunks:
            raise ValueError("FeatureTokenizer requires at least one feature")
        return feature_chunks
