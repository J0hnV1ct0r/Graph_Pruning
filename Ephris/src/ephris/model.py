from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import MEMORY_DEFAULTS, MODEL_DEFAULTS
from .data.preprocessing import FeaturePreprocessor
from .execution.memory import ChunkBudget
from .nn.cache import EdgeCache
from .nn.embeddings import LabelEmbedding, embed_observed_labels
from .nn.graph import GraphBlock
from .nn.layers.weights import LayerNorm, Linear
from .nn.refinement import Compressor, FeatureRefiner, GraphRefiner, RefineBlock
from .nn.tokenizer import FeatureTokenizer

_FEATURE_RESIDENT_BUDGET_FRACTION = 0.8


class EphrisModel(nn.Module):
    """Refine -> compress -> inject context labels -> propagate -> predict.

    d = width; K_N = node_latents; node_width = K_N*d.
    num_refinements includes the final symmetric refinement in compressor.
    """

    def __init__(
        self,
        num_outputs: int,
        graph_heads: int,
        task_type: str,
        refinement_layers: int = MODEL_DEFAULTS.refinement_layers,
        icl_layers: int = MODEL_DEFAULTS.icl_layers,
        max_classes: int = MODEL_DEFAULTS.max_classes,
        feature_basis_size: int = MODEL_DEFAULTS.feature_basis_size,
        scalar_input_clip: float = MODEL_DEFAULTS.scalar_input_clip,
        num_refinements: int = MODEL_DEFAULTS.num_refinements,
        feature_latents: int = MODEL_DEFAULTS.feature_latents,
        width: int = MODEL_DEFAULTS.width,
        set_heads: int = MODEL_DEFAULTS.set_heads,
        node_latents: int = MODEL_DEFAULTS.node_latents,
        num_globals: int = MODEL_DEFAULTS.num_globals,
        scaling_hidden_width: int = MODEL_DEFAULTS.scaling_hidden_width,
        expansion: int = MODEL_DEFAULTS.expansion,
        feature_dynamic_chunking: bool = MEMORY_DEFAULTS.feature_dynamic_chunking,
        feature_dynamic_memory_fraction: float = MEMORY_DEFAULTS.feature_dynamic_memory_fraction,
    ) -> None:
        super().__init__()
        if task_type != "multiclass":
            raise ValueError(
                f"EphrisModel supports only multiclass tasks, got: {task_type}"
            )
        if int(max_classes) <= 0 or int(num_outputs) > int(max_classes):
            raise ValueError("max_classes must be positive and cover num_outputs")
        self.max_classes = int(max_classes)
        self.feature_basis_size = int(feature_basis_size)
        self.scalar_input_clip = float(scalar_input_clip)
        self.num_refinements = int(num_refinements)
        self.feature_latents = int(feature_latents)
        self.width = int(width)
        self.set_heads = int(set_heads)
        self.node_latents = int(node_latents)
        self.graph_heads = int(graph_heads)
        self.refinement_layers = int(refinement_layers)
        self.icl_layers = int(icl_layers)
        self.num_globals = int(num_globals)
        self.scaling_hidden_width = int(scaling_hidden_width)
        self.expansion = int(expansion)
        self.feature_dynamic_chunking = bool(feature_dynamic_chunking)
        self.feature_dynamic_memory_fraction = float(feature_dynamic_memory_fraction)
        self.node_width = self.width * self.node_latents
        self.edge_cache = EdgeCache()
        self._validate_architecture()
        self.feature_preprocessor = FeaturePreprocessor()
        self.feature_tokenizer = FeatureTokenizer(
            max_classes=self.max_classes,
            width=self.width,
            basis_size=self.feature_basis_size,
            input_clip=self.scalar_input_clip,
        )
        common_feature_kwargs = dict(
            width=self.width,
            heads=self.set_heads,
            num_latents=self.feature_latents,
            scaling_hidden_width=self.scaling_hidden_width,
            expansion=self.expansion,
        )
        self.refiners = nn.ModuleList(
            (
                RefineBlock(
                    feature_refiner=FeatureRefiner(**common_feature_kwargs),
                    graph_refiner=GraphRefiner(
                        width=self.width,
                        set_heads=self.set_heads,
                        node_latents=self.node_latents,
                        graph_heads=self.graph_heads,
                        num_layers=self.refinement_layers,
                        num_globals=self.num_globals,
                        scaling_hidden_width=self.scaling_hidden_width,
                        expansion=self.expansion,
                    ),
                )
                for _ in range(self.num_refinements - 1)
            )
        )
        self.compressor = Compressor(
            feature_refiner=FeatureRefiner(**common_feature_kwargs),
            width=self.width,
            set_heads=self.set_heads,
            node_latents=self.node_latents,
            scaling_hidden_width=self.scaling_hidden_width,
            expansion=self.expansion,
        )
        self.icl_label_embedding = LabelEmbedding(
            num_classes=self.max_classes, embedding_dim=self.node_width
        )
        self.label_norm = LayerNorm(self.node_width)
        self.icl = GraphBlock(
            width=self.node_width,
            heads=self.graph_heads,
            num_layers=self.icl_layers,
            num_globals=self.num_globals,
            scaling_hidden_width=self.scaling_hidden_width,
            expansion=self.expansion,
        )
        self.final_norm = LayerNorm(self.node_width)
        self.head = Linear(self.node_width, self.max_classes)

    def _validate_architecture(self) -> None:
        positive = {
            "feature_basis_size": self.feature_basis_size,
            "scalar_input_clip": self.scalar_input_clip,
            "feature_latents": self.feature_latents,
            "width": self.width,
            "set_heads": self.set_heads,
            "node_latents": self.node_latents,
            "graph_heads": self.graph_heads,
            "scaling_hidden_width": self.scaling_hidden_width,
            "expansion": self.expansion,
            "feature_dynamic_memory_fraction": self.feature_dynamic_memory_fraction,
        }
        for name, value in positive.items():
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_globals <= 0:
            raise ValueError(f"num_globals must be positive, got {self.num_globals}")
        if self.num_refinements < 2:
            raise ValueError("num_refinements must be at least 2")
        if self.refinement_layers < 2:
            raise ValueError("refinement_layers must be at least 2")
        if self.icl_layers < 2:
            raise ValueError("icl_layers must be at least 2")
        if self.width % self.set_heads != 0:
            raise ValueError("width must be divisible by set_heads")
        if self.node_width % self.graph_heads != 0:
            raise ValueError("width * node_latents must be divisible by graph_heads")

    def prepare_features(self, x: Tensor) -> None:
        self.feature_preprocessor.prepare(x)

    def transform_features(self, x: Tensor) -> Tensor:
        return self.feature_preprocessor.transform(x)

    def _model_device(self) -> torch.device:
        return next(self.parameters()).device

    def _feature_compute_dtype(self) -> torch.dtype:
        weight = self.feature_tokenizer.cell_encoder.skip_proj.weight
        if weight.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return weight.dtype

    def _resolve_feature_inputs(
        self, x: Tensor, *, x_is_preprocessed: bool, feature_type_ids: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        x_preprocessed = x if x_is_preprocessed else self.transform_features(x)
        if feature_type_ids is not None:
            return (x_preprocessed, feature_type_ids)
        try:
            metadata = self.feature_preprocessor.feature_metadata(
                feature_dim=int(x_preprocessed.size(1))
            )
        except RuntimeError:
            metadata = torch.zeros((int(x_preprocessed.size(1)),), dtype=torch.long)
        return (x_preprocessed, metadata)

    def tokenize_features(
        self,
        x: Tensor,
        y: Tensor,
        context_mask: Tensor,
        *,
        x_is_preprocessed: bool = False,
        feature_type_ids: Tensor | None = None,
        chunk_budget: ChunkBudget | None = None,
        feature_order: Tensor | None = None,
    ) -> Tensor:
        x_preprocessed, feature_type_ids = self._resolve_feature_inputs(
            x, x_is_preprocessed=x_is_preprocessed, feature_type_ids=feature_type_ids
        )
        tokenizer = self.feature_tokenizer
        chunks = tokenizer(
            x_preprocessed,
            y.to(device=self._model_device()),
            context_mask.to(device=self._model_device()),
            feature_type_ids=feature_type_ids,
            chunk_budget=chunk_budget,
            feature_order=feature_order,
        )
        return torch.cat(chunks, dim=1)

    def should_stream_feature_tokens(
        self, x: Tensor, *, chunk_budget: ChunkBudget | None
    ) -> bool:
        if chunk_budget is None:
            return False
        chunk_budget.refresh()
        feature_workspace_bytes = (
            self.feature_tokenizer.estimated_feature_workspace_bytes(
                num_nodes=int(x.size(0)),
                num_features=int(x.size(1)),
                dtype=self._feature_compute_dtype(),
            )
        )
        resident_budget_bytes = int(
            chunk_budget.target_bytes * _FEATURE_RESIDENT_BUDGET_FRACTION
        )
        return feature_workspace_bytes > resident_budget_bytes

    def prepare_streaming_features(
        self,
        x: Tensor,
        *,
        x_is_preprocessed: bool = False,
        feature_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self._resolve_feature_inputs(
            x, x_is_preprocessed=x_is_preprocessed, feature_type_ids=feature_type_ids
        )

    def encode_features(self, feature_state: Tensor, edge_index: Tensor) -> Tensor:
        edge_index = edge_index.to(device=feature_state.device)
        for block in self.refiners:
            feature_state = block(feature_state, edge_index)
        return self.compressor(feature_state)

    def flatten_node_state(self, node_state: Tensor) -> Tensor:
        expected = (self.node_latents, self.width)
        if node_state.dim() != 3 or tuple(node_state.shape[1:]) != expected:
            raise ValueError(
                f"node_state must have shape [num_nodes, node_latents, width], got {tuple(node_state.shape)}"
            )
        return node_state.reshape(int(node_state.size(0)), self.node_width)

    def inject_labels(
        self, node_state: Tensor, y: Tensor, context_mask: Tensor
    ) -> Tensor:
        if node_state.dim() != 2 or int(node_state.size(1)) != self.node_width:
            raise ValueError(
                f"node_state must have shape [num_nodes, {self.node_width}], got {tuple(node_state.shape)}"
            )
        labels = embed_observed_labels(
            self.icl_label_embedding,
            y,
            context_mask,
            dtype=node_state.dtype,
            device=node_state.device,
        )
        return self.label_norm(node_state + labels)

    def propagate(
        self,
        node_state: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        return self.icl(
            node_state,
            edge_index,
        )

    def project_logits(self, node_state: Tensor) -> Tensor:
        if node_state.dim() != 2 or int(node_state.size(1)) != self.node_width:
            raise ValueError(
                f"readout expects [num_nodes, {self.node_width}], got {tuple(node_state.shape)}"
            )
        return self.head(self.final_norm(node_state))
