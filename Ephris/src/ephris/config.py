"""Typed model defaults and inference policy; checkpoint values remain authoritative."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefaults:
    max_classes: int = 10
    feature_basis_size: int = 8
    scalar_input_clip: float = 8.0
    num_refinements: int = 4
    feature_latents: int = 128
    width: int = 128
    set_heads: int = 8
    refinement_layers: int = 2
    icl_layers: int = 10
    node_latents: int = 4
    num_globals: int = 8
    scaling_hidden_width: int = 64
    expansion: int = 4


@dataclass(frozen=True)
class MemoryDefaults:
    feature_dynamic_chunking: bool = True
    feature_dynamic_memory_fraction: float = 0.85


MODEL_DEFAULTS = ModelDefaults()
MEMORY_DEFAULTS = MemoryDefaults()
