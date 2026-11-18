from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

FEATURE_TYPE_NUMERIC = 0
FEATURE_TYPE_BINARY = 1
FEATURE_TYPE_CATEGORICAL = 2
_SCHEMA_SCAN_CHUNK_FEATURES = 512


def infer_feature_schema(
    x: Tensor, *, max_cardinality: int = 64, integer_tolerance: float = 1e-06
) -> dict[str, Any]:
    if not isinstance(x, Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if x.dim() != 2:
        raise ValueError(
            f"x must have shape [num_nodes, feature_dim], got {tuple(x.shape)}"
        )
    if int(x.size(1)) <= 0:
        raise ValueError(f"feature dimension must be positive, got {int(x.size(1))}")
    if int(max_cardinality) < 2:
        raise ValueError(f"max_cardinality must be >= 2, got {max_cardinality}")
    if float(integer_tolerance) < 0.0:
        raise ValueError(
            f"integer_tolerance must be non-negative, got {integer_tolerance}"
        )
    x_cpu = torch.nan_to_num(
        x.detach().cpu().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    feature_dim = int(x_cpu.size(1))
    feature_types = torch.full((feature_dim,), FEATURE_TYPE_NUMERIC, dtype=torch.long)
    category_values = [
        torch.empty((0,), dtype=torch.float32) for _ in range(feature_dim)
    ]
    integer_like_columns = torch.zeros((feature_dim,), dtype=torch.bool)
    for start in range(0, feature_dim, _SCHEMA_SCAN_CHUNK_FEATURES):
        end = min(start + _SCHEMA_SCAN_CHUNK_FEATURES, feature_dim)
        values = x_cpu[:, start:end]
        integer_like = torch.all(
            (values - torch.round(values)).abs() <= float(integer_tolerance), dim=0
        )
        integer_like_columns[start:end] = integer_like
        if not bool(integer_like.any()):
            continue
        minimum, maximum = torch.aminmax(values, dim=0)
        binary_candidates = integer_like & minimum.ne(maximum)
        if not bool(binary_candidates.any()):
            continue
        exactly_two_values = torch.all(
            values.eq(minimum.view(1, -1)) | values.eq(maximum.view(1, -1)), dim=0
        )
        binary_local = torch.nonzero(
            binary_candidates & exactly_two_values, as_tuple=False
        ).view(-1)
        for local_idx in binary_local.tolist():
            col_idx = start + int(local_idx)
            feature_types[col_idx] = FEATURE_TYPE_BINARY
            category_values[col_idx] = torch.stack(
                (minimum[local_idx], maximum[local_idx])
            ).to(dtype=torch.float32)
    for col_idx in range(feature_dim):
        if (
            not bool(integer_like_columns[col_idx].item())
            or int(feature_types[col_idx].item()) == FEATURE_TYPE_BINARY
        ):
            continue
        col = x_cpu[:, col_idx].view(-1)
        unique = torch.unique(col, sorted=True)
        if int(unique.numel()) <= int(max_cardinality):
            if int(unique.numel()) >= 3:
                feature_types[col_idx] = FEATURE_TYPE_CATEGORICAL
                category_values[col_idx] = unique.to(dtype=torch.float32)
                continue
    return {
        "feature_types": feature_types,
        "category_values": category_values,
        "max_cardinality": int(max_cardinality),
        "integer_tolerance": float(integer_tolerance),
    }
