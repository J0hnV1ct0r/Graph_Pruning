from __future__ import annotations

import torch
from torch import Tensor

from .feature_schema import (
    FEATURE_TYPE_BINARY,
    FEATURE_TYPE_CATEGORICAL,
    FEATURE_TYPE_NUMERIC,
    infer_feature_schema,
)

_BINARY_TRANSFORM_CHUNK_FEATURES = 512


class FeaturePreprocessor:
    """Schema-aware standardization and ordinal encoding without CPU round-trips."""

    def __init__(self) -> None:
        self.feature_dim: int | None = None
        self.feature_type_ids: Tensor | None = None
        self.category_values: list[Tensor] | None = None
        self.numeric_indices: Tensor | None = None
        self.numeric_mean: Tensor | None = None
        self.numeric_scale: Tensor | None = None
        self.batched_binary_indices: Tensor | None = None
        self.batched_binary_categories: Tensor | None = None
        self.scalar_ordinal_indices: tuple[int, ...] = ()

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.dim() != 2:
            raise ValueError(
                f"x must have shape [num_nodes, feature_dim], got {tuple(x.shape)}"
            )
        if int(x.size(-1)) <= 0:
            raise ValueError(
                f"feature dimension must be positive, got {int(x.size(-1))}"
            )
        if not torch.is_floating_point(x):
            raise ValueError(f"x must be floating tensor, got dtype={x.dtype}")

    @staticmethod
    def _encode_ordinal_column(
        values: Tensor, categories: Tensor, *, binary: bool
    ) -> Tensor:
        categories = categories.to(device=values.device, dtype=values.dtype).view(-1)
        if int(categories.numel()) <= 1:
            return values.new_zeros(values.shape)
        ranks = torch.searchsorted(categories, values.contiguous()).clamp(
            max=int(categories.numel()) - 1
        )
        ranks = ranks.to(dtype=values.dtype)
        if bool(binary):
            return ranks.clamp(0, 1)
        return ranks / float(int(categories.numel()) - 1) * 2.0 - 1.0

    def prepare(self, x: Tensor) -> None:
        self._validate_input(x)
        self.feature_dim = int(x.size(-1))
        schema = infer_feature_schema(x)
        self.feature_type_ids = schema["feature_types"].to(dtype=torch.long).cpu()
        self.category_values = [
            values.detach().cpu().to(dtype=torch.float32)
            for values in schema["category_values"]
        ]
        self.numeric_indices = torch.nonzero(
            self.feature_type_ids == FEATURE_TYPE_NUMERIC, as_tuple=False
        ).view(-1)
        batched_binary_indices = [
            col_idx
            for col_idx, feature_type in enumerate(self.feature_type_ids.tolist())
            if int(feature_type) == FEATURE_TYPE_BINARY
            and int(self.category_values[col_idx].numel()) == 2
        ]
        self.batched_binary_indices = torch.as_tensor(
            batched_binary_indices, dtype=torch.long
        )
        self.batched_binary_categories = (
            torch.stack(
                [self.category_values[col_idx] for col_idx in batched_binary_indices],
                dim=0,
            )
            if batched_binary_indices
            else torch.empty((0, 2), dtype=torch.float32)
        )
        batched_binary_set = set(batched_binary_indices)
        self.scalar_ordinal_indices = tuple(
            (
                col_idx
                for col_idx, feature_type in enumerate(self.feature_type_ids.tolist())
                if int(feature_type) != FEATURE_TYPE_NUMERIC
                and col_idx not in batched_binary_set
            )
        )
        if int(self.numeric_indices.numel()) > 0:
            numeric = x.index_select(1, self.numeric_indices.to(device=x.device)).to(
                dtype=torch.float32
            )
            numeric = torch.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
            self.numeric_mean = numeric.mean(dim=0).cpu()
            self.numeric_scale = (
                numeric.std(dim=0, unbiased=False).clamp_min(1e-06).cpu()
            )
        else:
            self.numeric_mean = None
            self.numeric_scale = None

    def transform(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        if (
            self.feature_dim is None
            or self.feature_type_ids is None
            or self.category_values is None
        ):
            raise RuntimeError(
                "Feature preprocessor is not fitted. Call `prepare_features()` first."
            )
        if int(x.size(-1)) != self.feature_dim:
            raise ValueError(
                f"Feature width mismatch for preprocessing: fitted on {self.feature_dim}, got {int(x.size(-1))}"
            )
        out = torch.empty_like(x)
        numeric_indices = self.numeric_indices
        if numeric_indices is not None and int(numeric_indices.numel()) > 0:
            if self.numeric_mean is None or self.numeric_scale is None:
                raise RuntimeError(
                    "Feature preprocessor is missing numeric statistics."
                )
            numeric_idx_device = numeric_indices.to(device=x.device)
            numeric = x.index_select(1, numeric_idx_device)
            numeric = torch.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
            mean = self.numeric_mean.to(device=x.device, dtype=x.dtype)
            scale = self.numeric_scale.to(device=x.device, dtype=x.dtype)
            out[:, numeric_idx_device] = (numeric - mean) / scale
        binary_indices = self.batched_binary_indices
        binary_categories = self.batched_binary_categories
        if (
            binary_indices is not None
            and binary_categories is not None
            and (int(binary_indices.numel()) > 0)
        ):
            for start in range(
                0, int(binary_indices.numel()), _BINARY_TRANSFORM_CHUNK_FEATURES
            ):
                end = min(
                    start + _BINARY_TRANSFORM_CHUNK_FEATURES,
                    int(binary_indices.numel()),
                )
                binary_idx_device = binary_indices[start:end].to(device=x.device)
                binary_values = (
                    x.index_select(1, binary_idx_device).transpose(0, 1).contiguous()
                )
                boundaries = binary_categories[start:end].to(
                    device=x.device, dtype=x.dtype
                )
                binary_ranks = torch.searchsorted(boundaries, binary_values).clamp_(
                    0, 1
                )
                out.index_copy_(
                    1, binary_idx_device, binary_ranks.transpose(0, 1).to(dtype=x.dtype)
                )
        for col_idx in self.scalar_ordinal_indices:
            feature_type = int(self.feature_type_ids[col_idx].item())
            categories = self.category_values[col_idx].to(
                device=x.device, dtype=x.dtype
            )
            values = x[:, int(col_idx)]
            if feature_type == FEATURE_TYPE_BINARY:
                out[:, int(col_idx)] = self._encode_ordinal_column(
                    values, categories, binary=True
                )
            elif feature_type == FEATURE_TYPE_CATEGORICAL:
                out[:, int(col_idx)] = self._encode_ordinal_column(
                    values, categories, binary=False
                )
            else:
                out[:, int(col_idx)] = values
        return out

    def feature_metadata(self, feature_dim: int | None = None) -> Tensor:
        if self.feature_type_ids is None:
            width = (
                int(feature_dim)
                if feature_dim is not None
                else int(self.feature_dim or 0)
            )
            if width <= 0:
                raise RuntimeError("Feature metadata is unavailable before fitting.")
            return torch.full((width,), FEATURE_TYPE_NUMERIC, dtype=torch.long)
        if feature_dim is not None and int(feature_dim) != int(
            self.feature_type_ids.numel()
        ):
            raise ValueError(
                f"Feature metadata width mismatch: fitted on {int(self.feature_type_ids.numel())}, got {int(feature_dim)}"
            )
        return self.feature_type_ids.detach().cpu()
