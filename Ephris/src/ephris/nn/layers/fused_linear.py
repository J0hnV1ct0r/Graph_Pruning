from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .weights import Linear, linear

FusedLinearCache = tuple[tuple[object, ...], Tensor, Tensor | None]


def _tensor_version(tensor: Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        return None


def _linear_modules(linears: Sequence[Linear]) -> tuple[Linear, ...]:
    modules = tuple(linears)
    if not modules:
        raise ValueError("fused linear requires at least one projection")
    if any((module.in_features != modules[0].in_features for module in modules)):
        raise ValueError("fused linear projections must share their input width")
    has_bias = modules[0].bias is not None
    if any(((module.bias is not None) != has_bias for module in modules)):
        raise ValueError("fused linear projections must use a consistent bias setting")
    return modules


def _packed_parameters(modules: tuple[Linear, ...]) -> tuple[Tensor, Tensor | None]:
    weight = torch.cat([module.weight for module in modules], dim=0)
    if modules[0].bias is None:
        return (weight, None)
    bias = torch.cat(
        [module.bias for module in modules if module.bias is not None], dim=0
    )
    return (weight, bias)


def fused_linear(x: Tensor, linears: Sequence[Linear]) -> Tensor:
    """Run same-input projections in one GEMM."""
    weight, bias = _packed_parameters(_linear_modules(linears))
    return linear(x, weight, bias)


def fused_eval_linear(
    x: Tensor, linears: Sequence[Linear], cache: FusedLinearCache | None
) -> tuple[Tensor, FusedLinearCache]:
    """Cache packed projection weights for parameter-frozen inference."""
    modules = _linear_modules(linears)
    signature: tuple[object, ...] = tuple(
        (
            value
            for module in modules
            for value in (
                module.weight.device,
                module.weight.dtype,
                _tensor_version(module.weight),
                None if module.bias is None else _tensor_version(module.bias),
            )
        )
    )
    if cache is None or cache[0] != signature:
        weight, bias = _packed_parameters(modules)
        cache = (signature, weight, bias)
    return (linear(x, cache[1], cache[2]), cache)
