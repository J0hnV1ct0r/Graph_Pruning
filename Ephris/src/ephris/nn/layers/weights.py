"""Frozen tensor slots and operations for checkpoint-backed inference.

Meta tensors describe shapes without allocating or sampling weights. The loader
must supply every tensor with strict, assigning state-dict loading before use.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def parameter(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False)


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Use a flattened GEMM for expanded CPU queries and native CUDA projection."""
    if x.device.type == "cpu" and x.ndim > 2 and not x.is_contiguous():
        flat = x.reshape(-1, weight.shape[1])
        return F.linear(flat, weight, bias).reshape(*x.shape[:-1], weight.shape[0])
    return F.linear(x, weight, bias)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = parameter(self.out_features, self.in_features)
        self.bias = parameter(self.out_features) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = parameter(self.num_embeddings, self.embedding_dim)

    def forward(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.weight)


class LayerNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = (int(width),)
        self.eps = eps
        self.weight = parameter(int(width))
        self.bias = parameter(int(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = parameter(int(width))

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.float()
            scale = torch.rsqrt(
                work.square().mean(dim=-1, keepdim=True) + 1.1920928955078125e-7
            )
            weight = self.weight.to(dtype=x.dtype).float()
            return work * scale * weight
