from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def configure_cuda_matmul_precision(device: torch.device) -> None:
    """Allow CUDA FP32 matmuls to use Tensor Core-friendly implementations."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def bf16_autocast(device: torch.device) -> AbstractContextManager[None]:
    """Use BF16 autocast on a CUDA device that supports it."""
    if not bf16_autocast_enabled(device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def bf16_autocast_enabled(device: torch.device) -> bool:
    """Check the target GPU, which need not be the current CUDA device.

    CPU inference keeps FP32; CUDA BF16 is the validated inference policy.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    if device.index is None:
        return torch.cuda.is_bf16_supported()
    with torch.cuda.device(device):
        return torch.cuda.is_bf16_supported()
