"""Adaptive inference chunk budgets and recoverable CUDA-memory cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def empty_cuda_cache(device: torch.device) -> None:
    """Best-effort cleanup after OOM; leave non-CUDA devices untouched."""
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


@dataclass
class ChunkBudget:
    """Runtime CUDA-memory budget for adaptive inference chunking."""

    enabled: bool
    device: torch.device
    target_bytes: int
    memory_fraction: float
    cache: dict[tuple[object, ...], int] = field(default_factory=dict)

    @classmethod
    def for_device(
        cls,
        *,
        device: torch.device,
        enabled: bool,
        memory_fraction: float,
    ) -> "ChunkBudget | None":
        device = torch.device(device)
        if not bool(enabled) or device.type != "cuda":
            return None
        fraction = min(0.95, max(0.05, float(memory_fraction)))
        try:
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except RuntimeError:
            return None
        dynamic_target = int(float(free_bytes) * fraction)
        return cls(
            enabled=True,
            device=device,
            target_bytes=max(1, dynamic_target),
            memory_fraction=fraction,
        )

    def refresh(self) -> None:
        if self.device.type != "cuda":
            return
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except RuntimeError:
            return
        dynamic_target = int(float(free_bytes) * float(self.memory_fraction))
        self.target_bytes = max(1, int(dynamic_target))

    def choose_chunk(
        self,
        *,
        stage: str,
        shape_key: tuple[object, ...],
        max_items: int,
        bytes_per_item: int,
    ) -> int:
        self.refresh()
        max_items = max(1, int(max_items))
        bytes_per_item = max(1, int(bytes_per_item))
        estimated = max(1, min(max_items, self.target_bytes // bytes_per_item))
        key = (str(stage),) + tuple(shape_key)
        cached = self.cache.get(key)
        if cached is not None:
            estimated = min(estimated, max(1, int(cached)))
        return max(1, estimated)

    def shrink_chunk(
        self, *, stage: str, shape_key: tuple[object, ...], current_chunk: int
    ) -> int:
        key = (str(stage),) + tuple(shape_key)
        shrunk = max(1, int(current_chunk) // 2)
        self.cache[key] = shrunk
        return shrunk
