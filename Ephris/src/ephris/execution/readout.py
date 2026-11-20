"""Chunked readout and class probabilities with bounded device workspaces."""

import torch
from torch import Tensor

from ..model import EphrisModel
from .memory import ChunkBudget, empty_cuda_cache
from .precision import bf16_autocast


def _is_cuda_illegal_memory_access(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "illegal memory access" in message or "cudaerrorillegaladdress" in message


def _cuda_illegal_memory_help() -> str:
    return (
        "CUDA reports kernel failures asynchronously, so the stack trace may point at "
        "a later synchronization rather than the operation that caused it. Restart "
        "the process and rerun with CUDA_LAUNCH_BLOCKING=1 to locate the failure."
    )


def _readout_chunk_nodes(
    model: EphrisModel,
    chunk_budget: ChunkBudget | None = None,
    dtype: torch.dtype = torch.float32,
    num_nodes: int | None = None,
) -> int:
    if chunk_budget is None:
        return max(1, int(num_nodes)) if num_nodes is not None else 2**31 - 1
    hidden_width = int(model.node_width)
    dtype_size = torch.empty((), dtype=dtype).element_size()
    return chunk_budget.choose_chunk(
        stage="readout",
        shape_key=(hidden_width, str(dtype)),
        max_items=max(1, int(num_nodes)) if num_nodes is not None else 2**31 - 1,
        bytes_per_item=max(1, hidden_width * dtype_size * 4),
    )


@torch.inference_mode()
def project_logits_in_chunks(
    model: EphrisModel,
    node_state: Tensor,
    *,
    device: torch.device,
    chunk_budget: ChunkBudget | None = None,
) -> Tensor:
    num_nodes = int(node_state.size(0))
    if num_nodes == 0:
        return node_state.new_empty((0, model.max_classes), dtype=torch.float32).cpu()
    chunk = _readout_chunk_nodes(
        model,
        chunk_budget=chunk_budget,
        dtype=node_state.dtype,
        num_nodes=num_nodes,
    )
    shape_key = (int(model.node_width), str(node_state.dtype))
    logits_chunks: list[Tensor] = []
    start = 0
    while start < num_nodes:
        end = min(start + chunk, num_nodes)
        logits_chunk = None
        try:
            with bf16_autocast(device):
                logits_chunk = model.project_logits(node_state[start:end].to(device))
            logits_chunks.append(logits_chunk.detach().float().cpu())
            del logits_chunk
            start = end
        except torch.OutOfMemoryError:
            if logits_chunk is not None:
                del logits_chunk
            empty_cuda_cache(device)
            if chunk <= 1:
                raise
            if chunk_budget is not None:
                chunk = chunk_budget.shrink_chunk(
                    stage="readout", shape_key=shape_key, current_chunk=chunk
                )
            else:
                chunk = max(1, int(chunk) // 2)
        except RuntimeError as exc:
            if _is_cuda_illegal_memory_access(exc):
                empty_cuda_cache(device)
                raise RuntimeError(
                    "CUDA illegal memory access while projecting evaluation logits. "
                    + _cuda_illegal_memory_help()
                ) from exc
            raise
    return torch.cat(logits_chunks, dim=0)
