"""Undirected edges with self loops, cached without changing node order."""

import torch
from torch import Tensor
from torch_geometric.utils import add_remaining_self_loops, to_undirected


def undirected_edges(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


def _tensor_signature(tensor: Tensor) -> tuple[object, ...]:
    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors have no version counter.
        version = None
    return tuple(tensor.shape), tensor.dtype, tensor.device, tensor.data_ptr(), version


class EdgeCache:
    """Reuse preprocessing and device copies until source tensors or device change."""

    def __init__(self) -> None:
        self._source_key: tuple[object, ...] | None = None
        self._source_edges: Tensor | None = None
        self._device_key: tuple[object, ...] | None = None
        self._device_edges: Tensor | None = None

    def prepare(
        self,
        edge_index: Tensor,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> Tensor:
        source_key = (
            num_nodes,
            _tensor_signature(edge_index),
        )
        if self._source_edges is None or source_key != self._source_key:
            self._source_edges = undirected_edges(edge_index, num_nodes=num_nodes)
            self._source_key = source_key
            self._device_key = None
        edge_index = self._source_edges
        device_key = (
            _tensor_signature(edge_index),
            device,
        )
        if self._device_edges is None or device_key != self._device_key:
            self._device_edges = edge_index.to(device=device, non_blocking=True)
            self._device_key = device_key
        return self._device_edges
