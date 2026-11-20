"""Normalized graph input and labeled context for inference."""

from dataclasses import dataclass

import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch_geometric.data import Data


def context_graph(
    x: Tensor | ArrayLike,
    edge_index: Tensor | ArrayLike,
    context_indices: Tensor | ArrayLike,
    context_labels: Tensor | ArrayLike,
) -> tuple[Data, Tensor]:
    """Validate public inputs and copy only context labels into an all-node graph."""
    x = torch.as_tensor(x)
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("Expected nonempty x[N,F].")
    context_indices = torch.as_tensor(context_indices).detach().cpu()
    context_labels = torch.as_tensor(context_labels).detach().cpu()
    for name, values in (
        ("context_indices", context_indices),
        ("context_labels", context_labels),
    ):
        if values.ndim != 1 or not values.numel():
            raise ValueError(f"{name} must be a nonempty vector [K].")
        if (
            values.dtype == torch.bool
            or values.is_floating_point()
            or values.is_complex()
        ):
            raise ValueError(f"{name} must contain integers, not {values.dtype}.")
    context_indices = context_indices.to(torch.long)
    context_labels = context_labels.to(torch.long)
    if context_labels.numel() != context_indices.numel():
        raise ValueError("context_indices and context_labels must have equal length.")
    if bool(((context_indices < 0) | (context_indices >= x.shape[0])).any()):
        raise ValueError("context_indices must be in [0, N).")
    if context_indices.unique().numel() != context_indices.numel():
        raise ValueError("context_indices must not contain duplicate nodes.")
    classes, encoded_labels = torch.unique(
        context_labels, sorted=True, return_inverse=True
    )
    if classes.numel() < 2:
        raise ValueError("Ephris requires at least two distinct context classes.")
    context_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    context_mask[context_indices] = True
    node_labels = torch.full((x.shape[0],), -1, dtype=torch.long)
    node_labels[context_indices] = encoded_labels
    edge_index = torch.as_tensor(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected edge_index[2,E] with valid node indices.")
    if (
        edge_index.dtype == torch.bool
        or edge_index.is_floating_point()
        or edge_index.is_complex()
    ):
        raise ValueError(f"edge_index must contain integers, not {edge_index.dtype}.")
    edge_index = edge_index.to(torch.long)
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= len(node_labels)
    ):
        raise ValueError("Expected edge_index[2,E] with valid node indices.")
    # GraphContext normalizes on CPU; do not copy inputs to the GPU beforehand.
    return Data(
        x=x,
        edge_index=edge_index,
        y=node_labels,
        context_mask=context_mask,
    ), classes


@dataclass
class GraphContext:
    data: Data
    context_mask: Tensor
    num_outputs: int

    def __post_init__(self) -> None:
        data = self.data.clone()
        data.x = torch.as_tensor(data.x).detach().cpu()
        if data.x.ndim != 2 or data.x.shape[0] == 0 or data.x.shape[1] == 0:
            raise ValueError("Expected nonempty node features [N,F].")
        data.x = torch.nan_to_num(
            data.x.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        data.y = torch.as_tensor(data.y).detach().cpu().view(-1).to(torch.long)
        data.edge_index = (
            torch.as_tensor(data.edge_index, dtype=torch.long).detach().cpu()
        )
        self.context_mask = (
            torch.as_tensor(self.context_mask, dtype=torch.bool).view(-1).cpu()
        )
        if (
            data.y.numel() != data.num_nodes
            or self.context_mask.numel() != data.num_nodes
        ):
            raise ValueError("Labels and context mask must have length N.")
        self.data = data
        self.num_outputs = int(self.num_outputs)
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be positive.")
