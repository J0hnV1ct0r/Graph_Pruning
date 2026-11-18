"""Task preparation and dispatch to direct or ECOC inference."""

from copy import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data

from .checkpoints import load_checkpoint
from .data.context import GraphContext
from .ecoc import ECOCClassifier
from .execution.preparation import PreparedModel


@dataclass(frozen=True)
class PreparedTask:
    context: GraphContext
    model: PreparedModel
    ecoc: ECOCClassifier | None


class InferenceSession:
    """Own one prepared task; weights and features are shared across ECOC columns."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self._task: PreparedTask | None = None

    def prepare(
        self, graph: Data, *, device: torch.device, num_classes: int, seed: int
    ) -> None:
        self._task = None
        _validate_context(graph, num_classes)
        checkpoint = load_checkpoint(self.checkpoint)
        context = GraphContext(
            data=graph, context_mask=graph.context_mask, num_outputs=num_classes
        )
        max_classes = int(checkpoint["model"]["max_classes"])
        # The head width stays fixed; only labels change between ECOC columns.
        model_context = copy(context)
        model_context.num_outputs = min(num_classes, max_classes)
        model = PreparedModel.from_checkpoint(checkpoint, model_context, device)
        ecoc = None
        if num_classes > max_classes:
            ecoc = ECOCClassifier(model, max_classes=max_classes, seed=seed)
            ecoc.prepare(context)
        self._task = PreparedTask(context, model, ecoc)

    @torch.no_grad()
    def predict_log_proba(self) -> Tensor:
        task = self._task
        if task is None:
            raise RuntimeError("Call prepare() before prediction.")
        if task.ecoc is not None:
            return task.ecoc.predict_log_proba()
        return task.model.predict_log_proba(task.context)


def _validate_context(graph: Data, num_classes: int) -> None:
    labels = graph.y.view(-1).long()
    context_mask = graph.context_mask.view(-1).bool()
    if bool((labels[~context_mask] >= 0).any()):
        raise ValueError(
            "Labels outside the context must be hidden before preparation."
        )
    visible_labels = torch.unique(labels[context_mask & (labels >= 0)], sorted=True)
    if visible_labels.numel() < 2:
        raise ValueError("Ephris requires at least two visible classes.")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least two, got {num_classes}.")
    if int(visible_labels[-1]) >= num_classes:
        raise ValueError("Visible labels must be smaller than num_classes.")
