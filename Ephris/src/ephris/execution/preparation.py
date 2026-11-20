"""Frozen model and preprocessed features shared by all class encodings."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..checkpoints import Checkpoint, build_model_from_checkpoint
from ..data.context import GraphContext
from ..model import EphrisModel
from .runner import predict_log_probs


@dataclass(frozen=True)
class PreparedModel:
    model: EphrisModel
    features: Tensor
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Checkpoint, context: GraphContext, device: torch.device
    ) -> "PreparedModel":
        model = build_model_from_checkpoint(checkpoint, context=context, device=device)
        model.prepare_features(context.data.x)
        features = model.transform_features(context.data.x).to(
            device, non_blocking=True
        )
        return cls(model, features, device)

    def predict_log_proba(self, context: GraphContext) -> Tensor:
        return (
            predict_log_probs(
                model=self.model,
                context=context,
                x_all=self.features,
                labels=context.data.y,
                context_mask=context.context_mask,
                device=self.device,
            )
            .detach()
            .cpu()
        )
