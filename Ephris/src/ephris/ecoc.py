"""Error-correcting output codes for tasks wider than the checkpoint's head."""

import math
from copy import copy

import torch
from torch import Tensor
from torch.nn import functional as F

from .data.context import GraphContext
from .execution.preparation import PreparedModel

_ECOC_REDUNDANCY = 4


class ECOCClassifier:
    """Share one model across code columns and average their log probabilities."""

    def __init__(
        self,
        model: PreparedModel,
        *,
        max_classes: int,
        seed: int,
    ) -> None:
        if max_classes < 2:
            raise ValueError("max_classes must be at least two.")
        self.model = model
        self.max_classes = max_classes
        self.seed = seed
        self._codebook: Tensor | None = None
        self._contexts: list[GraphContext] = []

    def prepare(self, context: GraphContext) -> None:
        self._codebook = None
        self._contexts = []
        if context.num_outputs <= self.max_classes:
            raise ValueError(
                "Use direct inference when classes fit the checkpoint head."
            )
        labels = context.data.y.view(-1).long().cpu()
        visible = context.context_mask.cpu() & (labels >= 0)
        class_ids = labels[visible]
        if class_ids.numel() == 0 or bool((class_ids >= context.num_outputs).any()):
            raise ValueError("ECOC requires visible labels in [0, num_classes).")
        codebook = self._generate_codebook(context.num_outputs)
        # Gather once for all columns, not once per encoded classifier.
        context_codes = codebook.index_select(0, class_ids)
        encoded_contexts = []
        for column in range(codebook.size(1)):
            encoded_labels = labels.new_full(labels.shape, -1)
            encoded_labels[visible] = context_codes[:, column]
            encoded_context = copy(context)
            encoded_context.data = copy(context.data)
            encoded_context.data.y = encoded_labels
            encoded_context.context_mask = visible
            encoded_context.num_outputs = self.max_classes
            encoded_contexts.append(encoded_context)
        self._codebook = codebook
        self._contexts = encoded_contexts

    @torch.inference_mode()
    def predict_log_proba(self) -> Tensor:
        if self._codebook is None or not self._contexts:
            raise RuntimeError("Call prepare() before prediction.")
        class_scores: Tensor | None = None
        for column, context in enumerate(self._contexts):
            encoded_log_probs = self.model.predict_log_proba(context)
            class_codes = self._codebook[:, column].to(encoded_log_probs.device)
            column_scores = encoded_log_probs.index_select(1, class_codes)
            class_scores = (
                column_scores if class_scores is None else class_scores + column_scores
            )
        assert class_scores is not None
        mean_scores = class_scores / len(self._contexts)
        return F.log_softmax(mean_scores, dim=-1)

    def _generate_codebook(self, num_classes: int) -> Tensor:
        num_columns = max(
            1,
            math.ceil(math.log(num_classes, self.max_classes)) * _ECOC_REDUNDANCY,
        )
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        codes: list[Tensor] = []
        seen: set[tuple[int, ...]] = set()
        max_attempts = max(1000, num_classes * 1000)
        for _ in range(max_attempts):
            code = torch.randint(
                0,
                self.max_classes,
                (num_columns,),
                generator=generator,
                dtype=torch.long,
            )
            key = tuple(int(value) for value in code.tolist())
            if key in seen:
                continue
            seen.add(key)
            codes.append(code)
            if len(codes) == num_classes:
                return torch.stack(codes)
        raise RuntimeError(f"Could not generate {num_classes} unique ECOC codes.")
