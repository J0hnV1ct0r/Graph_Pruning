"""Public inference API. Query labels never enter the model."""

from pathlib import Path
from typing import Self

import torch
from huggingface_hub import hf_hub_download
from numpy.typing import ArrayLike
from torch import Tensor

from .checkpoints import (
    RELEASE_COMMIT,
    REPOSITORY,
    load_hub_config,
    verify_checkpoint,
)
from .data.context import context_graph
from .engine import InferenceSession


class EphrisClassifier:
    """Single-pass graph classification using only visible context labels."""

    def __init__(
        self,
        *,
        device: str | torch.device = "auto",
        random_state: int = 42,
        checkpoint: str | Path | None = None,
    ) -> None:
        """Store inference options without downloading or loading weights.

        ``random_state`` is a nonnegative integer controlling the ECOC codebook.
        ``checkpoint=None`` downloads the pinned Hub release on first use.
        An explicit local checkpoint bypasses the Hub.
        """
        self.device = device
        self.random_state = random_state
        self.checkpoint = checkpoint
        self._validate_random_state()
        self._hub_checkpoint: Path | None = None
        self._session: InferenceSession | None = None

    def _validate_random_state(self) -> None:
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer.")

    def _resolve_checkpoint(self) -> Path:
        if self.checkpoint is not None:
            path = Path(self.checkpoint).expanduser().resolve()
        else:
            if self._hub_checkpoint is None:
                config = load_hub_config(
                    hf_hub_download(
                        repo_id=REPOSITORY,
                        filename="config.json",
                        revision=RELEASE_COMMIT,
                    )
                )
                checkpoint = Path(
                    hf_hub_download(
                        repo_id=REPOSITORY,
                        filename=config["file"],
                        revision=RELEASE_COMMIT,
                    )
                )
                verify_checkpoint(checkpoint, config["sha256"])
                self._hub_checkpoint = checkpoint
            path = self._hub_checkpoint
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def prepare(
        self,
        x: Tensor | ArrayLike,
        edge_index: Tensor | ArrayLike,
        *,
        context_indices: Tensor | ArrayLike,
        context_labels: Tensor | ArrayLike,
    ) -> Self:
        """Prepare a transductive task; no gradient updates are performed.

        ``x`` is [N, F] and ``edge_index`` is [2, E]. Context indices and
        labels are integer vectors [K], paired in the supplied order.
        Only labels for the selected nodes are accepted. ``classes_`` contains
        their sorted unique labels; probability columns follow that order.
        All nodes' features remain available. ``random_state`` controls the ECOC codebook.
        A failed preparation invalidates any previously prepared task.
        """
        self._session = None
        self.__dict__.pop("classes_", None)
        self.__dict__.pop("checkpoint_", None)
        self._validate_random_state()
        graph, classes = context_graph(x, edge_index, context_indices, context_labels)
        device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else torch.device(self.device)
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        checkpoint = self._resolve_checkpoint()
        session = InferenceSession(checkpoint)
        session.prepare(
            graph, device=device, num_classes=classes.numel(), seed=self.random_state
        )
        self._session = session
        self.classes_ = classes
        self.checkpoint_ = checkpoint
        return self

    def predict_prepared_log_proba(self) -> Tensor:
        """Return FP32 CPU log probabilities [N, C] in ``classes_`` order."""
        if self._session is None:
            raise RuntimeError("Call prepare() first.")
        return self._session.predict_log_proba()

    def predict_log_proba(
        self,
        x: Tensor | ArrayLike,
        edge_index: Tensor | ArrayLike,
        *,
        context_indices: Tensor | ArrayLike,
        context_labels: Tensor | ArrayLike,
    ) -> Tensor:
        """Prepare a task and return all-node log probabilities; see :meth:`prepare`."""
        self.prepare(
            x,
            edge_index,
            context_indices=context_indices,
            context_labels=context_labels,
        )
        return self.predict_prepared_log_proba()

    def predict_proba(
        self,
        x: Tensor | ArrayLike,
        edge_index: Tensor | ArrayLike,
        *,
        context_indices: Tensor | ArrayLike,
        context_labels: Tensor | ArrayLike,
    ) -> Tensor:
        """Return FP32 CPU probabilities [N, C] in ``classes_`` order."""
        return self.predict_log_proba(
            x,
            edge_index,
            context_indices=context_indices,
            context_labels=context_labels,
        ).exp()

    def predict(
        self,
        x: Tensor | ArrayLike,
        edge_index: Tensor | ArrayLike,
        *,
        context_indices: Tensor | ArrayLike,
        context_labels: Tensor | ArrayLike,
    ) -> Tensor:
        """Return original context-class labels for all nodes as a CPU tensor [N]."""
        log_probabilities = self.predict_log_proba(
            x,
            edge_index,
            context_indices=context_indices,
            context_labels=context_labels,
        )
        return self.classes_[log_probabilities.argmax(dim=1)]
