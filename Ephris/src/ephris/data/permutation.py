"""One deterministic full-feature/class permutation for Ephris inference."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class InferencePermutation:
    feature_order: Tensor | None
    class_permutation: Tensor | None


def make_inference_permutation(
    *,
    x: Tensor,
    num_outputs: int,
) -> InferencePermutation:
    if x.dim() != 2:
        raise ValueError(
            f"x must have shape [num_nodes, num_features], got {tuple(x.shape)}"
        )
    num_classes = int(num_outputs)
    if num_classes < 2:
        raise ValueError(f"num_outputs must be at least two, got {num_outputs}")
    rng = np.random.default_rng(0)
    # Keep the checkpoint's inference RNG sequence, including the initial draw.
    start = int(rng.integers(0, 1000))
    rng.choice(np.arange(start, start + 1, dtype=np.int64), size=1, replace=False)
    candidates = np.unique(np.argsort(rng.random((3, num_classes)), axis=1), axis=0)
    classes = (
        candidates[0]
        if len(candidates) == 1
        else candidates[int(rng.choice(len(candidates), size=1)[0])]
    )
    order = np.arange(int(x.size(1)), dtype=np.int64)
    order = rng.permutation(order)
    return InferencePermutation(
        feature_order=None
        if np.array_equal(order, np.arange(x.size(1)))
        else torch.as_tensor(order, dtype=torch.long),
        class_permutation=torch.as_tensor(classes, dtype=torch.long),
    )
