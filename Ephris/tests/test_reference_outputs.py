"""Reference CPU predictions for the pinned inference environment."""

from pathlib import Path

import pytest
import torch

from ephris import EphrisClassifier
from ephris.execution import feature_encoding, runner
from ephris.model import EphrisModel
from ephris.nn import refinement


@pytest.mark.pretrained
@pytest.mark.parametrize("sparse_labels", [False, True])
@pytest.mark.parametrize("classes,route", [(5, "full"), (17, "cached")])
def test_reference_probabilities(classes, route, sparse_labels, monkeypatch):
    """Exercise direct classification and ECOC with feature/node chunking."""
    expected = torch.load(
        Path(__file__).parent / "reference/cpu-probabilities.pt", weights_only=True
    )[f"{classes}-{route}"]
    n = max(48, classes * 4)
    generator = torch.Generator().manual_seed(100 + classes)
    x = torch.randn(n, 13, generator=generator)
    x[:, 0] = 1
    x[0, 1] = float("nan")
    ids = torch.arange(n)
    edges = torch.stack((ids.repeat(3), torch.cat((ids, ids.roll(1), ids.roll(7)))))
    labels = ids % classes
    if sparse_labels:
        labels = labels * 7 + 2
    context = ids < n // 2
    monkeypatch.setattr(
        EphrisModel, "should_stream_feature_tokens", lambda *a, **kw: route != "full"
    )
    monkeypatch.setattr(refinement, "_MAX_RESIDENT_EVAL_NODES", 16)
    monkeypatch.setattr(refinement, "_EVAL_NODE_CHUNK_SIZE", 16)
    monkeypatch.setattr(refinement, "_MAX_RESIDENT_EVAL_FEATURES", 4)
    monkeypatch.setattr(refinement, "_EVAL_FEATURE_CHUNK_SIZE", 4)
    if route == "cached":

        def cached(
            model,
            x,
            labels,
            mask,
            types,
            edges,
            *,
            chunk_budget,
            feature_order=None,
        ):
            return feature_encoding.encode_cached(
                model,
                x,
                labels,
                mask,
                types,
                edges,
                chunk_features=3,
                feature_order=feature_order,
            )

        monkeypatch.setattr(runner, "encode_cached_or_replay", cached)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = EphrisClassifier(random_state=42, device="cpu")
        actual = model.predict_log_proba(
            x,
            edges,
            context_indices=context.nonzero().flatten(),
            context_labels=labels[context],
        )
        assert torch.equal(actual, expected)
        assert model.classes_.tolist() == sorted(set(labels[context].tolist()))
        assert actual.shape[1] == classes
        if sparse_labels:
            predicted = model.predict(
                x,
                edges,
                context_indices=context.nonzero().flatten(),
                context_labels=labels[context],
            )
            assert torch.equal(predicted, model.classes_[expected.argmax(dim=1)])
    finally:
        torch.set_num_threads(previous_threads)
