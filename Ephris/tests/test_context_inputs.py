"""Context node-label pairs are validated before preparing the model."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from ephris import EphrisClassifier
from ephris.data.context import context_graph


@pytest.mark.parametrize("array", [list, np.asarray, torch.tensor])
def test_unsorted_pairs_preserve_node_identity(array):
    graph, classes = context_graph(
        torch.zeros(5, 2), [[0], [1]], array([4, 1]), array([9, 2])
    )
    assert classes.tolist() == [2, 9]
    assert graph.y.tolist() == [-1, 0, -1, -1, 1]
    assert graph.context_mask.tolist() == [False, True, False, False, True]


@pytest.mark.parametrize(
    "indices,labels,message",
    [
        ([0, 0], [0, 1], "duplicate"),
        ([-1, 1], [0, 1], "context_indices"),
        ([0, 4], [0, 1], "context_indices"),
        ([0.5, 1], [0, 1], "integers"),
        ([True, False], [0, 1], "integers"),
        ([[0, 1]], [0, 1], "vector"),
        ([], [], "nonempty"),
        ([0, 1], [0, 1, 2, 0], "equal length"),
        ([0, 1], [0.5, 1], "integers"),
        ([0, 1], [False, True], "integers"),
        ([0, 1], [[0, 1]], "vector"),
        ([0, 1], [9, 9], "two distinct"),
    ],
)
def test_bad_context_is_rejected_before_session(
    indices, labels, message, local_checkpoint
):
    model = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    with patch("ephris.predictor.InferenceSession") as session:
        with pytest.raises(ValueError, match=message):
            model.prepare(
                torch.zeros(4, 2),
                [[0], [1]],
                context_indices=indices,
                context_labels=labels,
            )
        session.assert_not_called()


@pytest.mark.parametrize(
    "edges",
    [
        [[0.9], [1.9]],
        [[0.0], [1.0]],
        torch.tensor([[True], [False]]),
    ],
)
def test_noninteger_edges_are_rejected_before_session(edges, local_checkpoint):
    model = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    with patch("ephris.predictor.InferenceSession") as session:
        with pytest.raises(ValueError, match="edge_index must contain integers"):
            model.prepare(
                torch.zeros(4, 2),
                edges,
                context_indices=[0, 1],
                context_labels=[0, 1],
            )
        session.assert_not_called()


@pytest.mark.parametrize(
    "method", ["prepare", "predict_proba", "predict_log_proba", "predict"]
)
def test_context_arguments_are_keyword_only(method, local_checkpoint):
    model = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    with pytest.raises(TypeError):
        getattr(model, method)(torch.zeros(4, 2), [[0], [1]], [0, 1], [0, 1], 2)


def test_predict_returns_original_labels_and_replaces_classes(local_checkpoint):
    model = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    with patch("ephris.predictor.InferenceSession") as session:
        session.return_value.predict_log_proba.return_value = torch.tensor(
            [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]
        ).log()
        result = model.predict(
            torch.zeros(3, 2),
            [[0], [1]],
            context_indices=[2, 0, 1],
            context_labels=[9, 2, 5],
        )
        assert model.classes_.tolist() == [2, 5, 9]
        assert result.tolist() == [5, 2, 9]
        graph = session.return_value.prepare.call_args.args[0]
        assert graph.y.tolist() == [0, 1, 2]
        assert session.return_value.prepare.call_args.kwargs["num_classes"] == 3
        model.prepare(
            torch.zeros(3, 2),
            [[0], [1]],
            context_indices=[0, 2],
            context_labels=[-5, 100],
        )
        assert model.classes_.tolist() == [-5, 100]
        with pytest.raises(ValueError, match="two distinct"):
            model.prepare(
                torch.zeros(3, 2),
                [[0], [1]],
                context_indices=[0, 2],
                context_labels=[9, 9],
            )
        assert not hasattr(model, "classes_")
        with pytest.raises(RuntimeError, match="prepare"):
            model.predict_prepared_log_proba()


@pytest.mark.parametrize(
    "method", ["prepare", "predict", "predict_proba", "predict_log_proba"]
)
def test_class_count_cannot_be_supplied(method, local_checkpoint):
    model = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    with pytest.raises(TypeError, match="num_classes"):
        getattr(model, method)(
            torch.zeros(3, 2),
            [[0], [1]],
            context_indices=[0, 1],
            context_labels=[2, 9],
            num_classes=10,
        )
