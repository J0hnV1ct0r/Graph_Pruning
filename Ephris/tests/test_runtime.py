"""Preparation state, shared ECOC inference, and bounded-workspace behavior."""

import inspect
from unittest.mock import Mock, patch

import pytest
import torch
from torch_geometric.data import Data

from ephris import EphrisClassifier
from ephris.data.context import GraphContext
from ephris.ecoc import ECOCClassifier
from ephris.execution import feature_encoding
from ephris.execution.memory import ChunkBudget
from ephris.execution.readout import project_logits_in_chunks
from ephris.nn.cache import EdgeCache, undirected_edges
from ephris.nn.layers.graph_attention import GraphAttention


@pytest.mark.parametrize(
    "name", ["prepare", "predict_log_proba", "predict_proba", "predict"]
)
def test_public_signatures_are_explicit(name):
    signature = inspect.signature(getattr(EphrisClassifier, name))
    for argument in signature.parameters.values():
        assert argument.kind not in {argument.VAR_POSITIONAL, argument.VAR_KEYWORD}
        if argument.name != "self":
            assert argument.annotation is not inspect.Signature.empty
    assert signature.return_annotation is not inspect.Signature.empty


@pytest.mark.parametrize("failure", ["inputs", "session"])
def test_failed_preparation_invalidates_previous_task(failure, local_checkpoint):
    predictor = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    args = dict(
        x=torch.zeros(4, 2),
        edge_index=[[0, 1], [1, 0]],
        context_indices=[0, 1],
        context_labels=[0, 1],
    )
    with patch("ephris.predictor.InferenceSession") as session:
        session.return_value.predict_log_proba.return_value = torch.zeros(4, 2)
        assert predictor.prepare(**args) is predictor
        assert predictor.predict_prepared_log_proba().shape == (4, 2)
        assert predictor.classes_.tolist() == [0, 1]
        if failure == "session":
            session.return_value.prepare.side_effect = ValueError("invalid task")
            invalid = args
        else:
            invalid = dict(args, edge_index=[[0], [99]])
        with pytest.raises(ValueError):
            predictor.prepare(**invalid)
        assert not hasattr(predictor, "classes_")
        with pytest.raises(RuntimeError, match="prepare"):
            predictor.predict_prepared_log_proba()


def test_api_does_not_advertise_unsupported_edge_attributes(local_checkpoint):
    for name in ("prepare", "predict_proba", "predict_log_proba"):
        assert (
            "edge_attr"
            not in inspect.signature(getattr(EphrisClassifier, name)).parameters
        )
    model = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    with pytest.raises(TypeError, match="edge_attr"):
        model.prepare(
            torch.zeros(4, 2),
            [[0, 1], [1, 0]],
            context_indices=[0, 1],
            context_labels=[0, 1],
            edge_attr=torch.ones(2, 1),
        )


def test_edge_cache_reuses_and_invalidates_in_place_mutations():
    edges = torch.tensor([[0, 1], [1, 2]])
    cache = EdgeCache()
    kwargs = dict(num_nodes=3, device=torch.device("cpu"))
    first = cache.prepare(edges, **kwargs)
    assert cache.prepare(edges, **kwargs) is first
    edges[1, 0] = 2
    second = cache.prepare(edges, **kwargs)
    assert second is not first
    assert torch.equal(second, undirected_edges(edges, num_nodes=3))


def test_chunk_budget_remembers_oom_limit():
    budget = ChunkBudget(True, torch.device("cpu"), 1000, 0.5)
    kwargs = dict(stage="test", shape_key=(3,), max_items=100, bytes_per_item=10)
    assert budget.choose_chunk(**kwargs) == 100
    assert budget.shrink_chunk(stage="test", shape_key=(3,), current_chunk=100) == 50
    assert budget.choose_chunk(**kwargs) == 50


def test_readout_oom_retries_without_skipping_nodes():
    states = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    sizes = []

    def project(chunk):
        sizes.append(len(chunk))
        if len(chunk) > 2:
            raise torch.OutOfMemoryError("test workspace limit")
        return chunk[:, :3]

    model = Mock(node_width=4, max_classes=3, project_logits=project)
    actual = project_logits_in_chunks(
        model,
        states,
        device=torch.device("cpu"),
    )
    assert torch.equal(actual, states[:, :3])
    assert sizes == [8, 4, 2, 2, 2, 2]


def test_ecoc_shares_one_encoder_and_averages_columns():
    graph = Data(
        x=torch.zeros(4, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 16, -1, -1]),
        context_mask=torch.tensor([True, True, False, False]),
    )
    context = GraphContext(graph, graph.context_mask, 17)
    predictions = []

    def predict(encoded_context):
        assert encoded_context.data.x is context.data.x
        assert encoded_context.data.edge_index is context.data.edge_index
        assert encoded_context.data.y[2:].tolist() == [-1, -1]
        scores = torch.arange(40, dtype=torch.float32).reshape(4, 10)
        scores[:, int(encoded_context.data.y[0])] += 1
        log_probs = scores.log_softmax(-1)
        predictions.append(log_probs)
        return log_probs

    model = Mock(predict_log_proba=predict)
    classifier = ECOCClassifier(model, max_classes=10, seed=42)
    rng = torch.get_rng_state().clone()
    classifier.prepare(context)
    assert torch.equal(rng, torch.get_rng_state())
    assert classifier._codebook.shape == (17, 8)
    assert torch.unique(classifier._codebook, dim=0).shape[0] == 17
    actual = classifier.predict_log_proba()
    terms = [
        value.index_select(1, classifier._codebook[:, col])
        for col, value in enumerate(predictions)
    ]
    expected = (sum(terms) / len(terms)).log_softmax(-1)
    assert torch.equal(actual, expected)
    assert actual.shape == (4, 17)  # Includes classes absent from the context.
    assert context.data.y.tolist() == [0, 16, -1, -1]
    first_codebook = classifier._codebook.clone()
    classifier.prepare(context)
    assert torch.equal(first_codebook, classifier._codebook)


def test_graph_attention_retries_smaller_edge_chunks(monkeypatch):
    attention = GraphAttention(width=4, head_width=2, heads=2)
    sizes = []
    expected = torch.zeros(5, 4)

    def forward(**kwargs):
        sizes.append(kwargs["chunk_size"])
        if kwargs["chunk_size"] > 2:
            raise torch.OutOfMemoryError("test edge workspace")
        return expected

    monkeypatch.setattr(attention, "_forward_chunked", forward)
    actual = attention._forward_chunked_with_retry(
        x=expected,
        edge_index=torch.tensor([[0], [1]]),
        chunk_size=8,
        num_real_nodes=3,
        num_globals=2,
    )
    assert actual is expected
    assert sizes == [8, 4, 2]


def test_cached_encoding_oom_falls_back_to_replay(monkeypatch):
    model = Mock(width=4)
    model.feature_tokenizer.cell_encoder.skip_proj.weight = torch.zeros(4, 4)
    budget = ChunkBudget(True, torch.device("cpu"), 1_000_000, 0.85)
    attempted = []
    expected = torch.zeros(8, 4, 4)

    def cached(*args, **kwargs):
        attempted.append(kwargs["chunk_features"])
        raise torch.OutOfMemoryError("test feature workspace")

    replay = Mock(return_value=expected)
    monkeypatch.setattr(feature_encoding, "encode_cached", cached)
    monkeypatch.setattr(feature_encoding, "encode_replayed", replay)
    actual = feature_encoding.encode_cached_or_replay(
        model,
        torch.zeros(8, 4),
        torch.zeros(8, dtype=torch.long),
        torch.ones(8, dtype=torch.bool),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([[0], [1]]),
        chunk_budget=budget,
    )
    assert actual is expected
    assert attempted == [4, 2, 1]
    replay.assert_called_once()
