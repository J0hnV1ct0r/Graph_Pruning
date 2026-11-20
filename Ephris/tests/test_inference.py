import ast
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from huggingface_hub import hf_hub_download

from ephris import EphrisClassifier
from ephris.checkpoints import RELEASE_COMMIT, REPOSITORY, load_checkpoint
from ephris.execution.precision import bf16_autocast_enabled

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.pretrained
def test_checkpoint_is_weights_only_and_clean():
    path = Path(
        hf_hub_download(
            repo_id=REPOSITORY, filename="ephris.pt", revision=RELEASE_COMMIT
        )
    )
    payload = load_checkpoint(path)
    assert payload["format_version"] == 2
    assert len(payload["state_dict"]) == 689
    assert set(payload) == {"format_version", "model", "state_dict"}
    assert all(
        isinstance(value, (str, int, float, bool))
        for value in payload["model"].values()
    )
    manifest_path = hf_hub_download(
        repo_id=REPOSITORY, filename="manifest.json", revision=RELEASE_COMMIT
    )
    manifest = json.loads(Path(manifest_path).read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]


def test_query_labels_never_enter_runtime(local_checkpoint):
    predictor = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    captured = []
    with patch("ephris.predictor.InferenceSession") as runtime:
        runtime.return_value.prepare.side_effect = lambda graph, **kwargs: (
            captured.append(graph.y.clone())
        )
        for indices, labels in (([0, 1], [0, 1]), ([1, 0], [1, 0])):
            predictor.prepare(
                torch.randn(4, 2),
                torch.tensor([[0, 1], [1, 0]]),
                context_indices=indices,
                context_labels=labels,
            )
    assert torch.equal(captured[0], captured[1])
    assert captured[0].tolist() == [0, 1, -1, -1]


def test_invalid_edges_rejected(local_checkpoint):
    predictor = EphrisClassifier(checkpoint=local_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="edge_index"):
        predictor.prepare(
            torch.zeros(2, 2),
            [[0], [3]],
            context_indices=[0, 1],
            context_labels=[0, 1],
        )


def test_cpu_fallback():
    assert not bf16_autocast_enabled(torch.device("cpu"))


@pytest.mark.pretrained
def test_cpu_prediction_preserves_context_pair_order_end_to_end():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        predictor = EphrisClassifier(random_state=42, device="cpu")
        x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
        edges = torch.stack([torch.arange(8), torch.roll(torch.arange(8), 1)])
        first = predictor.predict_proba(
            x,
            edges,
            context_indices=[0, 1, 2, 3],
            context_labels=[0, 1, 0, 1],
        )
        second = predictor.predict_proba(
            x,
            edges,
            context_indices=[3, 1, 0, 2],
            context_labels=[90, 90, 20, 20],
        )
        assert torch.equal(first, second)
        assert predictor.classes_.tolist() == [20, 90]
        assert first.dtype == torch.float32
        assert bool(first.isfinite().all())
        torch.testing.assert_close(first.sum(-1), torch.ones(8))
    finally:
        torch.set_num_threads(previous_threads)


def test_target_gpu_is_checked():
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "device") as device,
        patch.object(torch.cuda, "is_bf16_supported", return_value=True),
    ):
        assert bf16_autocast_enabled(torch.device("cuda:3"))
        device.assert_called_once_with(torch.device("cuda:3"))


@pytest.mark.parametrize("classes", [5, 17])
@pytest.mark.pretrained
def test_multiclass_and_ecoc_probability_shape(classes):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        n = 4 * classes
        x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10
        edges = torch.stack([torch.arange(n), torch.roll(torch.arange(n), 1)])
        mask = torch.arange(n) < 2 * classes
        labels = torch.arange(2 * classes) % classes
        model = EphrisClassifier(random_state=42, device="cpu")
        model.prepare(
            x,
            edges,
            context_indices=mask.nonzero().flatten(),
            context_labels=labels,
        )
        log_probs = model.predict_prepared_log_proba()
        assert log_probs.shape == (n, classes)
        assert bool(log_probs.isfinite().all())
        torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(n))
    finally:
        torch.set_num_threads(previous_threads)


def test_only_declared_import_roots():
    allowed = {
        "__future__",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "ephris",
        "gc",
        "hashlib",
        "huggingface_hub",
        "json",
        "logging",
        "math",
        "numpy",
        "pathlib",
        "random",
        "re",
        "torch",
        "torch_geometric",
        "types",
        "typing",
    }
    for path in (ROOT / "src/ephris").rglob("*.py"):
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                assert all(name.name.split(".")[0] in allowed for name in node.names)
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed
