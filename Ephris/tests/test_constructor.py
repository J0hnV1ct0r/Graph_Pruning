"""Constructor options and lazy checkpoint resolution."""

import json
from unittest.mock import call, patch

import pytest
import torch

from ephris import EphrisClassifier
from ephris.checkpoints import RELEASE_COMMIT, REPOSITORY

INPUTS = dict(
    x=torch.zeros(3, 2),
    edge_index=[[0], [1]],
    context_indices=[0, 1],
    context_labels=[2, 9],
)


def test_construction_does_not_access_hub_files_or_cuda():
    with (
        patch("ephris.predictor.hf_hub_download") as download,
        patch("ephris.predictor.Path.is_file") as is_file,
        patch("ephris.predictor.torch.cuda.is_available") as cuda,
        patch("ephris.predictor.InferenceSession") as session,
    ):
        model = EphrisClassifier(random_state=7)
        EphrisClassifier(checkpoint="missing.pt")
        assert model.random_state == 7 and model.device == "auto"
        assert model.checkpoint is None
        for operation in (download, is_file, cuda, session):
            operation.assert_not_called()


def test_first_prediction_resolves_once_and_forwards_options(
    local_checkpoint, hub_config
):
    with (
        patch(
            "ephris.predictor.hf_hub_download",
            side_effect=[str(hub_config), str(local_checkpoint)],
        ) as download,
        patch("ephris.predictor.InferenceSession") as session,
    ):
        session.return_value.predict_log_proba.return_value = torch.tensor(
            [[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]]
        ).log()
        model = EphrisClassifier(device="cpu", random_state=123)
        assert model.predict(**INPUTS).tolist() == [2, 9, 9]
        model.predict_proba(**INPUTS)
        assert download.call_args_list == [
            call(repo_id=REPOSITORY, filename=filename, revision=RELEASE_COMMIT)
            for filename in ("config.json", "ephris.pt")
        ]
        assert model.checkpoint is None
        assert model.checkpoint_ == local_checkpoint
        assert session.return_value.prepare.call_args.kwargs == dict(
            device=torch.device("cpu"),
            num_classes=2,
            seed=123,
        )


def test_local_checkpoint_bypasses_hub_and_missing_file_fails_on_use(local_checkpoint):
    with patch("ephris.predictor.hf_hub_download") as download:
        model = EphrisClassifier(device="cpu", checkpoint=local_checkpoint)
        with patch("ephris.predictor.InferenceSession") as session:
            model.prepare(**INPUTS)
            session.assert_called_once_with(local_checkpoint.resolve())
        model.checkpoint = local_checkpoint.parent / "missing.pt"
        with pytest.raises(FileNotFoundError):
            model.prepare(**INPUTS)
        assert not hasattr(model, "classes_")
        assert not hasattr(model, "checkpoint_")
        download.assert_not_called()


def test_download_failure_can_be_retried(local_checkpoint, hub_config):
    model = EphrisClassifier(device="cpu")
    with (
        patch(
            "ephris.predictor.hf_hub_download",
            side_effect=[OSError("offline"), str(hub_config), str(local_checkpoint)],
        ) as download,
        patch("ephris.predictor.InferenceSession"),
    ):
        with pytest.raises(OSError, match="offline"):
            model.prepare(**INPUTS)
        assert not hasattr(model, "classes_")
        model.prepare(**INPUTS)
        assert download.call_count == 3
        assert model.classes_.tolist() == [2, 9]


def test_corrupted_checkpoint_is_rejected_and_can_be_retried(
    local_checkpoint, hub_config
):
    original = local_checkpoint.read_bytes()
    local_checkpoint.write_bytes(b"corrupted")
    model = EphrisClassifier(device="cpu")
    with (
        patch(
            "ephris.predictor.hf_hub_download",
            side_effect=[str(hub_config), str(local_checkpoint)] * 2,
        ),
        patch("ephris.predictor.InferenceSession") as session,
    ):
        with pytest.raises(ValueError, match="SHA256"):
            model.prepare(**INPUTS)
        session.assert_not_called()
        assert not hasattr(model, "classes_")
        local_checkpoint.write_bytes(original)
        model.prepare(**INPUTS)
        assert model.classes_.tolist() == [2, 9]


@pytest.mark.parametrize(
    "change",
    [
        {"file": "../ephris.pt"},
        {"format_version": 3},
        {"sha256": "invalid"},
        {"model": {}},
    ],
)
def test_invalid_hub_configuration_does_not_download_weights(hub_config, change):
    config = json.loads(hub_config.read_text())
    hub_config.write_text(json.dumps({**config, **change}))
    with patch(
        "ephris.predictor.hf_hub_download", return_value=str(hub_config)
    ) as download:
        with pytest.raises(ValueError):
            EphrisClassifier(device="cpu").prepare(**INPUTS)
        download.assert_called_once_with(
            repo_id=REPOSITORY, filename="config.json", revision=RELEASE_COMMIT
        )


@pytest.mark.parametrize("value", [None, -1, True, 1.5, "42"])
def test_random_state_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="random_state"):
        EphrisClassifier(random_state=value)


def test_invalid_inputs_do_not_download():
    with patch("ephris.predictor.hf_hub_download") as download:
        with pytest.raises(ValueError, match="duplicate"):
            EphrisClassifier().prepare(**dict(INPUTS, context_indices=[0, 0]))
        download.assert_not_called()
