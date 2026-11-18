"""Checkpoint decoding, architecture dimensions, and strict weight loading."""

import pytest
import torch
from huggingface_hub import hf_hub_download
from torch_geometric.data import Data

from ephris import EphrisClassifier
from ephris.checkpoints import (
    RELEASE_COMMIT,
    REPOSITORY,
    build_model_from_checkpoint,
    load_checkpoint,
    model_config,
)
from ephris.data.context import GraphContext
from ephris.model import EphrisModel
from ephris.nn.graph import GraphBlock, GraphLayer
from ephris.nn.refinement import Compressor, FeatureRefiner, GraphRefiner


def test_explicit_checkpoint_works_without_hub_access(monkeypatch, local_checkpoint):
    path = local_checkpoint

    def forbidden(**kwargs):
        raise AssertionError("A local checkpoint must not access Hugging Face")

    monkeypatch.setattr("ephris.predictor.hf_hub_download", forbidden)
    model = EphrisClassifier(checkpoint=path, device="cpu")
    assert model.checkpoint == path.resolve()


@pytest.fixture(scope="module")
def checkpoint():
    return load_checkpoint(
        hf_hub_download(
            repo_id=REPOSITORY, filename="ephris.pt", revision=RELEASE_COMMIT
        )
    )


def context():
    mask = torch.tensor([True, True, False, False])
    graph = Data(
        x=torch.zeros(4, 3),
        y=torch.tensor([0, 1, -1, -1]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        context_mask=mask,
    )
    return GraphContext(data=graph, context_mask=mask, num_outputs=2)


@pytest.mark.pretrained
def test_every_weight_loads_directly_without_changing_tensors(checkpoint):
    mapped = checkpoint["state_dict"]
    assert len(mapped) == 689
    model = build_model_from_checkpoint(checkpoint, context(), torch.device("cpu"))
    loaded = model.state_dict()
    assert list(loaded) == list(
        mapped
    )  # Preserve registration order, not just key sets.
    for key, tensor in loaded.items():
        assert tensor.dtype == mapped[key].dtype
        assert torch.equal(tensor, mapped[key])
    # Each parameter must be registered exactly once.
    all_params = list(model.named_parameters(remove_duplicate=False))
    assert len({id(p) for _, p in all_params}) == len(all_params)


@pytest.mark.pretrained
def test_paper_structure_and_dimensions(checkpoint):
    config = model_config(checkpoint["model"])
    assert config["width"] == 128
    assert config["node_latents"] == 4
    assert config["num_globals"] == 8
    model = build_model_from_checkpoint(checkpoint, context(), torch.device("cpu"))
    assert model.node_width == 512 and model.num_refinements == 4
    assert len(model.refiners) == 3
    assert isinstance(model.compressor, Compressor)
    for block in model.refiners:
        assert isinstance(block.feature_refiner, FeatureRefiner)
        assert isinstance(block.graph_refiner, GraphRefiner)
        assert len(block.graph_refiner.mp.layers) == 2
    assert isinstance(model.icl, GraphBlock)
    assert len(model.icl.layers) == 10
    assert all(isinstance(layer, GraphLayer) for layer in model.icl.layers)
    assert model.head.out_features == 10


@pytest.mark.pretrained
def test_checkpoint_loading_never_samples_weights(checkpoint, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint loading must not initialize weights")

    for name in (
        "uniform_",
        "normal_",
        "trunc_normal_",
        "xavier_uniform_",
        "kaiming_uniform_",
        "orthogonal_",
        "zeros_",
        "ones_",
    ):
        monkeypatch.setattr(torch.nn.init, name, forbidden)
    for name in ("rand", "randn", "rand_like", "randn_like"):
        monkeypatch.setattr(torch, name, forbidden)
    graph = context()
    rng = torch.get_rng_state().clone()
    model = build_model_from_checkpoint(checkpoint, graph, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), rng)
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(t.is_meta for t in model.state_dict().values())
    assert not any(hasattr(m, "reset_parameters") for m in model.modules())
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    mapped = checkpoint["state_dict"]
    for key, tensor in model.state_dict().items():
        assert tensor.data_ptr() == mapped[key].data_ptr()


@pytest.mark.pretrained
def test_model_structure_contains_only_shape_slots(checkpoint):
    model = EphrisModel(
        num_outputs=2,
        **model_config(checkpoint["model"]),
    )
    assert len(model.state_dict()) == 689
    assert all(t.is_meta for t in model.state_dict().values())
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("version", [0, 1, 3])
def test_unsupported_checkpoint_version_rejected(tmp_path, version):
    path = tmp_path / "checkpoint.pt"
    torch.save({"format_version": version, "model": {}, "state_dict": {}}, path)
    with pytest.raises(ValueError, match="expected version 2"):
        load_checkpoint(path)


@pytest.mark.parametrize("component", ["width", "node_latents", "scaling_hidden_width"])
@pytest.mark.pretrained
def test_incomplete_configuration_rejected(checkpoint, component):
    metadata = dict(checkpoint["model"])
    del metadata[component]
    with pytest.raises(ValueError, match="current schema"):
        model_config(metadata)


@pytest.mark.pretrained
def test_unknown_configuration_rejected(checkpoint):
    metadata = {**checkpoint["model"], "unknown_setting": True}
    with pytest.raises(ValueError, match="current schema"):
        model_config(metadata)


@pytest.mark.parametrize("change", ["missing", "unexpected", "shape"])
@pytest.mark.pretrained
def test_strict_loading_rejects_incompatible_weights(checkpoint, change):
    weights = dict(checkpoint["state_dict"])
    if change == "missing":
        del weights["head.weight"]
    elif change == "unexpected":
        weights["not_a_model_parameter"] = torch.zeros(1)
    else:
        weights["head.weight"] = torch.zeros(1)
    payload = {**checkpoint, "state_dict": weights}
    with pytest.raises(RuntimeError):
        build_model_from_checkpoint(payload, context(), torch.device("cpu"))
