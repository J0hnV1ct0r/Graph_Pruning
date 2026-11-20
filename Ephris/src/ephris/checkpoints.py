"""Load inference weights whose tensor names match the model directly."""

import hashlib
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import TypedDict

import torch

from .config import MEMORY_DEFAULTS, MODEL_DEFAULTS
from .data.context import GraphContext
from .model import EphrisModel

REPOSITORY = "nums-ai/ephris"
RELEASE_COMMIT = "94aa9caf6cf7fb7c44109e15e0de8a6a4b6b14a0"

_MODEL_FIELDS = (
    {field.name for field in fields(MODEL_DEFAULTS)}
    | {field.name for field in fields(MEMORY_DEFAULTS)}
    | {"graph_heads", "task_type"}
)


class Checkpoint(TypedDict):
    format_version: int
    model: dict[str, str | int | float | bool]
    state_dict: dict[str, torch.Tensor]


class HubConfig(TypedDict):
    file: str
    format_version: int
    sha256: str
    model: dict[str, str | int | float | bool]


def load_hub_config(path: str | Path) -> HubConfig:
    """Read the pinned Hub checkpoint's configuration and integrity metadata."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) != {"file", "format_version", "sha256", "model"}
        or config["file"] != "ephris.pt"
        or config["format_version"] != 2
        or not isinstance(config["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", config["sha256"]) is None
        or not isinstance(config["model"], dict)
    ):
        raise ValueError("Unsupported Hub checkpoint configuration.")
    model_config(config["model"])
    return config


def verify_checkpoint(path: str | Path, expected_sha256: str) -> None:
    """Check the downloaded checkpoint against its pinned configuration."""
    with Path(path).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError("Checkpoint SHA256 does not match the Hub configuration.")


def model_config(metadata: dict[str, str | int | float | bool]) -> dict:
    """Validate the checkpoint model configuration."""
    if metadata.get("architecture") != "ephris":
        raise ValueError("Expected an Ephris checkpoint.")
    if set(metadata) != _MODEL_FIELDS | {"architecture"}:
        raise ValueError(
            "Checkpoint model configuration must match the current schema."
        )
    if not all(type(value) in (str, int, float, bool) for value in metadata.values()):
        raise ValueError(
            "Checkpoint model configuration must contain primitive values."
        )
    return {key: value for key, value in metadata.items() if key != "architecture"}


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> Checkpoint:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format_version", "model", "state_dict"}
        or payload["format_version"] != 2
    ):
        raise ValueError("Unsupported inference checkpoint format; expected version 2.")
    if not isinstance(payload["model"], dict):
        raise ValueError("Checkpoint model configuration must be a dictionary.")
    model_config(payload["model"])
    weights = payload["state_dict"]
    if not isinstance(weights, dict) or not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in weights.items()
    ):
        raise ValueError("Checkpoint weights must map names to tensors.")
    return payload


def build_model_from_checkpoint(
    payload: Checkpoint, context: GraphContext, device: torch.device
) -> EphrisModel:
    model = EphrisModel(
        num_outputs=context.num_outputs, **model_config(payload["model"])
    )
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    return model.to(device).eval()
