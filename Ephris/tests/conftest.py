"""Files for tests that validate the API before any weights are loaded."""

import hashlib
import json
from dataclasses import asdict

import pytest

from ephris.config import MEMORY_DEFAULTS, MODEL_DEFAULTS


@pytest.fixture
def local_checkpoint(tmp_path):
    path = tmp_path / "unused-checkpoint.pt"
    path.touch()
    return path


@pytest.fixture
def hub_config(tmp_path, local_checkpoint):
    path = tmp_path / "config.json"
    config = dict(
        file="ephris.pt",
        format_version=2,
        sha256=hashlib.sha256(local_checkpoint.read_bytes()).hexdigest(),
        model={
            "architecture": "ephris",
            "task_type": "multiclass",
            "graph_heads": 8,
            **asdict(MODEL_DEFAULTS),
            **asdict(MEMORY_DEFAULTS),
        },
    )
    path.write_text(json.dumps(config), encoding="utf-8")
    return path
