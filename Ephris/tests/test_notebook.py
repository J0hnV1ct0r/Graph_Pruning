"""Validate the executed node-classification comparison notebook."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return json.loads((ROOT / "examples/inference_demo.ipynb").read_text())


def cell_map():
    return {cell["id"]: cell for cell in notebook()["cells"]}


def function(cell_id, name, scope):
    tree = ast.parse("".join(cell_map()[cell_id]["source"]))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<demo>", "exec"), scope)
    return scope[name]


def test_notebook_is_executed_portable_and_uses_public_api():
    nb = notebook()
    assert set(nb["metadata"]) <= {"kernelspec", "language_info"}
    assert len(cell_map()) == len(nb["cells"])
    source = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "from ephris import EphrisClassifier" in source
    assert "model.predict_proba(" in source
    assert "from torch_geometric.nn import GATConv, GCNConv" in source
    assert "REPEATS" not in source and "comparison_deadline" not in source
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is not None
        assert "execution" not in cell["metadata"]
        assert not cell.get("attachments")
        assert all(o["output_type"] != "error" for o in cell.get("outputs", []))
        assert not re.search(
            r"/(?:home\d*|Users)/", json.dumps(cell.get("outputs", []))
        )


def test_defaults_include_binary_multiclass_and_city_network():
    cells = cell_map()
    config = "".join(cells["configuration"]["source"])
    assert 'DATASETS = ("cornell", "reed98", "cora", "paris")' in config
    assert "GNN_EPOCHS = 200" in config
    assert "CONTEXT_RATIO = 0.5" in config
    data = "".join(cells["data"]["source"])
    assert '"paris": ("Paris", CityNetwork, "paris")' in data
    assert "subgraph" not in data


def test_saved_table_contains_all_measured_comparisons():
    outputs = cell_map()["results"]["outputs"]
    value = next(o["data"]["text/markdown"] for o in outputs if "text/markdown" in o.get("data", {}))
    table = value if isinstance(value, str) else "".join(value)
    rows = {}
    for line in table.splitlines()[2:]:
        columns = [item.strip() for item in line.strip().split("|")[1:-1]]
        rows[columns[0]] = columns
    assert set(rows) == {"Cornell", "Reed98", "Cora", "Paris"}
    assert rows["Reed98"][2] == "ROC-AUC"
    for name in ("Cornell", "Cora", "Paris"):
        assert rows[name][2] == "Accuracy"
    assert rows["Paris"][1] == "114,127"
    for columns in rows.values():
        assert len(columns) == 6
        assert all(0 <= float(item.strip("*")) <= 100 for item in columns[3:])
    plot = cell_map()["performance-plot"]
    assert any("image/png" in output.get("data", {}) for output in plot["outputs"])


def test_unknown_labels_are_excluded_from_context_and_query():
    import torch
    from torch_geometric.utils import (
        add_self_loops,
        coalesce,
        remove_self_loops,
        to_undirected,
    )

    scope = dict(
        torch=torch,
        SEED=42,
        CONTEXT_RATIO=0.5,
        add_self_loops=add_self_loops,
        coalesce=coalesce,
        remove_self_loops=remove_self_loops,
        to_undirected=to_undirected,
    )
    graph = SimpleNamespace(
        x=torch.zeros(9, 3),
        y=torch.tensor([20] * 4 + [90] * 4 + [-1]),
        edge_index=torch.tensor([[0, 8], [8, 0]]),
        num_nodes=9,
    )
    make_task = function("data", "make_task", scope)
    task = make_task("Fake", graph)
    assert len(task["x"]) == 9 and 8 in task["edge_index"]
    assert 8 not in task["context_indices"] and 8 not in task["query_indices"]
    assert len(task["context_indices"]) == len(task["query_indices"]) == 4
    assert task["context_labels"].unique().tolist() == [0, 1]
    assert not set(task["context_indices"].tolist()) & set(task["query_indices"].tolist())
    assert torch.equal(task["query_indices"], make_task("Fake", graph)["query_indices"])


def test_binary_score_uses_ranking_not_class_accuracy():
    import torch
    from sklearn.metrics import accuracy_score, roc_auc_score

    score = function(
        "comparison", "score",
        dict(accuracy_score=accuracy_score, roc_auc_score=roc_auc_score),
    )
    truth = torch.tensor([1, 1, 0, 0])
    positive = torch.tensor([0.49, 0.48, 0.40, 0.30])
    metric, value = score(truth, torch.stack((1 - positive, positive), dim=1))
    assert metric == "ROC-AUC"
    assert value == pytest.approx(1.0)


def test_multiclass_score_uses_class_accuracy():
    import torch
    from sklearn.metrics import accuracy_score, roc_auc_score

    score = function(
        "comparison", "score",
        dict(accuracy_score=accuracy_score, roc_auc_score=roc_auc_score),
    )
    probabilities = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.6, 0.1, 0.3]])
    metric, value = score(torch.tensor([0, 1, 2]), probabilities)
    assert metric == "Accuracy"
    assert value == pytest.approx(2 / 3)


@pytest.mark.parametrize("kind", ["GCN", "GAT"])
def test_baselines_train_on_context_labels_and_predict_all_nodes(kind, monkeypatch):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import GATConv, GCNConv

    scope = dict(
        torch=torch, nn=nn, F=F, GATConv=GATConv, GCNConv=GCNConv,
        SEED=42, GNN_EPOCHS=2, device=torch.device("cpu"),
    )
    exec(compile("".join(cell_map()["gnn-models"]["source"]), "<baselines>", "exec"), scope)
    targets_seen = []
    cross_entropy = F.cross_entropy

    def capture(logits, targets):
        assert logits.shape == (4, 2)
        targets_seen.append(targets.clone())
        return cross_entropy(logits, targets)

    monkeypatch.setattr(F, "cross_entropy", capture)
    nodes = torch.arange(8)
    edges = torch.stack((torch.cat((nodes, nodes)), torch.cat((nodes, nodes.roll(1)))))
    context = torch.tensor([0, 2, 4, 6])
    labels = torch.tensor([0, 1, 0, 1])
    probabilities = scope["fit_gnn"](kind, torch.randn(8, 3), edges, context, labels, 2)
    assert len(targets_seen) == 2
    for targets in targets_seen:
        assert torch.equal(targets, labels)
    assert probabilities.shape == (8, 2) and probabilities.device.type == "cpu"
    assert bool(probabilities.isfinite().all())
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(8))
    with pytest.raises(ValueError, match="One label per context"):
        scope["fit_gnn"](kind, torch.randn(8, 3), edges, context, labels[:1], 2)
