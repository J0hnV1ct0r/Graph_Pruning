"""
Avaliação ICL do NodePFN no protocolo do projeto (mesmos dados e splits do GraphPFN e do G2T-FM).

Repete o pipeline de nodepfn/node_classification.py (arestas não direcionadas + self-loops, gcn_norm,
suavização SimpleConv, TSVD, contexto = treino, consulta = validação + teste), trocando só a origem dos
dados e dos splits:
  - tolokers-2: GraphLand, split oficial RL (o mesmo do GraphPFN);
  - amazon-ratings, wiki-cs: PyG + G2T-FM/data/<ds>/split.npz (10/10/80, semente 0, igual ao GraphPFN).
Hiperparâmetros de inferência por dataset vêm de run.sh (tolokers-2 usa os do tolokers).
Métrica do artigo: AP (binária) e acurácia (multiclasse). Referência: Tabela 11 do artigo do GraphPFN.

Uso (da raiz do repositório, ~/venvs/nodepfn): python NodePFN/projeto/avaliar_projeto.py --dataset tolokers-2 --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AQUI = Path(__file__).resolve().parents[1]  # NodePFN/
sys.path.insert(0, str(AQUI / "nodepfn"))

from node_classification import fix_seed, update_edge_index  # noqa: E402
from scripts.transformer_prediction_interface import NodePFNClassifier  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score  # noqa: E402
from torch_geometric.nn import SimpleConv  # noqa: E402
from torch_geometric.nn.conv.gcn_conv import gcn_norm  # noqa: E402
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected  # noqa: E402

GRAPHLAND = Path.home() / ".cache" / "graphland"
PYG = Path.home() / ".cache" / "pyg_data"
SPLITS = AQUI.parent / "G2T-FM" / "data"
RESULTS = AQUI.parent / "Block_Influence_results" / "nodepfn_fase1"

# Hiperparâmetros de run.sh (n_components None = sem redução de dimensão)
HP = {
    "tolokers-2": dict(smoothing_steps=2, n_components=None, n_ensemble=4, svd_algorithm="randomized"),
    "amazon-ratings": dict(smoothing_steps=3, n_components=20, n_ensemble=8, svd_algorithm="randomized"),
    "wiki-cs": dict(smoothing_steps=2, n_components=15, n_ensemble=32, svd_algorithm="arpack"),
    # Transferência (2026-11): questions com os HPs de run.sh; artnet-exp não está no artigo do NodePFN e usa os do
    # tolokers-2, o outro dataset binário do GraphLand.
    "questions": dict(smoothing_steps=3, n_components=25, n_ensemble=1, svd_algorithm="randomized"),
    "artnet-exp": dict(smoothing_steps=2, n_components=None, n_ensemble=4, svd_algorithm="randomized"),
}
REFERENCIA = {"tolokers-2": 0.4379, "amazon-ratings": 0.4122, "wiki-cs": 0.7810}  # Tabela 11 do GraphPFN


def carregar(name):
    if name in ("tolokers-2", "artnet-exp"):
        d = GRAPHLAND / name / name / "raw" / name
        x = pd.read_csv(d / "features.csv", index_col=0).astype(np.float32).fillna(0.0).values
        y = pd.read_csv(d / "targets.csv", index_col=0).values.squeeze().astype(np.int64)
        e = pd.read_csv(d / "edgelist.csv").values[:, :2].T
        m = pd.read_csv(d / "split_masks_RL.csv", index_col=0)
        masks = {k: m[k].values.astype(bool) for k in ["train", "val", "test"]}
    else:
        from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
        g = (HeterophilousGraphDataset(PYG, name) if name in ("amazon-ratings", "questions")
             else WikiCS(PYG / "wiki-cs", is_undirected=True))[0]
        x, y, e = g.x.numpy().astype(np.float32), g.y.numpy().astype(np.int64), g.edge_index.numpy()
        masks = dict(np.load(SPLITS / name / "split.npz"))
    return torch.from_numpy(x), y, torch.from_numpy(np.ascontiguousarray(e)).long(), masks


def avaliar(name, seed, device, hp=None, preparar_modelo=None):
    """preparar_modelo(model) é chamado depois do fit e antes da predição (hooks, poda) e devolve uma função
    que desfaz as alterações; o NodePFNClassifier guarda o modelo num cache global entre execuções."""
    hp = hp or HP[name]
    x, y, edge_index, masks = carregar(name)
    n = x.shape[0]
    train_idx = torch.from_numpy(np.where(masks["train"])[0])
    valid_idx = torch.from_numpy(np.where(masks["val"])[0])
    test_idx = torch.from_numpy(np.where(masks["test"])[0])

    fix_seed(seed)
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes=n, add_self_loops=True)
    conv = SimpleConv(aggr="sum")
    X = x
    for _ in range(hp["smoothing_steps"]):
        X = conv(X, edge_index, edge_weight)
    if hp["n_components"] is not None:
        k = min(hp["n_components"], X.shape[1], X.shape[0] - 1)
        X = TruncatedSVD(n_components=k, algorithm=hp["svd_algorithm"], random_state=seed).fit_transform(X)
    X = np.asarray(X)

    query_idx = torch.from_numpy(np.where(~masks["train"])[0])
    valid_mask = torch.isin(query_idx, valid_idx).numpy()
    test_mask = torch.isin(query_idx, test_idx).numpy()
    edge_index_run = update_edge_index(edge_index, train_idx, query_idx)

    t0 = time.time()
    clf = NodePFNClassifier(device=device, base_path=str(AQUI / "models_ckpts" / "nodepfn"),
                            N_ensemble_configurations=hp["n_ensemble"], seed=seed,
                            # 1 configuração do ensemble por vez para caber em 16 GB de GPU; segundo a
                            # documentação do NodePFNClassifier, o resultado não depende deste valor
                            batch_size_inference=1,
                            subsample_features=True, i=0, e=30)
    clf.fit(X[train_idx.numpy()], y[train_idx.numpy()], edge_index_run, overwrite_warning=True)
    desfazer = preparar_modelo(clf.model[2]) if preparar_modelo else None
    try:
        prob = clf.predict_proba(X[query_idx.numpy()], normalize_with_test=True)
    finally:
        if desfazer:
            desfazer()
    dt = time.time() - t0

    y_q = y[query_idx.numpy()]
    out = {}
    for part, msk in [("val", valid_mask), ("test", test_mask)]:
        yt, pt = y_q[msk], prob[msk]
        out[f"acc_{part}"] = float(accuracy_score(yt, pt.argmax(1)))
        if pt.shape[1] == 2:
            out[f"ap_{part}"] = float(average_precision_score(yt, pt[:, 1]))
            out[f"rocauc_{part}"] = float(roc_auc_score(yt, pt[:, 1]))
    out["metrica_artigo_test"] = out["ap_test"] if prob.shape[1] == 2 else out["acc_test"]
    out["metrica_artigo_val"] = out["ap_val"] if prob.shape[1] == 2 else out["acc_val"]
    out["tempo_s"] = dt
    return out


# Grade como na Tabela 11 do artigo do GraphPFN: nº de componentes do TSVD (com a opção sem TSVD) e nº de
# passos de suavização, valores do run.sh do NodePFN; escolha pela métrica de validação.
GRADE_SMOOTHING = [1, 2, 3, 4]
GRADE_COMPONENTES = [15, 20, 25, None]


def buscar(name, device, seed=0, smoothing=None, componentes=None):
    linhas = []
    for s in smoothing or GRADE_SMOOTHING:
        for c in componentes or GRADE_COMPONENTES:
            hp = dict(HP[name], smoothing_steps=s, n_components=c)
            r = avaliar(name, seed, device, hp)
            linhas.append(dict(smoothing_steps=s, n_components=c, val=r["metrica_artigo_val"], test=r["metrica_artigo_test"]))
            print(f"  grade {name}: smoothing={s} componentes={c} val={r['metrica_artigo_val']:.4f}", flush=True)
    melhor = max(linhas, key=lambda x: x["val"])
    return dict(HP[name], smoothing_steps=melhor["smoothing_steps"], n_components=melhor["n_components"]), linhas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HP))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--grade", action="store_true", help="escolhe suavização e TSVD pela validação (semente 0)")
    ap.add_argument("--grade-smoothing", type=int, nargs="+", default=None)
    ap.add_argument("--grade-componentes", type=int, nargs="+", default=None, help="0 = sem TSVD")
    ap.add_argument("--saida", default=None, help="nome do JSON de saída (padrão: <dataset>.json)")
    a = ap.parse_args()
    device = "cpu" if a.cpu else "cuda"
    RESULTS.mkdir(parents=True, exist_ok=True)
    comps = [c or None for c in a.grade_componentes] if a.grade_componentes else None
    hp, grade = (buscar(a.dataset, device, smoothing=a.grade_smoothing, componentes=comps) if a.grade
                 else (HP[a.dataset], None))
    res = {s: avaliar(a.dataset, s, device, hp) for s in a.seeds}
    v = np.array([r["metrica_artigo_test"] for r in res.values()])
    resumo = dict(dataset=a.dataset, hp=hp, grade=grade, por_semente=res,
                  media=float(v.mean()), dp=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  referencia_tabela11_graphpfn=REFERENCIA.get(a.dataset))
    json.dump(resumo, open(RESULTS / (a.saida or f"{a.dataset}.json"), "w"), indent=1)
    print(f"{a.dataset}: métrica do artigo no teste = {v.mean():.4f} ± {resumo['dp']:.4f} "
          f"(referência {REFERENCIA.get(a.dataset)}); por semente {np.round(v, 4).tolist()}", flush=True)
