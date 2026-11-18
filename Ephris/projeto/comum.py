"""
Dados, splits e avaliação do Ephris no protocolo comum dos PFNs (datasets e splits do artigo do GraphPFN).

- GraphLand (tolokers-2, artnet-exp, city-reviews): split RL oficial, nós sem rótulo fora das máscaras;
  features como em graphpfn.data.GraphDataset.from_graphland (remove constantes; numéricas com NaN imputadas
  pela moda; categóricas em one-hot; frações como numéricas).
- Clássicos (amazon-ratings, wiki-cs, questions, facebook, pubmed): split aleatório 10/10/80 com semente 0,
  a mesma função random_split do GraphPFN (e o mesmo split.npz do G2T-FM/NodePFN); facebook do MUSAE com
  features reduzidas a 128 por TruncatedSVD (como no GraphPFN); pubmed sem PCA.
- O Ephris só faz classificação: os datasets de regressão do protocolo ficam de fora.

O Ephris recebe as features brutas e faz o próprio pré-processamento (EphrisClassifier); rótulos de nós
fora do contexto nunca entram no modelo.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score

from ephris import EphrisClassifier

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent  # Graph_Pruning/
R = RAIZ / "Block_Influence_results"
CACHE = AQUI / "cache"
GRAPHLAND = Path.home() / ".cache" / "graphland"
PYG = Path.home() / ".cache" / "pyg_data"

GRAPHLAND_DS = ["tolokers-2", "artnet-exp", "city-reviews"]
CLASSIC_DS = ["amazon-ratings", "wiki-cs", "questions", "facebook", "pubmed"]
DATASETS = GRAPHLAND_DS + CLASSIC_DS
BINARIOS = {"tolokers-2", "artnet-exp", "city-reviews", "questions"}  # métrica AP; demais: acurácia
SEEDS = [0, 1, 2]


def random_split(n, seed=0):
    """Igual a random_split de GraphFPN/poda_cosseno_artigo_graphpfn.ipynb."""
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.1 * n), int(0.2 * n)
    masks = {k: np.zeros(n, bool) for k in ["train", "val", "test"]}
    masks["train"][perm[:a]] = True
    masks["val"][perm[a:b]] = True
    masks["test"][perm[b:]] = True
    return masks


def _graphland(name):
    d = GRAPHLAND / name / name / "raw" / name
    info = yaml.safe_load(open(d / "info.yaml"))
    f = pd.read_csv(d / "features.csv", index_col=0).astype(np.float32)
    f = f.loc[:, f.apply(pd.Series.nunique) != 1]
    cols = set(f.columns)
    partes = []
    num = [c for c in info.get("numerical_features_names") or [] if c in cols]
    num += [c for c in info.get("fraction_features_names") or [] if c in cols]
    if num:
        v = f[num].values
        if np.isnan(v).any():
            from sklearn.impute import SimpleImputer
            v = SimpleImputer(strategy="most_frequent").fit_transform(v)
        partes.append(v.astype(np.float32))
    cat = [c for c in info.get("categorical_features_names") or [] if c in cols]
    if cat:
        partes.append(pd.get_dummies(f[cat].astype("Int64").astype(str)).values.astype(np.float32))
    resto = [c for c in f.columns if c not in set(num) | set(cat)]
    if resto:
        partes.append(np.nan_to_num(f[resto].values.astype(np.float32)))
    x = np.concatenate(partes, axis=1)
    y = pd.read_csv(d / "targets.csv", index_col=0).values.squeeze().astype(np.float32)
    m = pd.read_csv(d / "split_masks_RL.csv", index_col=0).astype(bool)
    rotulado = ~np.isnan(y)
    masks = {k: m[k].values & rotulado for k in ["train", "val", "test"]}
    e = pd.read_csv(d / "edgelist.csv").values[:, :2].T.astype(np.int64)
    return x, np.nan_to_num(y, nan=-1).astype(np.int64), e, masks


def _classico(name):
    if name == "facebook":
        from sklearn.decomposition import TruncatedSVD
        import scipy.sparse as sp
        d = PYG / "facebook_musae"
        e, f = pd.read_csv(d / "edges.csv"), pd.read_csv(d / "features.csv")
        t = pd.read_csv(d / "target.csv").sort_values("id")
        n = len(t)
        X = sp.csr_matrix((f["value"], (f["node_id"], f["feature_id"])), shape=(n, f["feature_id"].max() + 1))
        x = TruncatedSVD(128, random_state=0).fit_transform(X).astype(np.float32)
        ei = np.stack([e["id_1"].values, e["id_2"].values]).astype(np.int64)
        y = t["target"].values.astype(np.int64)
    else:
        from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid, WikiCS
        g = {"amazon-ratings": lambda: HeterophilousGraphDataset(PYG, "amazon-ratings"),
             "questions": lambda: HeterophilousGraphDataset(PYG, "questions"),
             "wiki-cs": lambda: WikiCS(PYG / "wiki-cs", is_undirected=True),
             "pubmed": lambda: Planetoid(PYG, "PubMed")}[name]()[0]
        x, y, ei, n = g.x.numpy().astype(np.float32), g.y.numpy().astype(np.int64), g.edge_index.numpy(), g.num_nodes
    return x, y, ei, random_split(len(y))


def carregar(name):
    """x [N,F] float32, y [N] int64 (−1 = sem rótulo), edge_index [2,E] int64, masks {train,val,test}."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}.npz"
    if not p.exists():
        x, y, e, m = _graphland(name) if name in GRAPHLAND_DS else _classico(name)
        np.savez_compressed(p, x=x, y=y, e=e, **{f"m_{k}": v for k, v in m.items()})
    z = np.load(p)
    return z["x"], z["y"], z["e"], {k: z[f"m_{k}"] for k in ["train", "val", "test"]}


def metrica(name, y, logp):
    """AP (binária) ou acurácia (multiclasse); logp [n, C] na ordem das classes 0..C-1."""
    if name in BINARIOS:
        return float(average_precision_score(y, np.exp(logp[:, 1])))
    return float(accuracy_score(y, logp.argmax(1)))


def acaso(name, y_test):
    """AP do classificador constante = fração de positivos; acurácia = classe majoritária."""
    if name in BINARIOS:
        return float((y_test == 1).mean())
    return float(np.bincount(y_test).max() / len(y_test))


class Tarefa:
    """Um dataset preparado uma vez no Ephris (contexto = nós de treino). O modelo preparado fica acessível
    em .model (EphrisModel) para hooks e poda; predict() roda só o forward com o modelo atual."""

    def __init__(self, name, seed=0, device="cuda:0", context_idx=None):
        self.name, self.seed = name, seed
        x, y, e, masks = carregar(name)
        self.x, self.y, self.e, self.masks = x, y, e, masks
        self.n_classes = int(y.max()) + 1
        ctx = np.flatnonzero(masks["train"]) if context_idx is None else np.asarray(context_idx)
        self.context_idx = ctx
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.clf = EphrisClassifier(device=device, random_state=seed)
        t0 = time.time()
        self.clf.prepare(torch.from_numpy(x), torch.from_numpy(e), context_indices=torch.from_numpy(ctx),
                         context_labels=torch.from_numpy(y[ctx]))
        self.t_prepare = time.time() - t0
        assert self.clf.classes_.tolist() == list(range(self.n_classes)), self.clf.classes_
        task = self.clf._session._task
        assert task.ecoc is None, "ECOC (mais de 10 classes) não suportado neste projeto"
        self.prepared = task.model
        self.model = task.model.model
        self.context = task.context

    def predict(self):
        """log-probabilidades [N, C] (CPU, numpy) com o modelo no estado atual."""
        return self.clf.predict_prepared_log_proba().numpy()

    def avaliar(self, logp=None):
        logp = self.predict() if logp is None else logp
        out = {}
        for part in ["val", "test"]:
            m = self.masks[part]
            out[part] = metrica(self.name, self.y[m], logp[m])
        return out


def salvar_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)
