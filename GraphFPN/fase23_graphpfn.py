"""
Fases 2 e 3 do projeto para o GraphPFN, no MESMO protocolo dos novos GFMs (EXP-006/EXP-007), para que os três PFNs
(GraphPFN, G2T-LimiX, NodePFN) sejam comparáveis:
  - datasets tolokers-2, amazon-ratings, wiki-cs, city-roads-M; protocolo do artigo do GraphPFN (split RL no
    GraphLand, 10/10/80 com semente 0 nos clássicos), ensemble de 10, sementes 0, 1, 2;
  - Fase 2: baseline + BI (todos os tokens e token-alvo) + ablação individual (remover cada camada sozinha);
  - Fase 3: poda cumulativa k = 1..6 pelas ordens de fase3_ordens.py (BI todos, BI alvo, ablação, 3 aleatórias),
    camada 0 protegida.

Funções de inferência copiadas de poda_cosseno_artigo_graphpfn.ipynb (EXP-003, fase 4). O baseline e o BI desse
notebook usam exatamente o mesmo pipeline e as mesmas sementes, então são reaproveitados de
Block_Influence_results/graphfpn_poda_cosseno_artigo/ (origem "artigo"); avaliações de conjuntos de camadas que o
notebook já tinha feito também são reaproveitadas.

Uso (da raiz do repositório):
    python GraphFPN/fase23_graphpfn.py --dataset tolokers-2 --fase 2
    python GraphFPN/fase23_graphpfn.py --dataset tolokers-2 --fase 3
Saída: Block_Influence_results/graphfpn_fase2/<ds>/{baseline,block_influence,ablacao_individual}.csv
       Block_Influence_results/graphfpn_fase3/<ds>/{plano,avaliacoes}.csv   (grava a cada avaliação e retoma)
"""
import argparse
import json
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
os.chdir(AQUI)  # LimiXWrapper procura ./checkpoints/LimiX-16M.ckpt
sys.path.insert(0, str(AQUI / "src"))
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
import dgl  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402

from graphpfn import GraphDataset, GraphPFN, GraphPFNLight  # noqa: E402
from graphpfn.inference.preprocessing import prepare_features_tensor, prepare_graph, standardize_targets  # noqa: E402
from graphpfn.inference.util import apply_model, compute_metrics  # noqa: E402
from graphpfn._vendor.limix.model.layer import MLP, MultiheadAttention  # noqa: E402
from graphpfn.model.layers import GraphPFNResidualModule  # noqa: E402
from graphpfn._vendor.limix.model.encoders import MaskEmbEncoder  # noqa: E402

import fase3_ordens as f3  # noqa: E402

DEVICE = torch.device("cuda")
CHECKPOINT = AQUI / "checkpoints" / "graphpfn-adapters-1_3.pt"
GRAPHLAND_ROOT = Path.home() / ".cache" / "graphland"
PYG_ROOT = Path.home() / ".cache" / "pyg_data"
R = RAIZ / "Block_Influence_results"
ARTIGO = R / "graphfpn_poda_cosseno_artigo"
DATASETS = {"tolokers-2": dict(source="graphland"), "amazon-ratings": dict(source="classic", d_pca=64),
            "wiki-cs": dict(source="classic"), "city-roads-M": dict(source="graphland"),
            # transferência entre datasets (2026-11): mesmos d_pca do protocolo do artigo
            "artnet-exp": dict(source="graphland"), "questions": dict(source="classic", d_pca=64)}
PRIMARY = {"binclass": "average_precision", "multiclass": "accuracy", "regression": "r2"}
N_MEMBERS, N_LAYERS, AMP = 10, 12, True
CHUNK_BUDGET = 2**26

# ---------------------------------------------------------------- inferência em blocos (cópia do notebook)


def _ck(fn, *args):
    if torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _wrap_attention(orig):
    def forward(x, x_kv=None, copy_first_head_kv=False, attn_mask=None,
                calculate_sample_attention=False, calculate_feature_attention=False):
        if attn_mask is not None or calculate_sample_attention or calculate_feature_attention:
            return orig(x, x_kv, copy_first_head_kv, attn_mask, calculate_sample_attention, calculate_feature_attention)
        per_slice = max(x.shape[2], x_kv.shape[2] if x_kv is not None else 0) * x.shape[3] * 4
        step = max(1, CHUNK_BUDGET // per_slice)
        att = lambda a, b: orig(a, b, copy_first_head_kv)[0]  # noqa: E731
        out = torch.cat([_ck(att, x[:, i:i + step], None if x_kv is None else x_kv[:, i:i + step])
                         for i in range(0, x.shape[1], step)], dim=1)
        return out, None, None
    return forward


def _wrap_mlp(orig, hidden):
    def forward(x):
        flat = x.reshape(-1, x.shape[-1])
        step = max(1, CHUNK_BUDGET // hidden)
        return torch.cat([_ck(orig, flat[i:i + step]) for i in range(0, flat.shape[0], step)]).reshape(*x.shape[:-1], -1)
    return forward


def _wrap_graph_residual(orig):
    def forward(graph, x):
        n_tokens = x.shape[2]
        per_token = max(graph.num_edges() * 4 * 3, x.shape[1] * x.shape[3] * 6)
        step = max(1, CHUNK_BUDGET // per_token)
        if step >= n_tokens:
            return _ck(lambda a: orig(graph, a), x)
        return torch.cat([_ck(lambda a: orig(graph, a), x[:, :, i:i + step].contiguous())
                          for i in range(0, n_tokens, step)], dim=2)
    return forward


def _wrap_feature_encoder(orig, out_key):
    def forward(inp):
        n = inp["data"].shape[1]
        step = max(1, CHUNK_BUDGET // (inp["data"].shape[2] * 192 * 6))
        keys = [k for k, v in inp.items() if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == n]
        rest = {k: v for k, v in inp.items() if k not in keys}

        def run(*tensors):
            return orig({**rest, **dict(zip(keys, tensors))})[out_key]
        out = dict(inp)
        out[out_key] = torch.cat([_ck(run, *[inp[k][:, i:i + step] for k in keys]) for i in range(0, n, step)], dim=1)
        return out
    return forward


def enable_low_memory(model):
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.forward = _wrap_attention(m.forward)
        elif isinstance(m, MLP):
            m.forward = _wrap_mlp(m.forward, m.mlp[0].out_features)
        elif isinstance(m, GraphPFNResidualModule):
            m.forward = _wrap_graph_residual(m.forward)
        elif isinstance(m, MaskEmbEncoder):
            m.forward = _wrap_feature_encoder(m.forward, m.out_key)
    model.tfm.module.feature_decoder.forward = lambda x: None
    return model


def load_base():
    base = GraphPFNLight()
    base.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"], strict=False)
    return enable_low_memory(base.eval().to(DEVICE))

# ---------------------------------------------------------------- dados (cópia do notebook)


def random_split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.1 * n), int(0.2 * n)
    masks = {k: np.zeros(n, bool) for k in ["train", "val", "test"]}
    masks["train"][perm[:a]] = True
    masks["val"][perm[a:b]] = True
    masks["test"][perm[b:]] = True
    return masks


def load_classic(name):
    from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
    g = {"amazon-ratings": lambda: HeterophilousGraphDataset(PYG_ROOT, "amazon-ratings"),
         "questions": lambda: HeterophilousGraphDataset(PYG_ROOT, "questions"),
         "wiki-cs": lambda: WikiCS(PYG_ROOT / "wiki-cs", is_undirected=True)}[name]()[0]
    x, y, n = g.x.numpy().astype(np.float32), g.y.numpy(), g.num_nodes
    src, dst = g.edge_index.numpy()
    return GraphDataset(name=name, graph=dgl.graph((src, dst), num_nodes=n, idtype=torch.int32),
                        features={"other": x}, targets=y.astype(np.float32), masks=random_split(n),
                        task_type="binclass" if name == "questions" else "multiclass")


def prepare_task(name):
    cfg = DATASETS[name]
    if cfg["source"] == "graphland":
        ds = GraphDataset.from_graphland(name, root_dir=GRAPHLAND_ROOT / name / name / "raw", split="RL")
    else:
        ds = load_classic(name)
    graph = prepare_graph(ds.graph).to(DEVICE)
    np.random.seed(0)
    features = prepare_features_tensor(ds.features, ds.masks["train"], d_pca=cfg.get("d_pca")).to(DEVICE)
    train_mask = torch.tensor(ds.masks["train"]).to(DEVICE)
    y_train = torch.tensor(ds.targets).to(DEVICE)[train_mask]
    stats = None
    if ds.task_type == "regression":
        y_train, stats = standardize_targets(y_train=y_train)
    else:
        y_train = y_train.long()
    return dict(name=name, ds=ds, task=ds.task_type, graph=graph, features=features, train_mask=train_mask,
                y_train=y_train, stats=stats, metric=PRIMARY[ds.task_type])


@contextmanager
def pruned(base, remove=()):
    stack = base.tfm.module.transformer_encoder
    full = stack.layers
    remove = set(int(i) for i in remove)
    stack.layers = nn.ModuleList([l for i, l in enumerate(full) if i not in remove])
    try:
        yield
    finally:
        stack.layers = full


@torch.no_grad()
def avaliar(base, t, seed, remove=()):
    """(val, test) na métrica do artigo; mesma semente = mesmas permutações do ensemble e features aleatórias."""
    torch.manual_seed(seed)
    model = GraphPFN(base, n_members=N_MEMBERS, n_features=t["features"].shape[1], n_classes=t["ds"].n_classes,
                     device=DEVICE, verbose=False).eval()
    with pruned(base, remove):
        preds = apply_model(model=model, dataset=t["ds"], graph=t["graph"], features=t["features"],
                            y_train=t["y_train"], train_mask=t["train_mask"],
                            regression_target_stats=t["stats"], amp=AMP, device=DEVICE)
    out = []
    for split in ["val", "test"]:
        m = t["ds"].masks[split]
        out.append(compute_metrics(y_true=t["ds"].targets[m], y_pred=preds[m], task_type=t["task"])[t["metric"]])
    return tuple(out)


def chave_artigo(conj):
    return str(sorted(int(i) for i in conj))


def avaliacoes_artigo(name, metric):
    """{chave '[1, 2]': {seed: (val, test)}} do notebook do protocolo do artigo."""
    a = pd.read_csv(ARTIGO / "avaliacoes.csv")
    a = a[a.dataset == name]
    return {k: {int(r.seed): (getattr(r, f"val_{metric}"), getattr(r, f"test_{metric}")) for r in g.itertuples()}
            for k, g in a.groupby("removidas")}


def fase2(name, seeds):
    out = R / "graphfpn_fase2" / name
    out.mkdir(parents=True, exist_ok=True)
    t = prepare_task(name)
    conhecidas = avaliacoes_artigo(name, t["metric"])
    # baseline e BI: reaproveitados do notebook (mesmo pipeline e sementes)
    base_rows = [dict(dataset=name, seed=s, val=v, test=te) for s, (v, te) in sorted(conhecidas["[]"].items()) if s in seeds]
    pd.DataFrame(base_rows).to_csv(out / "baseline.csv", index=False)
    bi = pd.read_csv(ARTIGO / "block_influence.csv")
    bi[(bi.dataset == name) & bi.seed.isin(seeds)].to_csv(out / "block_influence.csv", index=False)
    print(f"{name}: baseline/BI reaproveitados do protocolo do artigo; teste = "
          f"{np.mean([r['test'] for r in base_rows]):.4f}", flush=True)
    caminho = out / "ablacao_individual.csv"
    feitos = pd.read_csv(caminho) if caminho.exists() else pd.DataFrame(columns=["dataset", "seed", "removida", "val", "test"])
    base = load_base()
    for seed in seeds:
        for i in range(N_LAYERS):
            if ((feitos["seed"] == seed) & (feitos["removida"] == i)).any():
                continue
            val, test = avaliar(base, t, seed, remove=(i,))
            feitos = pd.concat([feitos, pd.DataFrame([dict(dataset=name, seed=seed, removida=i, val=val, test=test)])],
                               ignore_index=True)
            feitos.to_csv(caminho, index=False)
            print(f"{name} seed={seed} sem camada {i}: val={val:.4f} test={test:.4f}", flush=True)


def fase3(name, seeds):
    out = R / "graphfpn_fase3" / name
    out.mkdir(parents=True, exist_ok=True)
    plano = f3.plano("graphfpn_fase2", name)
    pd.DataFrame([dict(estrategia=e, k=k, removidas="-".join(map(str, c))) for e, k, c in plano]).to_csv(out / "plano.csv", index=False)
    caminho = out / "avaliacoes.csv"
    feitos = pd.read_csv(caminho, dtype={"removidas": str}) if caminho.exists() else pd.DataFrame(
        columns=["dataset", "seed", "removidas", "k", "val", "test", "origem"])
    t = prepare_task(name)
    conhecidas = avaliacoes_artigo(name, t["metric"])
    for c, d in f3.ablacao_conhecida("graphfpn_fase2", name).items():
        conhecidas[chave_artigo(c)] = {**conhecidas.get(chave_artigo(c), {}), **d}
    base = None
    for conj in sorted({c for _, _, c in plano}, key=lambda c: (len(c), c)):
        chave = "-".join(map(str, conj))
        for seed in seeds:
            if ((feitos.seed == seed) & (feitos.removidas == chave)).any():
                continue
            k_art = chave_artigo(conj)
            if k_art in conhecidas and seed in conhecidas[k_art]:
                (val, test), origem = conhecidas[k_art][seed], ("fase2" if len(conj) == 1 else "artigo")
            else:
                base = base or load_base()
                val, test = avaliar(base, t, seed, remove=conj)
                origem = "fase3"
            feitos = pd.concat([feitos, pd.DataFrame([dict(dataset=name, seed=seed, removidas=chave, k=len(conj),
                                                           val=val, test=test, origem=origem)])], ignore_index=True)
            feitos.to_csv(caminho, index=False)
            print(f"{name} seed={seed} sem {chave}: val={val:.4f} test={test:.4f} ({origem})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--fase", type=int, choices=[2, 3], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    (fase2 if a.fase == 2 else fase3)(a.dataset, a.seeds)
