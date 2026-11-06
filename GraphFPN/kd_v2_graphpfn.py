"""
KD do GraphPFN, versão 2 (student podado × aleatório). Funções copiadas de
kd_podado_vs_aleatorio_graphpfn.ipynb; o que muda está no fim do arquivo.

Uso (da raiz do repositório):
    python GraphFPN/kd_v2_graphpfn.py --dataset tolokers-2 --seed 0
"""
import os, sys, json, time, random, warnings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from unittest import mock

cwd = Path.cwd().resolve()
PROJECT_ROOT = cwd if (cwd / "src" / "graphpfn").exists() else cwd / "GraphFPN"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import dgl
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, average_precision_score, r2_score

import graphpfn.model.light as light_module
from graphpfn import GraphDataset, GraphPFN, GraphPFNLight
from graphpfn.model.limix import LimiXWrapper
from graphpfn.model.ensemble import apply_feature_transform, apply_target_transform
from graphpfn.inference.preprocessing import prepare_features_tensor, prepare_graph, standardize_targets
from graphpfn.inference.util import CandidateQueue

DEVICE = torch.device("cuda")
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "graphpfn-adapters-1_3.pt"
GRAPHLAND_ROOT = Path.home() / ".cache" / "graphland"
PYG_ROOT = Path.home() / ".cache" / "pyg_data"
RESULTS_DIR = PROJECT_ROOT.parent / "Block_Influence_results" / "graphfpn_kd_podado_vs_aleatorio"

DATASETS = {
    "tolokers-2":     dict(source="graphland"),
    "amazon-ratings": dict(source="classic", d_pca=64),
    "city-roads-M":   dict(source="graphland"),
}
N_LAYERS = 12
PRUNE_K = 6                 # camadas removidas → student de 6 camadas
PROTECTED = {0}
SEED = 0

# KD (mesmos valores padrão de predict_ft)
TRAIN_BATCH = 1024
MIN_CONTEXT_RATIO = 0.75
TEMPERATURE = 2.0
N_STEPS = 200
EVAL_EVERY = 10
LR_GRID = [1e-4, 3e-4, 1e-3]
SWEEP_STEPS = 20
N_MEMBERS_FINAL = 10        # avaliação final com o ensemble do artigo
AMP = True

PRIMARY_METRIC = {"binclass": "AP", "multiclass": "Acurácia", "regression": "R²"}
STUDENTS = ["Podado", "Aleatório"]
COLORS = {"Podado": "#2a78d6", "Aleatório": "#eb6834"}
GRAY = "#52514e"
print(f"GPU: {torch.cuda.get_device_name()}")

import torch.utils.checkpoint
from graphpfn._vendor.limix.model.layer import MLP, MultiheadAttention
from graphpfn.model.layers import GraphPFNResidualModule
from graphpfn._vendor.limix.model.encoders import MaskEmbEncoder

CHUNK_BUDGET = 2**26  # elementos por bloco de computação


def _ck(fn, *args):
    """No treino (com gradiente), cada bloco é recalculado no backward em vez de guardar suas ativações."""
    if torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _wrap_attention(orig):
    """Atenção de features ([1, nós, tokens, E]) e de amostras ([1, tokens, nós, E]): independente ao longo da dim 1."""
    def forward(x, x_kv=None, copy_first_head_kv=False, attn_mask=None,
                calculate_sample_attention=False, calculate_feature_attention=False):
        if attn_mask is not None or calculate_sample_attention or calculate_feature_attention:
            return orig(x, x_kv, copy_first_head_kv, attn_mask, calculate_sample_attention, calculate_feature_attention)
        per_slice = max(x.shape[2], x_kv.shape[2] if x_kv is not None else 0) * x.shape[3] * 4
        step = max(1, CHUNK_BUDGET // per_slice)
        att = lambda a, b: orig(a, b, copy_first_head_kv)[0]
        out = torch.cat([_ck(att, x[:, i:i + step], None if x_kv is None else x_kv[:, i:i + step])
                         for i in range(0, x.shape[1], step)], dim=1)
        return out, None, None
    return forward


def _wrap_mlp(orig, hidden):
    """MLP atua token a token."""
    def forward(x):
        flat = x.reshape(-1, x.shape[-1])
        step = max(1, CHUNK_BUDGET // hidden)
        return torch.cat([_ck(orig, flat[i:i + step]) for i in range(0, flat.shape[0], step)]).reshape(*x.shape[:-1], -1)
    return forward


def _wrap_graph_residual(orig):
    """Adaptadores de grafo (atenção nas arestas e MLP): independentes por posição de token."""
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
    """Encoder de features: cada valor vira um embedding sem olhar os outros nós."""
    def forward(inp):
        n = inp["data"].shape[1]
        step = max(1, CHUNK_BUDGET // (inp["data"].shape[2] * 192 * 6))
        keys = [k for k, v in inp.items() if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == n]
        rest = {k: v for k, v in inp.items() if k not in keys}

        def run(*tensors):
            return orig({**rest, **dict(zip(keys, tensors))})[out_key]
        out = dict(inp)  # não altera o dicionário de entrada (o recálculo do checkpoint o reutiliza)
        out[out_key] = torch.cat([_ck(run, *[inp[k][:, i:i + step] for k in keys])
                                  for i in range(0, n, step)], dim=1)
        return out
    return forward


def enable_low_memory(model):
    """Processa em blocos as operações independentes por nó/token. Matematicamente idêntico ao forward original."""
    if getattr(model, "_low_memory", False):
        return model
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.forward = _wrap_attention(m.forward)
        elif isinstance(m, MLP):
            m.forward = _wrap_mlp(m.forward, m.mlp[0].out_features)
        elif isinstance(m, GraphPFNResidualModule):
            m.forward = _wrap_graph_residual(m.forward)
        elif isinstance(m, MaskEmbEncoder):
            m.forward = _wrap_feature_encoder(m.forward, m.out_key)
    # feature_decoder só alimenta a perda auxiliar de reconstrução de features (MGM, usada no pré-treino);
    # a predição sai de cls_y_decoder/reg_y_decoder. Na inferência ele é o pico de memória, então é desligado.
    model.tfm.module.feature_decoder.forward = lambda x: None
    model._low_memory = True
    return model

def random_split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.1 * n), int(0.2 * n)
    masks = {k: np.zeros(n, bool) for k in ["train", "val", "test"]}
    masks["train"][perm[:a]] = True; masks["val"][perm[a:b]] = True; masks["test"][perm[b:]] = True
    return masks

def load_classic(name):
    # Copiada de poda_cosseno_artigo_graphpfn.ipynb (a versão do notebook de KD só carregava HeterophilousGraphDataset)
    if name == "facebook":
        d = PYG_ROOT / "facebook_musae"   # MUSAE: edges.csv, features.csv (esparso), target.csv
        e, f = pd.read_csv(d / "edges.csv"), pd.read_csv(d / "features.csv")
        t = pd.read_csv(d / "target.csv").sort_values("id")
        n = len(t)
        X = sp.csr_matrix((f["value"], (f["node_id"], f["feature_id"])), shape=(n, f["feature_id"].max() + 1))
        x = TruncatedSVD(128, random_state=0).fit_transform(X).astype(np.float32)
        src, dst, y = e["id_1"].values, e["id_2"].values, t["target"].values
    else:
        from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS, Planetoid
        g = {"amazon-ratings": lambda: HeterophilousGraphDataset(PYG_ROOT, "amazon-ratings"),
             "questions": lambda: HeterophilousGraphDataset(PYG_ROOT, "questions"),
             "wiki-cs": lambda: WikiCS(PYG_ROOT / "wiki-cs", is_undirected=True),
             "pubmed": lambda: Planetoid(PYG_ROOT, "PubMed")}[name]()[0]
        x, y, n = g.x.numpy().astype(np.float32), g.y.numpy(), g.num_nodes
        src, dst = g.edge_index.numpy()
    return GraphDataset(name=name, graph=dgl.graph((src, dst), num_nodes=n, idtype=torch.int32),
                        features={"other": x}, targets=y.astype(np.float32), masks=random_split(n),
                        task_type="binclass" if name == "questions" else "multiclass")

def prepare_task(name, cfg):
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
    return dict(name=name, ds=ds, task=ds.task_type, graph=graph, features=features,
                train_mask=train_mask, y_train=y_train, stats=stats, n_classes=ds.n_classes)

def load_pretrained():
    base = GraphPFNLight()
    base.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["state_dict"], strict=False)
    return base.to(DEVICE)

def load_random(seed):
    torch.manual_seed(seed)
    with mock.patch.object(light_module, "LimiXWrapper", lambda load_weights=True: LimiXWrapper(load_weights=False)):
        base = GraphPFNLight()
    return base.to(DEVICE)

def keep_layers(base, keep):
    stack = base.tfm.module.transformer_encoder
    stack.layers = nn.ModuleList([stack.layers[i] for i in keep])
    return base

def checkpoint_layers(base):
    """Gradient checkpointing por camada inteira durante o treino: guarda só a entrada de cada camada e
    recalcula a camada no backward. Mesmo resultado, muito menos memória. (O GraphPFN só liga o seu
    checkpointing interno para entradas com mais de 400 mil elementos, o que deixa de fora o tolokers-2.)"""
    def wrap(layer):
        orig = layer.forward
        holder = layer._graph_holder
        def forward(*a, **k):
            if not torch.is_grad_enabled():
                return orig(*a, **k)
            graph = holder.graph  # o GraphPFN limpa o grafo ao fim do forward; o recálculo precisa dele
            def run(*a2, **k2):
                saved, holder.graph = holder.graph, graph
                try:
                    return orig(*a2, **k2)
                finally:
                    holder.graph = saved
            return torch.utils.checkpoint.checkpoint(run, *a, use_reentrant=False, **k)
        layer.forward = forward
    for layer in base.tfm.module.transformer_encoder.layers:
        wrap(layer)
    return base

def n_params(m):
    return sum(p.numel() for p in m.parameters())

def raw_forward(base, t, member=None, context_mask=None, y_context=None, seed=0):
    """Forward do GraphPFNLight. Classificação: log-probabilidades; regressão: alvo padronizado."""
    feats = t["features"] if member is None else apply_feature_transform(t["features"], member)
    if context_mask is None:
        context_mask, y_context = t["train_mask"], t["y_train"]
    if member is not None:
        y_context = apply_target_transform(y_context, member)
    torch.manual_seed(seed)  # mesmas features aleatórias para teacher e student
    with torch.autocast("cuda", enabled=AMP, dtype=torch.bfloat16):
        return base(graph=t["graph"], features=feats, y_train=y_context, train_mask=context_mask,
                    task_type=t["task"]).float()

@torch.no_grad()
def measure_bi(base, t):
    """BI cosseno (todos os tokens) por camada, em um forward com o modelo completo."""
    acc = {}
    def hook(i):
        def h(m, a, kw, out):
            x, y = a[0][0], out[0][0]
            s = n = 0.0
            for c in range(0, x.shape[0], 4096):
                cos = F.cosine_similarity(x[c:c + 4096].float(), y[c:c + 4096].float(), dim=-1)
                s += (1 - cos).sum().item(); n += cos.numel()
            acc[i] = s / n
        return h
    hooks = [l.register_forward_hook(hook(i), with_kwargs=True)
             for i, l in enumerate(base.tfm.module.transformer_encoder.layers)]
    try:
        raw_forward(base, t)
    finally:
        for h in hooks:
            h.remove()
    return pd.Series(acc).sort_index()

def kd_loss(s, t_out, task, n_classes):
    if task == "regression":
        return F.mse_loss(s, t_out)
    zs, zt = s[..., :n_classes] / TEMPERATURE, t_out[..., :n_classes] / TEMPERATURE
    return F.kl_div(F.log_softmax(zs, -1), F.log_softmax(zt, -1), log_target=True, reduction="batchmean") * TEMPERATURE ** 2

def task_metric(t, raw, split):
    m = t["ds"].masks[split]
    y = t["ds"].targets[m]
    p = raw[m]
    if t["task"] == "regression":
        return r2_score(y, float(t["stats"].mean) + float(t["stats"].std) * p)
    probs = np.exp(p[:, :t["n_classes"]])
    if t["task"] == "binclass":
        return average_precision_score(y, probs[:, 1] / probs.sum(1))
    return accuracy_score(y, probs.argmax(1))

@torch.no_grad()
def evaluate(base, t, teacher_raw):
    """Loss de KD (val/teste) contra o teacher e métrica do artigo, em ICL com todo o treino como contexto."""
    base.eval()
    raw = raw_forward(base, t, seed=SEED)
    out = {}
    for split in ["val", "test"]:
        m = torch.tensor(t["ds"].masks[split], device=DEVICE)
        out[f"kd_{split}"] = kd_loss(raw[m], teacher_raw[m], t["task"], t["n_classes"]).item()
        out[f"metric_{split}"] = task_metric(t, raw.cpu().numpy(), split)
    return out

@torch.no_grad()
def evaluate_ensemble(base, t, n_members=N_MEMBERS_FINAL):
    """Métrica no teste com o ensemble do artigo (10 membros)."""
    base.eval()
    torch.manual_seed(SEED)
    model = GraphPFN(base, n_members=n_members, n_features=t["features"].shape[1], n_classes=t["n_classes"],
                     device=DEVICE, verbose=False).eval()
    with torch.autocast("cuda", enabled=AMP, dtype=torch.bfloat16):
        p = model(graph=t["graph"], features=t["features"], y_train=t["y_train"],
                  train_mask=t["train_mask"], task_type=t["task"]).float()
    raw = p.cpu().numpy() if t["task"] == "regression" else p[:, :t["n_classes"]].cpu().numpy()
    return task_metric(t, raw, "test")

def kd_train(student, teacher, t, lr, n_steps, teacher_raw, label, eval_every=EVAL_EVERY):
    torch.manual_seed(SEED); random.seed(SEED)
    ens = GraphPFN(teacher, n_members=10, n_features=t["features"].shape[1], n_classes=t["n_classes"],
                   device=DEVICE, verbose=False).ensemble
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    train_idx = t["train_mask"].nonzero().squeeze(1)
    n_train, n_nodes = len(train_idx), t["train_mask"].shape[0]
    queue = CandidateQueue(n_train, n_candidates=min(TRAIN_BATCH, int(n_train * (1 - MIN_CONTEXT_RATIO))), device=DEVICE)

    log = [dict(step=0, t_kd=0.0, train_loss=np.nan, **evaluate(student, t, teacher_raw))]
    t_kd, losses = 0.0, []
    for step in range(1, n_steps + 1):
        torch.cuda.synchronize(); tic = time.time()
        student.train()
        q = next(queue)
        query = torch.zeros(n_train, dtype=torch.bool, device=DEVICE); query[q] = True
        ctx_nodes = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE); ctx_nodes[train_idx[~query]] = True
        q_nodes = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE); q_nodes[train_idx[query]] = True
        _, member = ens.sample_member()
        y_ctx = t["y_train"][~query]
        with torch.no_grad():
            t_out = raw_forward(teacher, t, member, ctx_nodes, y_ctx, seed=SEED * 100_000 + step)[q_nodes]
        s_out = raw_forward(student, t, member, ctx_nodes, y_ctx, seed=SEED * 100_000 + step)[q_nodes]
        loss = kd_loss(s_out, t_out, t["task"], t["n_classes"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(); t_kd += time.time() - tic
        losses.append(loss.item())
        if not np.isfinite(losses[-1]):
            print(f"  {label}: loss não finita no passo {step}, interrompido")
            break
        if step % eval_every == 0 or step == n_steps:
            log.append(dict(step=step, t_kd=t_kd, train_loss=float(np.mean(losses[-eval_every:])),
                            **evaluate(student, t, teacher_raw)))
    return pd.DataFrame(log)

def build_student(kind, t, teacher):
    """Devolve (student, tempo_de_poda_s, camadas_mantidas)."""
    if kind == "Podado":
        torch.cuda.synchronize(); tic = time.time()
        bi = measure_bi(teacher, t)
        removed = [l for l in bi.sort_values().index if l not in PROTECTED][:PRUNE_K]
        keep = [i for i in range(N_LAYERS) if i not in removed]
        student = keep_layers(load_pretrained(), keep)
        torch.cuda.synchronize()
        return checkpoint_layers(enable_low_memory(student)), time.time() - tic, keep
    keep = list(range(N_LAYERS - PRUNE_K))
    return checkpoint_layers(enable_low_memory(keep_layers(load_random(SEED), keep))), 0.0, keep



# =============================================================================
# kd_v2: mudanças em relação a kd_podado_vs_aleatorio_graphpfn.ipynb
#   - melhor ponto: a cada EVAL_EVERY passos o student é avaliado na validação; guarda-se o estado
#     com a melhor métrica de validação e para-se após PATIENCE avaliações sem melhora, como em
#     graphpfn.inference.finetune.predict_ft (o teste nunca escolhe nada);
#   - semente, nº de camadas removidas, passos e students configuráveis pela linha de comando;
#   - um vigia encerra o processo se a RAM disponível da máquina cair abaixo do limite.
# =============================================================================
import argparse
import copy
import threading

import psutil


def vigia_memoria(limite_gb):
    while True:
        disp = psutil.virtual_memory().available / 2**30
        if disp < limite_gb:
            print(f"\n[vigia] RAM disponível {disp:.1f} GB < {limite_gb} GB: encerrando.", flush=True)
            os._exit(3)
        time.sleep(0.5)


ALL_DATASETS = {
    "tolokers-2": dict(source="graphland"),
    "amazon-ratings": dict(source="classic", d_pca=64),
    "city-roads-M": dict(source="graphland"),
    "artnet-exp": dict(source="graphland"),
    "artnet-views": dict(source="graphland"),
    "hm-prices": dict(source="graphland"),
    "twitch-views": dict(source="graphland"),
    "questions": dict(source="classic", d_pca=64),
    "wiki-cs": dict(source="classic"),
}


def kd_train_v2(student, teacher, t, lr, n_steps, teacher_raw, label, eval_every=EVAL_EVERY, patience=None,
                estado_path=None, extra=None):
    """kd_train do notebook + melhor ponto pela métrica de validação e parada antecipada opcional.
    Com estado_path, o estado é salvo a cada avaliação e o treino retoma dali se o processo for encerrado
    (a sequência aleatória de consultas e membros do ensemble após a retomada não é a mesma de uma execução contínua)."""
    torch.manual_seed(SEED); random.seed(SEED)
    ens = GraphPFN(teacher, n_members=10, n_features=t["features"].shape[1], n_classes=t["n_classes"],
                   device=DEVICE, verbose=False).ensemble
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    train_idx = t["train_mask"].nonzero().squeeze(1)
    n_train, n_nodes = len(train_idx), t["train_mask"].shape[0]
    queue = CandidateQueue(n_train, n_candidates=min(TRAIN_BATCH, int(n_train * (1 - MIN_CONTEXT_RATIO))), device=DEVICE)

    inicio, retomadas = 0, []
    if estado_path is not None and estado_path.exists():
        e = torch.load(estado_path, map_location="cpu", weights_only=False)
        student.load_state_dict(e["student"]); opt.load_state_dict(e["opt"])
        log, best, sem_melhora, t_kd, losses = e["log"], e["best"], e["sem_melhora"], e["t_kd"], e["losses"]
        inicio, retomadas = e["step"], e["retomadas"] + [e["step"]]
        torch.manual_seed(SEED * 1000 + inicio); random.seed(SEED * 1000 + inicio)
        print(f"  {label}: retomando do passo {inicio}", flush=True)
    else:
        log = [dict(step=0, t_kd=0.0, train_loss=np.nan, **evaluate(student, t, teacher_raw))]
        best = dict(step=0, metric_val=log[0]["metric_val"],
                    state={k: v.detach().cpu().clone() for k, v in student.state_dict().items()})
        sem_melhora = 0
        t_kd, losses = 0.0, []
    for step in range(inicio + 1, n_steps + 1):
        torch.cuda.synchronize(); tic = time.time()
        student.train()
        q = next(queue)
        query = torch.zeros(n_train, dtype=torch.bool, device=DEVICE); query[q] = True
        ctx_nodes = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE); ctx_nodes[train_idx[~query]] = True
        q_nodes = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE); q_nodes[train_idx[query]] = True
        _, member = ens.sample_member()
        y_ctx = t["y_train"][~query]
        with torch.no_grad():
            t_out = raw_forward(teacher, t, member, ctx_nodes, y_ctx, seed=SEED * 100_000 + step)[q_nodes]
        s_out = raw_forward(student, t, member, ctx_nodes, y_ctx, seed=SEED * 100_000 + step)[q_nodes]
        loss = kd_loss(s_out, t_out, t["task"], t["n_classes"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(); t_kd += time.time() - tic
        losses.append(loss.item())
        if not np.isfinite(losses[-1]):
            print(f"  {label}: loss não finita no passo {step}, interrompido", flush=True)
            break
        if step % eval_every == 0 or step == n_steps:
            log.append(dict(step=step, t_kd=t_kd, train_loss=float(np.mean(losses[-eval_every:])),
                            **evaluate(student, t, teacher_raw)))
            mv = log[-1]["metric_val"]
            print(f"  {label} passo {step}: kd_val={log[-1]['kd_val']:.4f} metric_val={mv:.4f}", flush=True)
            if mv > best["metric_val"]:
                best = dict(step=step, metric_val=mv,
                            state={k: v.detach().cpu().clone() for k, v in student.state_dict().items()})
                sem_melhora = 0
            else:
                sem_melhora += 1
                if patience is not None and sem_melhora >= patience:
                    print(f"  {label}: parada antecipada no passo {step} (melhor: {best['step']})", flush=True)
                    break
            if estado_path is not None:
                tmp = estado_path.with_suffix(".tmp")
                torch.save(dict(student=student.state_dict(), opt=opt.state_dict(), log=log, best=best,
                                sem_melhora=sem_melhora, t_kd=t_kd, losses=losses[-eval_every:], step=step,
                                retomadas=retomadas, extra=extra), tmp)
                tmp.replace(estado_path)
    best["retomadas"] = retomadas
    return pd.DataFrame(log), best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ALL_DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prune-k", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=4, help="avaliações sem melhora na validação; 0 = sem parada")
    ap.add_argument("--students", default="Podado,Aleatório")
    ap.add_argument("--lr", type=float, default=None, help="pula a varredura e usa este lr")
    ap.add_argument("--limite-ram-gb", type=float, default=3.0)
    ap.add_argument("--out", default="graphfpn_kd_v2")
    a = ap.parse_args()
    threading.Thread(target=vigia_memoria, args=(a.limite_ram_gb,), daemon=True).start()

    SEED = a.seed
    PRUNE_K = a.prune_k
    patience = a.patience or None
    out_dir = PROJECT_ROOT.parent / "Block_Influence_results" / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    name, cfg = a.dataset, ALL_DATASETS[a.dataset]
    t = prepare_task(name, cfg)
    teacher = enable_low_memory(load_pretrained()).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        teacher_raw = raw_forward(teacher, t, seed=SEED)
    teacher_ens = evaluate_ensemble(teacher, t)
    print(f"{name} seed={SEED}: teacher (ensemble 10) = {teacher_ens:.4f}", flush=True)

    for kind in a.students.split(","):
        tag = f"{name}_k{PRUNE_K}_s{SEED}_{kind}_{a.steps}p{a.patience}" + (f"_lr{a.lr:g}" if a.lr else "")
        out_log, out_sum = out_dir / f"kd_log_{tag}.csv", out_dir / f"kd_resumo_{tag}.json"
        if out_log.exists() and out_sum.exists():
            print(f"(já feito: {tag})", flush=True)
            continue
        estado = out_dir / f"estado_{tag}.pt"
        salvo = torch.load(estado, map_location="cpu", weights_only=False)["extra"] if estado.exists() else None
        if salvo:
            best_lr, sweep = salvo["lr"], salvo["sweep"]
        elif a.lr:
            best_lr, sweep = a.lr, {}
        else:
            sweep = {}
            for lr in LR_GRID:
                st, _, _ = build_student(kind, t, teacher)
                try:
                    d = kd_train(st, teacher, t, lr, SWEEP_STEPS, teacher_raw, f"{kind} lr={lr:g}", eval_every=SWEEP_STEPS)
                    sweep[lr] = d["kd_val"].iloc[-1] if np.isfinite(d["kd_val"].iloc[-1]) else np.inf
                except torch.OutOfMemoryError:
                    sweep[lr] = np.inf
                del st; torch.cuda.empty_cache()
            best_lr = min(sweep, key=sweep.get)
            print(f"{tag}: varredura {sweep} → lr = {best_lr:g}", flush=True)
        st, t_prune, keep = build_student(kind, t, teacher)
        antes = salvo["antes"] if salvo else evaluate_ensemble(st, t)
        d, best = kd_train_v2(st, teacher, t, best_lr, a.steps, teacher_raw, kind, patience=patience,
                              estado_path=estado, extra=dict(lr=best_lr, sweep=sweep, antes=antes))
        ultimo = evaluate_ensemble(st, t)
        st.load_state_dict(best["state"])
        melhor = evaluate_ensemble(st, t)
        info = dict(dataset=name, task=t["task"], metric=PRIMARY_METRIC[t["task"]], seed=SEED, prune_k=PRUNE_K,
                    student=kind, camadas=keep, params=n_params(st), params_teacher=n_params(teacher),
                    lr=best_lr, sweep={f"{k:g}": v for k, v in sweep.items()}, t_poda=t_prune,
                    max_steps=a.steps, patience=a.patience, passos_feitos=int(d.step.iloc[-1]),
                    melhor_passo=best["step"], metric_val_melhor=best["metric_val"],
                    teacher_test_ens=teacher_ens, test_ens_antes=antes,
                    test_ens_ultimo=ultimo, test_ens_melhor=melhor, retomado_nos_passos=best["retomadas"])
        d.assign(dataset=name, student=kind, lr=best_lr, t_poda=t_prune, seed=SEED, prune_k=PRUNE_K).to_csv(out_log, index=False)
        json.dump(info, open(out_sum, "w"), default=float, indent=1)
        estado.unlink(missing_ok=True)
        print(f"{tag}: antes {antes:.4f} | último {ultimo:.4f} | melhor (passo {best['step']}) {melhor:.4f} | teacher {teacher_ens:.4f}", flush=True)
        del st; torch.cuda.empty_cache()
