"""
AnyGraph no mesmo formato de análise dos PFNs (Fases 2 e 3 do EXP-006/EXP-007), para o TCC ficar comparável:
  - ablação individual (remover cada camada l = 0..7, a mesma em todos os experts);
  - poda cumulativa k = 1..6 pelas ordens: BI cosseno (EXP-001), ablação (referência; usa rótulos) e 3 permutações
    aleatórias; camada 0 protegida, como nos PFNs;
  - custo: parâmetros e latência por k.

Avaliação como no EXP-002 reavaliado (evaluate_layer_pruning_repetido.py): zero-shot, 10 repetições pareadas
(projetores com semente r e atribuição de expert com semente 10.000 + r, iguais para todas as configurações).

O AnyGraph não tem conjunto de validação no zero-shot. Para que a ordem por ablação não seja escolhida no mesmo
conjunto em que é reportada, os nós de teste de cada dataset são divididos ao meio (semente 0): metade A ("val")
escolhe a ordem da ablação; metade B ("test") é onde todas as estratégias são reportadas. O teste inteiro também é
gravado ("full"), comparável ao EXP-002.

Uso (da raiz do repositório):
    python AnyGraph/projeto/anygraph_fases.py --checkpoint pretrain_link2 --dataset cora --fase 2
    python AnyGraph/projeto/anygraph_fases.py --checkpoint pretrain_link2 --dataset cora --fase 3
    python AnyGraph/projeto/anygraph_fases.py --checkpoint pretrain_link2 --custo
Saída: Block_Influence_results/anygraph_fase{2,3}/<checkpoint>/<dataset>.csv, anygraph_custo/<checkpoint>.csv
"""
import argparse
import copy
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

RAIZ = Path(__file__).resolve().parents[2]
ANYGRAPH = RAIZ / "AnyGraph"  # pesos (.mod) e Datasets/
R = RAIZ / "Block_Influence_results"
DATASETS = ["cora", "arxiv", "pubmed", "home", "tech"]
N_LAYERS = 8
KS = range(1, 7)
N_ALEATORIAS = 3


def vigia_memoria(limite_gb):
    while True:
        if psutil.virtual_memory().available / 2**30 < limite_gb:
            print(f"[vigia] RAM disponível < {limite_gb} GB: encerrando", flush=True)
            os._exit(3)
        time.sleep(0.5)


def fixar_semente(seed):
    import torch as t
    random.seed(seed); np.random.seed(seed); t.manual_seed(seed); t.cuda.manual_seed_all(seed)


def preparar(checkpoint, dataset):
    import torch as t
    os.chdir(ANYGRAPH)
    sys.path.insert(0, str(ANYGRAPH))
    sys.argv = ["anygraph_fases"]
    from node_classification.params import args
    args.gpu, args.devices = "0", ["cuda:0", "cuda:0"]
    args.tst_mode, args.trn_mode, args.shot = "tst", "train-all", 0
    args.proj_method, args.gnn_layer, args.latdim, args.load_model = "both", 3, 512, checkpoint
    from node_classification.data_handler import MultiDataHandler
    from node_classification.main import Exp
    mh = MultiDataHandler([dataset], [[dataset]])
    handler = mh.tst_handlers_group[0][0]
    exp = Exp(mh)
    exp.prepare_model()
    ckp = t.load(f"./Models/{checkpoint}.mod", map_location="cpu", weights_only=False, mmap=True)
    exp.model.load_state_dict(ckp["model"].state_dict(), strict=True)
    del ckp
    return exp, handler


def podar(base, remover):
    """Cópia do modelo sem as camadas `remover` (mesmas em todos os experts), como BlockInfluenceLayerPruner."""
    import torch.nn as nn
    if not remover:
        return base
    m = copy.deepcopy(base)
    rem = set(remover)
    for e in m.experts:  # remove a camada densa e a LayerNorm dela, como BlockInfluenceLayerPruner
        mlp = e.trainable_nn
        mlp.dense_layers = nn.Sequential(*[l for i, l in enumerate(mlp.dense_layers) if i not in rem])
        mlp.layer_norms = nn.Sequential(*[l for i, l in enumerate(mlp.layer_norms) if i not in rem])
    return m


def predizer(exp, handler, modelo):
    """Mesmo laço de Exp.test_epoch, devolvendo predições e rótulos por nó."""
    import torch as t
    from node_classification.params import args
    exp.model = modelo
    loader = handler.tst_loader
    class_num = loader.dataset.class_num
    modelo.eval()
    expert = modelo.summon(0)
    P, Y = [], []
    with t.no_grad():
        for i, (nodes, labels) in enumerate(loader):
            nodes, labels = nodes.long().to(args.devices[0]), labels.long().to(args.devices[0])
            P.append(expert.pred_for_node_test(nodes, class_num, handler.projectors, rerun_embed=(i == 0)).cpu())
            Y.append(labels.cpu())
    expert.final_embeds = None  # o cache de embeddings fica em cada modelo podado e estoura a GPU no arxiv
    t.cuda.empty_cache()
    return t.cat(P).numpy(), t.cat(Y).numpy()


def metade_a(n):
    a = np.zeros(n, bool)
    a[np.random.default_rng(0).permutation(n)[: n // 2]] = True
    return a


def avaliar_conjuntos(checkpoint, dataset, conjuntos, caminho, n_rep=10):
    """Avalia cada conjunto de camadas removidas nas n_rep repetições pareadas; grava a cada repetição e retoma."""
    from sklearn.metrics import f1_score
    exp, handler = preparar(checkpoint, dataset)
    base = exp.model
    feitos = pd.read_csv(caminho, dtype={"removidas": str}, keep_default_na=False) if caminho.exists() else pd.DataFrame()
    feito = set(zip(feitos.removidas, feitos.rep)) if len(feitos) else set()
    modelos = {c: podar(base, c) for c in conjuntos}
    linhas = []
    for rep in range(n_rep):
        pend = [c for c in conjuntos if ("-".join(map(str, c)), rep) not in feito]
        if not pend:
            continue
        fixar_semente(rep)
        handler.make_projectors()
        for c in pend:
            fixar_semente(10_000 + rep)
            modelos[c].assign_experts([handler], reca=False, log_assignment=False)
            p, y = predizer(exp, handler, modelos[c])
            a = metade_a(len(y))
            row = dict(checkpoint=checkpoint, dataset=dataset, removidas="-".join(map(str, c)), k=len(c), rep=rep,
                       expert=int(modelos[c].assignment[0]), n_full=len(y), n_val=int(a.sum()), n_test=int((~a).sum()))
            for nome, m in [("full", np.ones_like(a)), ("val", a), ("test", ~a)]:
                row[f"acc_{nome}"] = float((p[m] == y[m]).mean())
                row[f"f1_{nome}"] = float(f1_score(y[m], p[m], average="macro"))
                row[f"acaso_{nome}"] = float(np.bincount(y[m]).max() / m.sum())
            linhas.append(row)
            print(f"{checkpoint} {dataset} rep {rep} sem [{row['removidas']}]: acc full={row['acc_full']:.4f} "
                  f"val={row['acc_val']:.4f} test={row['acc_test']:.4f} expert={row['expert']}", flush=True)
        pd.concat([feitos, pd.DataFrame(linhas)], ignore_index=True).to_csv(caminho, index=False)
    return pd.read_csv(caminho, dtype={"removidas": str}, keep_default_na=False)


def ordem_bi(checkpoint):
    """Menor BI cosseno primeiro, média dos 8 experts e de cora + pubmed (EXP-001); camada 0 fora."""
    bi = pd.concat([pd.read_csv(R / "cosseno" / f"block_influence_cosseno_{checkpoint}_{d}_raw.csv") for d in ["cora", "pubmed"]])
    m = bi.groupby("Layer")["Block Influence (Cosseno)"].mean().drop(index=0)
    return [int(i) for i in m.sort_values(kind="stable").index]


def ordem_ablacao(checkpoint):
    """Maior acurácia retida na metade A ("val") ao remover a camada sozinha, média dos 5 datasets e 10 repetições."""
    d = pd.concat([pd.read_csv(R / "anygraph_fase2" / checkpoint / f"{ds}.csv", dtype={"removidas": str}, keep_default_na=False) for ds in DATASETS])
    base = d[d.removidas == ""].set_index(["dataset", "rep"])
    d = d[d.k == 1].copy()
    idx = list(zip(d.dataset, d.rep))
    d["ret_val"] = (d.acc_val - d.acaso_val) / (base.loc[idx, "acc_val"].values - d.acaso_val)
    r = d.groupby("removidas").ret_val.mean().drop(index="0")
    return [int(i) for i in r.sort_values(ascending=False, kind="stable").index]


def ordens(checkpoint):
    out = {"bi": ordem_bi(checkpoint), "ablacao": ordem_ablacao(checkpoint)}
    rng = np.random.default_rng(0)
    for r in range(N_ALEATORIAS):
        out[f"aleatoria_{r}"] = [int(i) for i in rng.permutation(np.arange(1, N_LAYERS))]
    return out


def custo(checkpoint):
    """Parâmetros totais e do expert ativo, e latência (mediana de 5) de uma passada de teste no arxiv (o maior)."""
    import torch as t
    exp, handler = preparar(checkpoint, "arxiv")
    base = exp.model
    o = ordem_bi(checkpoint)
    linhas = []
    for k in [0] + list(KS):
        m = podar(base, tuple(sorted(o[:k])))
        fixar_semente(10_000)
        m.assign_experts([handler], reca=False, log_assignment=False)
        tempos = []
        for _ in range(6):
            t.cuda.synchronize(); tic = time.time()
            predizer(exp, handler, m)
            t.cuda.synchronize(); tempos.append(time.time() - tic)
        e = m.experts[m.assignment[0]]
        linhas.append(dict(checkpoint=checkpoint, k=k, camadas=N_LAYERS - k,
                           params_total_M=sum(p.numel() for p in m.parameters()) / 1e6,
                           params_expert_M=sum(p.numel() for p in e.parameters()) / 1e6,
                           latencia_s=float(np.median(tempos[1:]))))
        print(linhas[-1], flush=True)
    out = R / "anygraph_custo"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(out / f"{checkpoint}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=["pretrain_link1", "pretrain_link2"], required=True)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--fase", type=int, choices=[2, 3])
    ap.add_argument("--custo", action="store_true")
    ap.add_argument("--limite-ram-gb", type=float, default=4.0)
    a = ap.parse_args()
    threading.Thread(target=vigia_memoria, args=(a.limite_ram_gb,), daemon=True).start()
    if a.custo:
        custo(a.checkpoint)
        sys.exit(0)
    pasta = R / f"anygraph_fase{a.fase}" / a.checkpoint
    pasta.mkdir(parents=True, exist_ok=True)
    if a.fase == 2:
        conjuntos = [()] + [(i,) for i in range(N_LAYERS)]
    else:
        o = ordens(a.checkpoint)
        plano = [(e, k, tuple(sorted(v[:k]))) for e, v in o.items() for k in KS]
        pd.DataFrame([dict(estrategia=e, k=k, removidas="-".join(map(str, c))) for e, k, c in plano]).to_csv(
            pasta / "plano.csv", index=False)
        conjuntos = [()] + sorted({c for _, _, c in plano} - {(i,) for i in range(N_LAYERS)}, key=lambda c: (len(c), c))
        conjuntos.append(tuple(range(N_LAYERS)))  # controle: sem nenhuma camada (só projetores; a MLP vira identidade)
    avaliar_conjuntos(a.checkpoint, a.dataset, conjuntos, pasta / f"{a.dataset}.csv")
