"""
Avaliação repetida e pareada da poda de camadas (Tier 1 e Tier 2) no AnyGraph.

Corrige dois pontos de evaluate_layer_pruning.ipynb (EXP-002):
  1. Repetições: cada avaliação refaz os projetores com t.svd_lowrank (aleatório). O notebook
     usava 1 repetição; o main.py oficial usa 10. Aqui são N_REPETICOES, e em cada repetição os
     projetores são sorteados uma vez e usados pelos três modelos (comparação pareada). A
     atribuição de expert também usa a mesma semente para os três modelos.
  2. Recarga: no notebook, exp.load_model() antes do Tier 2 carregava os pesos no modelo já
     podado do Tier 1. Aqui os três modelos saem de cópias do mesmo checkpoint original.

Roda um dataset por vez para limitar a memória e para se a RAM disponível cair abaixo do limite.

Uso (da raiz do repositório):
    python Notebooks_pruning/evaluate_layer_pruning_repetido.py --checkpoint pretrain_link2 --dataset cora
    python Notebooks_pruning/evaluate_layer_pruning_repetido.py --resumo
"""
import argparse
import os
import random
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "Block_Influence_results", "layer_pruning_evaluation_repetido")
DATASETS = ["cora", "arxiv", "pubmed", "home", "tech"]
CONFIGS = {
    "Modelo Inteiro (Baseline)": [],
    "Poda Tier 1 (Sem [5, 6])": [5, 6],
    "Poda Tier 2 (Sem [4, 5, 6, 7])": [4, 5, 6, 7],
}


def vigia_memoria(limite_gb):
    """Encerra o processo se a RAM disponível da máquina ficar abaixo do limite."""
    while True:
        disp = psutil.virtual_memory().available / 2**30
        if disp < limite_gb:
            print(f"\n[vigia] RAM disponível {disp:.1f} GB < {limite_gb} GB: encerrando para não afetar outros processos.", flush=True)
            os._exit(3)
        time.sleep(0.5)


def fixar_semente(seed):
    import torch as t
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def avaliar(checkpoint, dataset, n_rep):
    import torch as t

    os.chdir(os.path.join(REPO_ROOT, "AnyGraph"))
    sys.path.insert(0, os.getcwd())
    sys.argv = ["pruning_evaluation"]

    from node_classification.params import args
    if t.cuda.is_available():
        args.gpu = "0"
        args.devices = ["cuda:0", "cuda:0"]
    else:
        args.gpu = "-1"
        args.devices = ["cpu", "cpu"]
    # Mesma configuração de evaluate_layer_pruning.ipynb
    args.tst_mode = "tst"
    args.trn_mode = "train-all"
    args.shot = 0
    args.proj_method = "both"
    args.gnn_layer = 3
    args.latdim = 512
    args.load_model = checkpoint

    from node_classification.data_handler import MultiDataHandler
    from node_classification.main import Exp
    from Pruning_Methods.layer_pruner import BlockInfluenceLayerPruner

    multi_handler = MultiDataHandler([dataset], [[dataset]])
    handler = multi_handler.tst_handlers_group[0][0]
    exp = Exp(multi_handler)
    exp.prepare_model()

    # Carrega o checkpoint com strict=True para garantir que todos os pesos entraram.
    # mmap=True: o .mod tem 3–7,5 GB (a maior parte são tensores fora do state_dict) e assim
    # os dados ficam mapeados do disco em vez de copiados para a RAM.
    ckp = t.load(f"./Models/{checkpoint}.mod", map_location="cpu", weights_only=False, mmap=True)
    exp.model.load_state_dict(ckp["model"].state_dict(), strict=True)
    del ckp
    base = exp.model

    modelos = {}
    for nome, camadas in CONFIGS.items():
        if camadas:
            pruner = BlockInfluenceLayerPruner(base)
            modelos[nome] = pruner.prune_specific_layers({i: camadas for i in range(len(base.experts))})
        else:
            modelos[nome] = base
        n_camadas = {len(e.trainable_nn.dense_layers) for e in modelos[nome].experts}
        n_params = sum(p.numel() for p in modelos[nome].parameters()) / 1e6
        print(f"{nome}: camadas por expert = {n_camadas}, parâmetros = {n_params:.2f} M", flush=True)

    linhas = []
    for rep in range(n_rep):
        fixar_semente(rep)
        handler.make_projectors()
        for nome, modelo in modelos.items():
            exp.model = modelo
            fixar_semente(10_000 + rep)
            modelo.assign_experts([handler], reca=False, log_assignment=False)
            res = exp.test_epoch(handler, 0)
            linhas.append({
                "Checkpoint": checkpoint, "Dataset": dataset, "Repetição": rep,
                "Configuração": nome, "Expert": modelo.assignment[0],
                "Acurácia": res["Acc"], "F1-Score": res["F1"], "tstNum": res["tstNum"],
            })
            print(f"\nrep {rep} | {nome}: acc={res['Acc']:.4f} f1={res['F1']:.4f} expert={linhas[-1]['Expert']}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    caminho = os.path.join(OUT_DIR, f"bruto_{checkpoint}_{dataset}.csv")
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    print(f"Salvo: {caminho}", flush=True)


def resumir():
    from scipy import stats

    arquivos = [f for f in os.listdir(OUT_DIR) if f.startswith("bruto_")]
    df = pd.concat([pd.read_csv(os.path.join(OUT_DIR, f)) for f in arquivos], ignore_index=True)

    # OVERALL por repetição: média ponderada pelo nº de nós de teste, como no Exp.evaluate()
    completos = df.groupby(["Checkpoint"])["Dataset"].nunique()
    overall = []
    for ckpt in completos[completos == len(DATASETS)].index:
        sub = df[df["Checkpoint"] == ckpt]
        for (rep, cfg), g in sub.groupby(["Repetição", "Configuração"]):
            w = g["tstNum"] / g["tstNum"].sum()
            overall.append({"Checkpoint": ckpt, "Dataset": "OVERALL", "Repetição": rep, "Configuração": cfg,
                            "Acurácia": (w * g["Acurácia"]).sum(), "F1-Score": (w * g["F1-Score"]).sum()})
    df = pd.concat([df, pd.DataFrame(overall)], ignore_index=True)

    linhas = []
    base_nome = "Modelo Inteiro (Baseline)"
    for (ckpt, ds), g in df.groupby(["Checkpoint", "Dataset"]):
        base = g[g["Configuração"] == base_nome].set_index("Repetição")
        for cfg in CONFIGS:
            c = g[g["Configuração"] == cfg].set_index("Repetição")
            linha = {"Checkpoint": ckpt, "Dataset": ds, "Configuração": cfg, "n": len(c)}
            for met in ["Acurácia", "F1-Score"]:
                linha[f"{met} média"] = c[met].mean()
                linha[f"{met} dp"] = c[met].std(ddof=1)
                if cfg != base_nome:
                    d = (c[met] - base.loc[c.index, met]).to_numpy()
                    meia = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                    linha[f"Δ {met} (podado − inteiro)"] = d.mean()
                    linha[f"Δ {met} IC95% inf"] = d.mean() - meia
                    linha[f"Δ {met} IC95% sup"] = d.mean() + meia
            linhas.append(linha)
    res = pd.DataFrame(linhas)
    caminho = os.path.join(OUT_DIR, "resumo_pareado.csv")
    res.to_csv(caminho, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(res.round(4).to_string(index=False))
    print(f"Salvo: {caminho}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["pretrain_link1", "pretrain_link2"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--repeticoes", type=int, default=10)
    parser.add_argument("--limite-ram-gb", type=float, default=3.0)
    parser.add_argument("--resumo", action="store_true")
    a = parser.parse_args()
    if a.resumo:
        resumir()
    else:
        threading.Thread(target=vigia_memoria, args=(a.limite_ram_gb,), daemon=True).start()
        avaliar(a.checkpoint, a.dataset, a.repeticoes)
