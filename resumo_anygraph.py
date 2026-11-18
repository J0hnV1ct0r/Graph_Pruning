"""Resumo das Fases 2–3, do custo e do KD do AnyGraph (Notebooks_pruning/anygraph_fases.py e anygraph_kd.py), no
formato dos PFNs.

Retido = 100·(média podado − acaso)/(média inteiro − acaso) por dataset, com o baseline da MESMA execução e as mesmas
10 repetições (o AnyGraph não é determinístico entre execuções na GPU; dentro de uma execução os projetores são os
mesmos para todas as configurações). Razão das médias, e não média das razões: em algumas repetições do tech o
inteiro fica perto do acaso e a razão por repetição explode. Reportado na metade B dos nós de teste ("test"); a metade A ("val") só escolhe a ordem da ablação.
"Média" = média dos 5 datasets; OVERALL = acurácia ponderada pelo nº de nós de teste, como no EXP-002.

Uso (da raiz do worktree): python resumo_anygraph.py
Saída: Block_Influence_results/anygraph_fase3/<ckpt>/curvas_poda.csv, resumo_anygraph.csv, resumo_kd_anygraph.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parent / "Block_Influence_results"
DATASETS = ["cora", "arxiv", "pubmed", "home", "tech"]


def ler(pasta, ck):
    fs = [R / pasta / ck / f"{d}.csv" for d in DATASETS]
    fs = [f for f in fs if f.exists()]
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, dtype={"removidas": str}, keep_default_na=False) for f in fs], ignore_index=True)
    base = d[d.removidas == ""].set_index(["dataset", "rep"])
    idx = list(zip(d.dataset, d.rep))
    for s in ["val", "test", "full"]:
        b = base.loc[idx, f"acc_{s}"].values
        d[f"base_{s}"] = b
        d[f"dacc_{s}"] = d[f"acc_{s}"] - b
    return d


def retido(d, chaves, s):
    g = d.groupby(chaves)[[f"acc_{s}", f"base_{s}", f"acaso_{s}"]].mean()
    return 100 * (g[f"acc_{s}"] - g[f"acaso_{s}"]) / (g[f"base_{s}"] - g[f"acaso_{s}"])


linhas = []
for ck in ["pretrain_link2", "pretrain_link1"]:
    f2 = ler("anygraph_fase2", ck)
    if f2 is None:
        continue
    print(f"===== AnyGraph {ck}")
    base = f2[f2.removidas == ""].groupby("dataset")[["acc_test", "acaso_test", "acc_full"]].mean()
    print("inteiro (metade B / teste inteiro) e acaso:\n", base.round(4).to_string())
    ab = f2[f2.k == 1]
    tab = retido(ab, ["dataset", "removidas"], "test").unstack("removidas").round(0)
    tab.loc["média"] = tab.mean().round(0)
    print("retido na metade B (%) ao remover cada camada:\n", tab.to_string())
    bi = pd.concat([pd.read_csv(R / "cosseno" / f"block_influence_cosseno_{ck}_{d}_raw.csv") for d in ["cora", "pubmed"]])
    bi = bi.groupby("Layer")["Block Influence (Cosseno)"].mean()
    imp = 100 - retido(ab, ["dataset", "removidas"], "val").groupby("removidas").mean()
    imp.index = imp.index.astype(int)
    print(f"Spearman BI × importância (metade A, média dos datasets): {spearmanr(bi.loc[imp.index], imp).correlation:.2f}")

    f3 = ler("anygraph_fase3", ck)
    if f3 is None:
        continue
    ident = f3[f3.k == 8]
    if len(ident):
        r = retido(ident, ["dataset"], "test")
        print("controle identidade (as 8 camadas removidas, só projetores): retido na metade B (%)\n",
              r.round(0).to_string(), f"\n média {r.mean():.0f}")
        r.rename("ret_identidade").to_csv(R / "anygraph_fase3" / ck / "controle_identidade.csv")
    plano = pd.read_csv(R / "anygraph_fase3" / ck / "plano.csv", dtype={"removidas": str})
    todas = pd.concat([f2[f2.k == 1], f3[f3.k > 1]], ignore_index=True)
    por_ds = todas.groupby(["removidas", "dataset"]).agg(acc=("acc_test", "mean"), dacc=("dacc_full", "mean"),
                                                        n=("n_full", "first")).reset_index()
    por_ds = por_ds.merge(retido(todas, ["removidas", "dataset"], "test").rename("ret").reset_index())
    overall = por_ds.groupby("removidas").apply(lambda g: np.average(g.dacc, weights=g.n)).rename("dacc_overall_full")
    m = plano.merge(por_ds, on="removidas", how="left")
    cur = m.groupby(["estrategia", "k", "removidas"]).agg(ret_media=("ret", "mean"), n_ds=("dataset", "nunique")).reset_index()
    cur = cur.merge(overall, left_on="removidas", right_index=True, how="left")
    cur.insert(0, "checkpoint", ck)
    cur.to_csv(R / "anygraph_fase3" / ck / "curvas_poda.csv", index=False)
    cur["grupo"] = cur.estrategia.where(~cur.estrategia.str.startswith("aleatoria"), "aleatoria (mediana)")
    print("retido médio na metade B (%) por k:\n", cur.groupby(["grupo", "k"]).ret_media.median().unstack("k").round(0).to_string())
    for _, r in cur.iterrows():
        linhas.append(r.drop("grupo").to_dict())
    c = R / "anygraph_custo" / f"{ck}.csv"
    if c.exists():
        print("custo:\n", pd.read_csv(c).round(3).to_string(index=False))
if linhas:
    pd.DataFrame(linhas).to_csv(R / "anygraph_fase3" / "resumo_anygraph.csv", index=False)

kd = sorted((R / "anygraph_kd").glob("kd_*.json")) if (R / "anygraph_kd").exists() else []
if kd:
    d = pd.DataFrame([json.load(open(f)) for f in kd])
    d["config"] = [Path(f).stem.split("_")[-1] for f in kd]
    d["prof"] = d.professor.map(lambda x: x["test"]); d["acaso"] = d.professor.map(lambda x: x["acaso_test"])
    d["antes_t"] = d.antes.map(lambda x: x["test"]); d["depois"] = d.melhor.map(lambda x: x["test"])
    d["pct_prof"] = 100 * d.depois / d.prof
    d["retido"] = 100 * (d.depois - d.acaso) / (d.prof - d.acaso)
    d["retido_antes"] = 100 * (d.antes_t - d.acaso) / (d.prof - d.acaso)
    g = d.groupby(["config", "dataset", "student"])[["prof", "antes_t", "depois", "pct_prof", "retido_antes", "retido"]].mean()
    print("===== KD AnyGraph (link2, Tier 2), metade B, média de 3 sementes\n", g.round(3).to_string())
    g.reset_index().to_csv(R / "anygraph_kd" / "resumo_kd_anygraph.csv", index=False)
