"""Resumo da Fase 2 (BI por camada e ablação individual) com o que já estiver pronto em Block_Influence_results/.

Uso (da raiz do repositório): python resumo_fase2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parent / "Block_Influence_results"
ACASO = {"tolokers-2": 0.218, "amazon-ratings": 0.369, "wiki-cs": 0.230, "city-roads-M": 0.0}  # tabela do EXP-003

for modelo, pasta in [("GraphPFN", "graphfpn_fase2"), ("G2T-LimiX", "g2t_fase2"), ("NodePFN", "nodepfn_fase2")]:
    print(f"== {modelo}")
    for ds in ["tolokers-2", "amazon-ratings", "wiki-cs", "city-roads-M"]:
        d = R / pasta / ds
        if not (d / "block_influence.csv").exists():
            continue
        b = pd.read_csv(d / "block_influence.csv").groupby("layer").mean(numeric_only=True)
        linha = (f"  {ds:14s} menor BI (todos): {list(b.cos_all.sort_values().index[:5])}  "
                 f"menor BI (alvo): {list(b.cos_tgt.sort_values().index[:5])}  "
                 f"BI alvo camadas 0/1: {b.cos_tgt[0]:.2f}/{b.cos_tgt[1]:.2f}")
        print(linha)
        if (d / "ablacao_individual.csv").exists():
            base = pd.read_csv(d / "baseline.csv").set_index("seed")
            ab = pd.read_csv(d / "ablacao_individual.csv")
            c = ACASO[ds]
            ab["ret_val"] = 100 * (ab.val - c) / (ab.seed.map(base.val) - c)
            ab["ret_test"] = 100 * (ab.test - c) / (ab.seed.map(base.test) - c)
            r = ab.groupby("removida")[["ret_val", "ret_test"]].mean()
            if len(r) == 12:
                rho_all = spearmanr(b.cos_all, 100 - r.ret_val).correlation
                rho_tgt = spearmanr(b.cos_tgt, 100 - r.ret_val).correlation
                print(f"  {'':14s} retido no teste sem cada camada 0..11 (%): {r.ret_test.round(0).astype(int).tolist()}")
                print(f"  {'':14s} Spearman BI × importância (validação): todos = {rho_all:.2f}, alvo = {rho_tgt:.2f}")
            else:
                print(f"  {'':14s} ablação parcial: {len(r)}/12 camadas")
