"""Resumo das fases do Ephris com o que já estiver pronto, no formato dos resumos dos outros PFNs.

Retido = 100·(s − acaso)/(inteiro − acaso), pareado por semente; acaso = fração de positivos (AP) ou classe
majoritária (acurácia) no teste (na validação para a coluna ret_val). Ordens aleatórias resumidas pela mediana.

Uso: python projeto/resumo.py
Saídas: Block_Influence_results/ephris_fase2/resumo_fase2.csv, ephris_fase3/resumo_fase3.csv,
        ephris_fase4_kd/resumo_kd.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from comum import DATASETS, R

pd.set_option("display.width", 200)

# Fase 1
f1 = pd.read_csv(R / "ephris_fase1" / "modelo_inteiro.csv")
print("== Fase 1: modelo inteiro (teste, média ± dp de 3 sementes)")
print(f1.groupby("dataset").agg(test=("test", "mean"), dp=("test", "std"), acaso=("acaso", "first"),
                                nos=("n_nos", "first"), pico_gb=("pico_gb", "max")).round(4).to_string())

# Fase 2
linhas2, linhas3 = [], []
for ds in DATASETS:
    d2, d3 = R / "ephris_fase2" / ds, R / "ephris_fase3" / ds
    if not (d2 / "ablacao_individual.csv").exists():
        continue
    base = pd.read_csv(d2 / "baseline.csv").set_index("seed")
    bi = pd.read_csv(d2 / "block_influence.csv")
    ab = pd.read_csv(d2 / "ablacao_individual.csv")
    ct, cv = base.acaso_test.mean(), base.acaso_val.mean()
    ab["ret_test"] = 100 * (ab.test - ct) / (ab.seed.map(base.test) - ct)
    ab["ret_val"] = 100 * (ab.val - cv) / (ab.seed.map(base.val) - cv)
    imp = ab.groupby("removida")[["ret_test", "ret_val"]].mean()
    b = bi.groupby("layer")[["cos_all", "cos_tgt", "ang_all"]].mean()
    t_ = b.join(imp)
    # Importância = perda de retido na validação ao remover a camada; ρ entre BI e importância (camadas 1..9)
    s = t_.drop(index=0)
    rho_all = spearmanr(s.cos_all, 100 - s.ret_val).statistic
    rho_tgt = spearmanr(s.cos_tgt, 100 - s.ret_val).statistic
    rho_ca = spearmanr(b.cos_all, b.ang_all).statistic
    print(f"\n== Fase 2 · {ds}: inteiro {base.test.mean():.4f} (acaso {ct:.3f}); ρ(BI todos, importância) = "
          f"{rho_all:.2f}, ρ(BI alvo, importância) = {rho_tgt:.2f}; ρ(cos, ang) = {rho_ca:.3f}")
    print(t_.round(3).T.to_string())
    for layer, r in t_.iterrows():
        linhas2.append(dict(dataset=ds, layer=layer, **r.to_dict(), rho_todos=rho_all, rho_alvo=rho_tgt))

    if (d3 / "poda.csv").exists():
        p = pd.read_csv(d3 / "poda.csv", dtype={"removidas": str})
        p["ret_test"] = 100 * (p.test - ct) / (p.seed.map(base.test) - ct)
        cur = p.groupby(["estrategia", "k", "removidas"]).agg(test=("test", "mean"), ret_test=("ret_test", "mean"))
        cur = cur.reset_index()
        cur["grupo"] = cur.estrategia.where(~cur.estrategia.str.startswith("aleatoria"), "aleatoria (mediana)")
        tab = cur.groupby(["grupo", "k"]).ret_test.median().unstack("k").round(0)
        print(f"-- Fase 3 · {ds}: retido no teste (%) por k (camadas MP_ICL removidas de 10)")
        print(tab.to_string())
        for _, r in cur.iterrows():
            linhas3.append(dict(dataset=ds, inteiro=base.test.mean(), acaso=ct, **r.drop("grupo").to_dict()))
if linhas2:
    pd.DataFrame(linhas2).to_csv(R / "ephris_fase2" / "resumo_fase2.csv", index=False)
if linhas3:
    pd.DataFrame(linhas3).to_csv(R / "ephris_fase3" / "resumo_fase3.csv", index=False)

# Fase 4
kd = R / "ephris_fase4_kd"
if kd.exists():
    rs = [json.load(open(f)) for f in sorted(kd.glob("kd_resumo_*.json"))]
    if rs:
        k = pd.DataFrame(rs)
        k["protocolo"] = k.max_steps.astype(str) + "p" + k.patience.astype(str)
        for c in ["antes", "melhor"]:
            k[f"ret_{c}"] = 100 * (k[f"test_{c}"] - k.acaso_test) / (k.teacher_test - k.acaso_test)
        k["pct_prof"] = 100 * k.test_melhor / k.teacher_test
        k.to_csv(kd / "resumo_kd.csv", index=False)
        print("\n== Fase 4: KD (retido no teste, %, média das sementes)")
        print(k.groupby(["dataset", "protocolo", "student"]).agg(
            n=("seed", "count"), professor=("teacher_test", "mean"), antes=("ret_antes", "mean"),
            depois=("ret_melhor", "mean"), pct_prof=("pct_prof", "mean"), passo=("melhor_passo", "mean"),
            t_kd_s=("t_kd", "mean")).round(1).to_string())
