"""Resumo unificado do KD (student podado × aleatório) nos 4 modelos, com a MESMA normalização da poda:
retido = 100·(student − acaso)/(professor − acaso), calculado por semente e depois agregado (média ± dp). A métrica
bruta e o "% do professor" (student ÷ professor, como no EXP-004) também aparecem, para comparação.

Fontes: graphfpn_kd_v2/ (GraphPFN), g2t_fase4_kd/ (G2T-LimiX), nodepfn_fase4_kd/ (NodePFN), anygraph_kd/ (AnyGraph,
metade B do teste, acaso por semente). "config" = sufixo do arquivo: 200p4 = protocolo principal (até 200 passos,
paciência 4); 200p0 / 1000p0 = sensibilidade sem parada antecipada; _ctx900 = contexto de treino limitado.

Uso (da raiz do worktree): python resumo_kd.py
Saída: Block_Influence_results/resumo_kd_todos.csv
"""
import json
import re
from pathlib import Path

import pandas as pd

from fase3_ordens import ACASO, R

linhas = []
for modelo, pasta in [("GraphPFN", "graphfpn_kd_v2"), ("G2T-LimiX", "g2t_fase4_kd"), ("NodePFN", "nodepfn_fase4_kd")]:
    for f in sorted((R / pasta).glob("kd_resumo_*.json")):
        d = json.loads(f.read_text())
        if d.get("prune_k", 6) != 6:
            continue
        cfg = re.sub(r"^.*?_(Podado|Aleatório)_", "", f.stem)
        linhas.append(dict(modelo=modelo, dataset=d["dataset"], seed=d["seed"], student=d["student"], config=cfg,
                           professor=d["teacher_test_ens"], antes=d["test_ens_antes"], depois=d["test_ens_melhor"],
                           acaso=ACASO[d["dataset"]], passos=d["passos_feitos"], melhor_passo=d["melhor_passo"],
                           params=d["params"], params_prof=d["params_teacher"]))
for f in sorted((R / "anygraph_kd").glob("kd_*.json")) if (R / "anygraph_kd").exists() else []:
    d = json.loads(f.read_text())
    linhas.append(dict(modelo="AnyGraph link2", dataset=d["dataset"], seed=d["seed"], student=d["student"],
                       config=f.stem.split("_")[-1], professor=d["professor"]["test"], antes=d["antes"]["test"],
                       depois=d["melhor"]["test"], acaso=d["professor"]["acaso_test"], passos=d["passos"],
                       melhor_passo=d["passo_melhor"], params=d["params_student_M"] * 1e6, params_prof=d["params_professor_M"] * 1e6))

d = pd.DataFrame(linhas)
d["retido_antes"] = 100 * (d.antes - d.acaso) / (d.professor - d.acaso)
d["retido"] = 100 * (d.depois - d.acaso) / (d.professor - d.acaso)
d["pct_prof"] = 100 * d.depois / d.professor
tab = (d.groupby(["modelo", "dataset", "config", "student"])
       .agg(n=("seed", "count"), professor=("professor", "mean"), acaso=("acaso", "first"), antes=("antes", "mean"),
            depois=("depois", "mean"), dp=("depois", "std"), retido_antes=("retido_antes", "mean"),
            retido=("retido", "mean"), retido_dp=("retido", "std"), pct_prof=("pct_prof", "mean"),
            passos=("passos", "mean"))
       .reset_index())
red = d.groupby(["modelo", "student"]).apply(lambda g: 100 * (1 - g.params.iloc[0] / g.params_prof.iloc[0])).rename("reducao_params_pct")
tab = tab.merge(red.reset_index(), on=["modelo", "student"], how="left")
tab.to_csv(R / "resumo_kd_todos.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", None):
    print(tab.round(3).to_string(index=False))
