"""Resumo da Fase 4 (KD, student podado × aleatório) com o que já estiver pronto em Block_Influence_results/.

Para cada modelo, dataset e student: métrica no teste (ensemble) do professor, do student antes do KD e do student
no melhor ponto pela validação, média das sementes; "% prof." = 100·student/professor (métrica bruta, como no
EXP-004 do GraphPFN).

Uso (da raiz do repositório): python resumo_fase4.py
Saída: Block_Influence_results/{g2t,nodepfn}_fase4_kd/resumo_fase4.csv
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "Block_Influence_results"

for modelo, pasta in [("G2T-LimiX", "g2t_fase4_kd"), ("NodePFN", "nodepfn_fase4_kd")]:
    linhas = [json.loads(f.read_text()) for f in sorted((R / pasta).glob("kd_resumo_*.json"))]
    if not linhas:
        continue
    d = pd.DataFrame(linhas)
    tab = (d.groupby(["dataset", "metric", "student"])
           .agg(n=("seed", "count"), professor=("teacher_test_ens", "mean"), antes=("test_ens_antes", "mean"),
                depois=("test_ens_melhor", "mean"), dp_depois=("test_ens_melhor", "std"),
                passos=("passos_feitos", "mean"), params=("params", "first"), params_prof=("params_teacher", "first"))
           .reset_index())
    tab["pct_prof"] = 100 * tab.depois / tab.professor
    tab.insert(0, "modelo", modelo)
    tab.to_csv(R / pasta / "resumo_fase4.csv", index=False)
    print(f"== {modelo}")
    print(tab.drop(columns=["modelo"]).round(4).to_string(index=False))
