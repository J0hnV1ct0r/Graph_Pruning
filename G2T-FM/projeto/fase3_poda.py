"""
Fase 3 do projeto para o G2T-LimiX: poda cumulativa das k camadas (k = 1..6) sem re-treino, pelas ordens de
fase3_ordens.py (BI de todos os tokens, BI do token-alvo, ablação individual e 3 aleatórias; camada 0 protegida).

Avalia cada conjunto distinto de camadas uma vez por semente, com o mesmo pipeline da Fase 2
(fase2_bi_ablacao.rodar); conjuntos de 1 camada vêm da ablação da Fase 2.

Uso (da raiz do repositório, ~/venvs/g2tfm): python G2T-FM/projeto/fase3_poda.py --dataset tolokers-2 [--seeds 0 1 2]
Saída: ../Block_Influence_results/g2t_fase3/<dataset>/{plano,avaliacoes}.csv (grava a cada avaliação e retoma)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parents[1]  # G2T-FM/
sys.path.insert(0, str(AQUI.parent / "pipeline"))

import fase2_bi_ablacao as f2  # noqa: E402  (aplica as interceptações do pipeline)
import fase3_ordens as f3  # noqa: E402

RESULTS = AQUI.parent / "Block_Influence_results" / "g2t_fase3"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    out = RESULTS / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    plano = f3.plano("g2t_fase2", a.dataset)
    pd.DataFrame([dict(estrategia=e, k=k, removidas="-".join(map(str, c))) for e, k, c in plano]).to_csv(out / "plano.csv", index=False)

    caminho = out / "avaliacoes.csv"
    feitos = pd.read_csv(caminho, dtype={"removidas": str}) if caminho.exists() else pd.DataFrame(
        columns=["dataset", "seed", "removidas", "k", "val", "test", "origem"])
    conhecida = f3.ablacao_conhecida("g2t_fase2", a.dataset)
    f2.lib.init()
    f2.lib.configure_torch()
    for conj in sorted({c for _, _, c in plano}, key=lambda c: (len(c), c)):
        chave = "-".join(map(str, conj))
        for seed in a.seeds:
            if ((feitos.seed == seed) & (feitos.removidas == chave)).any():
                continue
            if conj in conhecida and seed in conhecida[conj]:
                (val, test), origem = conhecida[conj][seed], "fase2"
            else:
                val, test, _ = f2.rodar(a.dataset, seed, remover=conj)
                origem = "fase3"
            feitos = pd.concat([feitos, pd.DataFrame([dict(dataset=a.dataset, seed=seed, removidas=chave, k=len(conj),
                                                           val=val, test=test, origem=origem)])], ignore_index=True)
            feitos.to_csv(caminho, index=False)
            print(f"{a.dataset} seed={seed} sem {chave}: val={val:.4f} test={test:.4f} ({origem})", flush=True)
