"""
Fase 3 do projeto (G2T-LimiX e NodePFN): ordens de poda cumulativa sem re-treino, como no EXP-003 do GraphPFN.

Estratégias (camada 0 sempre protegida: nos dois modelos e no GraphPFN, removê-la sozinha derruba o modelo ao acaso):
  - bi_todos: menor BI cosseno de todos os tokens primeiro (média das 3 sementes da Fase 2);
  - bi_alvo: menor BI cosseno do token-alvo primeiro;
  - ablacao: maior desempenho retido na validação ao remover a camada sozinha primeiro (usa rótulos; referência);
  - aleatoria_<r>: 3 permutações aleatórias das camadas 1..11 (semente fixa).
k = 1..6 (6 camadas = configuração do KD do GraphPFN, ~metade dos parâmetros).
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "Block_Influence_results"
ACASO = {"tolokers-2": 0.218, "amazon-ratings": 0.369, "wiki-cs": 0.230, "city-roads-M": 0.0,  # tabela do EXP-003
         "questions": 0.030, "artnet-exp": 0.100}  # fração de positivos no teste (transferência, 2026-11)
KS = range(1, 7)
N_ALEATORIAS = 3


def ordens(pasta_fase2, dataset):
    d = R / pasta_fase2 / dataset
    bi = pd.read_csv(d / "block_influence.csv").groupby("layer").mean(numeric_only=True).drop(index=0)
    base = pd.read_csv(d / "baseline.csv").set_index("seed")
    ab = pd.read_csv(d / "ablacao_individual.csv")
    ab["ret_val"] = (ab.val - ACASO[dataset]) / (ab.seed.map(base.val) - ACASO[dataset])
    ret = ab[ab.removida != 0].groupby("removida").ret_val.mean()
    out = {
        "bi_todos": list(bi.cos_all.sort_values(kind="stable").index),
        "bi_alvo": list(bi.cos_tgt.sort_values(kind="stable").index),
        "ablacao": list(ret.sort_values(ascending=False, kind="stable").index),
    }
    rng = np.random.default_rng(0)
    for r in range(N_ALEATORIAS):
        out[f"aleatoria_{r}"] = list(rng.permutation(np.arange(1, 12)))
    return {nome: [int(c) for c in o] for nome, o in out.items()}


def plano(pasta_fase2, dataset):
    """Lista de (estrategia, k, conjunto ordenado de camadas removidas)."""
    return [(nome, k, tuple(sorted(o[:k]))) for nome, o in ordens(pasta_fase2, dataset).items() for k in KS]


def ablacao_conhecida(pasta_fase2, dataset):
    """Resultados da Fase 2 para conjuntos de 1 camada: {(camada,): {seed: (val, test)}}."""
    ab = pd.read_csv(R / pasta_fase2 / dataset / "ablacao_individual.csv")
    out = {}
    for row in ab.itertuples():
        out.setdefault((int(row.removida),), {})[int(row.seed)] = (row.val, row.test)
    return out
