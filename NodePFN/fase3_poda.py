"""
Fase 3 do projeto para o NodePFN: poda cumulativa das k camadas (k = 1..6) sem re-treino, pelas ordens de
fase3_ordens.py (BI de todos os tokens, BI do token-alvo, ablação individual e 3 aleatórias; camada 0 protegida).

Mesmo pipeline e hiperparâmetros da Fase 2 (avaliar_projeto.avaliar com os da Fase 1); conjuntos de 1 camada vêm
da ablação da Fase 2.

Uso (de dentro de NodePFN/): python fase3_poda.py --dataset tolokers-2 [--seeds 0 1 2]
Saída: ../Block_Influence_results/nodepfn_fase3/<dataset>/{plano,avaliacoes}.csv (grava a cada avaliação e retoma)
"""
import argparse
import json
import sys

import pandas as pd
import torch.nn as nn

import avaliar_projeto as ap_

sys.path.insert(0, str(ap_.AQUI.parent))
import fase3_ordens as f3  # noqa: E402

RESULTS = ap_.AQUI.parent / "Block_Influence_results" / "nodepfn_fase3"


def removedor(camadas):
    def preparar(model):
        stack = model.transformer_encoder
        original = stack.layers
        stack.layers = nn.ModuleList([l for i, l in enumerate(original) if i not in camadas])

        def desfazer():
            stack.layers = original
        return desfazer
    return preparar


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(ap_.HP))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    hp = json.load(open(ap_.RESULTS / f"{a.dataset}.json"))["hp"]
    out = RESULTS / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    plano = f3.plano("nodepfn_fase2", a.dataset)
    pd.DataFrame([dict(estrategia=e, k=k, removidas="-".join(map(str, c))) for e, k, c in plano]).to_csv(out / "plano.csv", index=False)

    caminho = out / "avaliacoes.csv"
    feitos = pd.read_csv(caminho, dtype={"removidas": str}) if caminho.exists() else pd.DataFrame(
        columns=["dataset", "seed", "removidas", "k", "val", "test", "origem"])
    conhecida = f3.ablacao_conhecida("nodepfn_fase2", a.dataset)
    for conj in sorted({c for _, _, c in plano}, key=lambda c: (len(c), c)):
        chave = "-".join(map(str, conj))
        for seed in a.seeds:
            if ((feitos.seed == seed) & (feitos.removidas == chave)).any():
                continue
            if conj in conhecida and seed in conhecida[conj]:
                (val, test), origem = conhecida[conj][seed], "fase2"
            else:
                r = ap_.avaliar(a.dataset, seed, "cuda", hp, preparar_modelo=removedor(set(conj)))
                (val, test), origem = (r["metrica_artigo_val"], r["metrica_artigo_test"]), "fase3"
            feitos = pd.concat([feitos, pd.DataFrame([dict(dataset=a.dataset, seed=seed, removidas=chave, k=len(conj),
                                                           val=val, test=test, origem=origem)])], ignore_index=True)
            feitos.to_csv(caminho, index=False)
            print(f"{a.dataset} seed={seed} sem {chave}: val={val:.4f} test={test:.4f} ({origem})", flush=True)
