"""
Custo da poda no NodePFN, medido como em custo_g2t.py e custo_graphpfn.py: tolokers-2, semente 0, pipeline completo
da Fase 2 (avaliar_projeto.avaliar com os hiperparâmetros da Fase 1), k = 0, 2, 4, 6 pela ordem bi_todos (camada 0
protegida); parâmetros, latência (mediana de 3 depois de 1 aquecimento) e pico de memória da GPU.

Uso (da raiz do repositório, ~/venvs/nodepfn): python NodePFN/projeto/custo_nodepfn.py
Saída: ../Block_Influence_results/custo_pfns/nodepfn.csv
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

import avaliar_projeto as ap_
from fase3_poda import removedor

sys.path.insert(0, str(ap_.AQUI.parent / "pipeline"))
import fase3_ordens as f3  # noqa: E402

PARAMS = {}


def contando(camadas):
    prep = removedor(camadas)

    def preparar(model):
        desfazer = prep(model)
        PARAMS["n"] = sum(p.numel() for p in model.parameters())
        return desfazer
    return preparar


if __name__ == "__main__":
    ds = "tolokers-2"
    hp = json.load(open(ap_.RESULTS / f"{ds}.json"))["hp"]
    ordem = f3.ordens("nodepfn_fase2", ds)["bi_todos"]
    linhas = []
    for k in [0, 2, 4, 6]:
        rem = tuple(sorted(ordem[:k]))
        ap_.avaliar(ds, 0, "cuda", hp, preparar_modelo=contando(rem))  # aquecimento
        tempos, picos = [], []
        for _ in range(3):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            r = ap_.avaliar(ds, 0, "cuda", hp, preparar_modelo=contando(rem))
            torch.cuda.synchronize()
            tempos.append(time.time() - tic); picos.append(torch.cuda.max_memory_allocated() / 2**30)
        linhas.append(dict(modelo="NodePFN", dataset=ds, k=k, removidas="-".join(map(str, rem)), params_M=PARAMS["n"] / 1e6,
                           latencia_s=float(np.median(tempos)), pico_gpu_gb=float(np.max(picos)),
                           test=r["metrica_artigo_test"]))
        print(linhas[-1], flush=True)
    out = ap_.AQUI.parent / "Block_Influence_results" / "custo_pfns"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(out / "nodepfn.csv", index=False)
