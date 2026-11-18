"""
Custo da poda no G2T-LimiX, medido como nos outros PFNs (custo_graphpfn.py, custo_nodepfn.py) para comparação direta:
tolokers-2, semente 0, pipeline de avaliação completo da Fase 2 (ICL, ensemble de 10, pré e pós-processamento
incluídos), k = 0, 2, 4, 6 camadas removidas pela ordem bi_todos (camada 0 protegida). Para cada k: parâmetros do
modelo, latência (mediana de 3 execuções depois de 1 de aquecimento) e pico de memória da GPU.

Uso (da raiz do repositório, ~/venvs/g2tfm): python G2T-FM/projeto/custo_g2t.py
Saída: ../Block_Influence_results/custo_pfns/g2t.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AQUI = Path(__file__).resolve().parents[1]  # G2T-FM/
sys.path.insert(0, str(AQUI.parent / "pipeline"))
import fase2_bi_ablacao as f2  # noqa: E402
import fase3_ordens as f3  # noqa: E402

PARAMS = {}
_load = f2.load_tfm_projeto


def load_contando(*a, **k):
    tfm = _load(*a, **k)
    PARAMS["n"] = sum(p.numel() for p in tfm.module.parameters())
    return tfm


f2.g2t_fm.load_tfm = load_contando

if __name__ == "__main__":
    ds = "tolokers-2"
    ordem = f3.ordens("g2t_fase2", ds)["bi_todos"]
    f2.lib.init()
    f2.lib.configure_torch()
    linhas = []
    for k in [0, 2, 4, 6]:
        rem = tuple(sorted(ordem[:k]))
        f2.rodar(ds, 0, remover=rem)  # aquecimento
        tempos, picos = [], []
        for _ in range(3):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            val, test, _ = f2.rodar(ds, 0, remover=rem)
            torch.cuda.synchronize()
            tempos.append(time.time() - tic); picos.append(torch.cuda.max_memory_allocated() / 2**30)
        linhas.append(dict(modelo="G2T-LimiX", dataset=ds, k=k, removidas="-".join(map(str, rem)),
                           params_M=PARAMS["n"] / 1e6, latencia_s=float(np.median(tempos)),
                           pico_gpu_gb=float(np.max(picos)), test=test))
        print(linhas[-1], flush=True)
    out = AQUI.parent / "Block_Influence_results" / "custo_pfns"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(out / "g2t.csv", index=False)
