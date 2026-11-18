"""
Custo da poda no GraphPFN, medido como em custo_g2t.py e custo_nodepfn.py (o EXP-003 mediu no city-roads-M com
1 membro de ensemble; aqui é o protocolo comum aos 3 PFNs): tolokers-2, semente 0, avaliação completa do protocolo
do artigo (ensemble de 10, inferência em blocos), k = 0, 2, 4, 6 pela ordem bi_todos (camada 0 protegida);
parâmetros, latência (mediana de 3 depois de 1 aquecimento; inclui pré-processamento e carga dos dados) e pico de GPU.

Uso (da raiz do repositório): python GraphFPN/custo_graphpfn.py
Saída: Block_Influence_results/custo_pfns/graphpfn.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase23_graphpfn as g  # noqa: E402

if __name__ == "__main__":
    ds = "tolokers-2"
    ordem = g.f3.ordens("graphfpn_fase2", ds)["bi_todos"]
    base = g.load_base()
    p_total = sum(p.numel() for p in base.parameters())
    p_layer = sum(p.numel() for p in base.tfm.module.transformer_encoder.layers[0].parameters())
    linhas = []
    for k in [0, 2, 4, 6]:
        rem = tuple(sorted(ordem[:k]))
        g.avaliar(base, g.prepare_task(ds), 0, remove=rem)  # aquecimento
        tempos, picos = [], []
        for _ in range(3):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            val, test = g.avaliar(base, g.prepare_task(ds), 0, remove=rem)
            torch.cuda.synchronize()
            tempos.append(time.time() - tic); picos.append(torch.cuda.max_memory_allocated() / 2**30)
        linhas.append(dict(modelo="GraphPFN", dataset=ds, k=k, removidas="-".join(map(str, rem)),
                           params_M=(p_total - k * p_layer) / 1e6, latencia_s=float(np.median(tempos)),
                           pico_gpu_gb=float(np.max(picos)), test=test))
        print(linhas[-1], flush=True)
    out = g.R / "custo_pfns"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(out / "graphpfn.csv", index=False)
