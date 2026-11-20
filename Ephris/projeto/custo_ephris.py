"""
Custo da poda no Ephris, medido como nos outros PFNs (custo_graphpfn.py, custo_nodepfn.py, custo_g2t.py):
tolokers-2, semente 0, k = 0, 2, 4, 6 camadas MP_ICL removidas pela ordem bi_todos (camada 0 protegida).
Para cada k: parâmetros do modelo, latência da predição completa (forward de todos os nós com o contexto já
preparado; mediana de 3 execuções depois de 1 de aquecimento) e pico de memória da GPU. O prepare (pré-processamento
e cache de arestas) não depende das camadas e é medido uma vez (t_prepare_s).

Uso (de dentro de Ephris/projeto/): python custo_ephris.py
Saída: Block_Influence_results/custo_pfns/ephris.csv
"""
import time

import numpy as np
import pandas as pd
import torch

from comum import R, Tarefa
from fase23 import Podador

if __name__ == "__main__":
    ds = "tolokers-2"
    ordem = [int(c) for c in pd.read_csv(R / "ephris_fase3" / ds / "ordens.csv", index_col=0)
             .loc["bi_todos", "ordem"].split()]
    t = Tarefa(ds, 0)
    pod = Podador(t.model)
    linhas = []
    for k in [0, 2, 4, 6]:
        rem = tuple(sorted(ordem[:k]))
        pod.remover(rem)
        n_params = sum(p.numel() for p in t.model.parameters())
        t.avaliar()  # aquecimento
        tempos, picos = [], []
        for _ in range(3):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tic = time.time()
            r = t.avaliar()
            torch.cuda.synchronize()
            tempos.append(time.time() - tic); picos.append(torch.cuda.max_memory_allocated() / 2**30)
        pod.restaurar()
        linhas.append(dict(modelo="Ephris", dataset=ds, k=k, removidas="-".join(map(str, rem)),
                           params_M=n_params / 1e6, latencia_s=float(np.median(tempos)),
                           pico_gpu_gb=float(np.max(picos)), test=r["test"], t_prepare_s=t.t_prepare))
        print(linhas[-1], flush=True)
    out = R / "custo_pfns"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(out / "ephris.csv", index=False)
