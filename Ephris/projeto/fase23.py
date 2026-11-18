"""
Fases 2 e 3 do Ephris, no mesmo desenho do G2T-LimiX e do NodePFN (fase2_bi_ablacao.py, fase3_poda.py, fase3_ordens.py).

Camadas analisadas: as 10 camadas MP_ICL (model.icl.layers; 2,93 M parâmetros cada, 58% do modelo). Cada nó é um
único estado de 512 dimensões nessas camadas (mais 8 nós globais, fora das médias):
  - BI "todos": média sobre todos os nós reais;
  - BI "alvo": média sobre os nós de consulta (fora do contexto), que são os que o modelo prediz; é o análogo
    do token-alvo dos PFNs.
BI cosseno = 1 − cos(h_in, h_out) e angular = arccos(cos)/π, por camada.

Fase 2: BI (sem rótulos) e ablação individual (val e teste) de cada camada, 3 sementes.
Fase 3: poda cumulativa sem re-treino, k = 1..6, camada 0 protegida, ordens:
  bi_todos, bi_alvo (menor BI primeiro, média das sementes), ablacao (maior retido na VALIDAÇÃO primeiro),
  aleatoria_0..2 (permutações de 1..9 com semente 0).

Uso: python projeto/fase23.py --dataset tolokers-2
Saída: Block_Influence_results/ephris_fase2/<ds>/{baseline,block_influence,ablacao_individual}.csv,
       Block_Influence_results/ephris_fase3/<ds>/poda.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from comum import R, SEEDS, Tarefa, acaso

N_LAYERS = 10
PROTECTED = {0}
KS = range(1, 7)
N_ALEATORIAS = 3


class Podador:
    """Remove/restaura camadas MP_ICL do modelo preparado."""

    def __init__(self, model):
        self.model = model
        self.orig = list(model.icl.layers)

    def manter(self, keep):
        self.model.icl.layers = nn.ModuleList([self.orig[i] for i in keep])

    def remover(self, removidas):
        self.manter([i for i in range(len(self.orig)) if i not in set(removidas)])

    def restaurar(self):
        self.model.icl.layers = nn.ModuleList(self.orig)


def medir_bi(t):
    """Um forward com hooks nas 10 camadas. Retorna [(layer, cos_all, ang_all, cos_tgt, ang_tgt)]."""
    n = len(t.y)
    query = torch.ones(n, dtype=torch.bool)
    query[torch.from_numpy(t.context_idx)] = False
    res = {}

    def hook(i):
        def h(mod, args, kwargs, out):
            x, y = args[0][:n].float(), out[:n].float()
            cos = F.cosine_similarity(x, y, dim=-1).clamp(-1, 1)
            ang = torch.arccos(cos) / np.pi
            q = query.to(cos.device)
            res[i] = (float((1 - cos).mean()), float(ang.mean()), float((1 - cos[q]).mean()), float(ang[q].mean()))
        return h

    hs = [l.register_forward_hook(hook(i), with_kwargs=True) for i, l in enumerate(t.model.icl.layers)]
    try:
        logp = t.predict()
    finally:
        for h in hs:
            h.remove()
    return [(i, *res[i]) for i in range(N_LAYERS)], logp


def ordens(bi, ab, base_val, acaso_val):
    bi = bi.groupby("layer").mean(numeric_only=True).drop(index=list(PROTECTED))
    ab = ab.copy()
    ab["ret_val"] = (ab.val - acaso_val) / (ab.seed.map(base_val) - acaso_val)
    ret = ab[~ab.removida.isin(PROTECTED)].groupby("removida").ret_val.mean()
    out = {
        "bi_todos": list(bi.cos_all.sort_values(kind="stable").index),
        "bi_alvo": list(bi.cos_tgt.sort_values(kind="stable").index),
        "ablacao": list(ret.sort_values(ascending=False, kind="stable").index),
    }
    rng = np.random.default_rng(0)
    for r in range(N_ALEATORIAS):
        out[f"aleatoria_{r}"] = list(rng.permutation(np.arange(1, N_LAYERS)))
    return {k: [int(c) for c in v] for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    name = a.dataset
    d2, d3 = R / "ephris_fase2" / name, R / "ephris_fase3" / name
    d2.mkdir(parents=True, exist_ok=True)
    d3.mkdir(parents=True, exist_ok=True)

    base, bis, abl, tarefas = [], [], [], {}
    for seed in SEEDS:
        t = Tarefa(name, seed)
        pod = Podador(t.model)
        rows, logp = medir_bi(t)
        r = t.avaliar(logp)
        base.append(dict(dataset=name, seed=seed, val=r["val"], test=r["test"],
                         acaso_val=acaso(name, t.y[t.masks["val"]]), acaso_test=acaso(name, t.y[t.masks["test"]])))
        bis += [dict(dataset=name, seed=seed, layer=i, cos_all=c, ang_all=a_, cos_tgt=ct, ang_tgt=at)
                for i, c, a_, ct, at in rows]
        for i in range(N_LAYERS):
            pod.remover([i])
            r = t.avaliar()
            abl.append(dict(dataset=name, seed=seed, removida=i, val=r["val"], test=r["test"]))
            print(f"{name} s{seed} sem camada {i}: val={r['val']:.4f} test={r['test']:.4f}", flush=True)
        pod.restaurar()
        tarefas[seed] = (t, pod)
        print(f"{name} s{seed}: inteiro val={base[-1]['val']:.4f} test={base[-1]['test']:.4f}", flush=True)
    base, bis, abl = pd.DataFrame(base), pd.DataFrame(bis), pd.DataFrame(abl)
    base.to_csv(d2 / "baseline.csv", index=False)
    bis.to_csv(d2 / "block_influence.csv", index=False)
    abl.to_csv(d2 / "ablacao_individual.csv", index=False)

    ords = ordens(bis, abl, base.set_index("seed").val, float(base.acaso_val.mean()))
    pd.Series({k: " ".join(map(str, v)) for k, v in ords.items()}).to_csv(d3 / "ordens.csv", header=["ordem"])
    poda, cache = [], {}
    for estr, o in ords.items():
        for k in KS:
            rem = tuple(sorted(o[:k]))
            for seed, (t, pod) in tarefas.items():
                if (rem, seed) not in cache:
                    pod.remover(rem)
                    cache[(rem, seed)] = t.avaliar()
                    pod.restaurar()
                r = cache[(rem, seed)]
                poda.append(dict(dataset=name, estrategia=estr, k=k, removidas=" ".join(map(str, rem)), seed=seed,
                                 val=r["val"], test=r["test"]))
            m = pd.DataFrame(poda[-len(SEEDS):]).test.mean()
            print(f"{name} {estr} k={k} {rem}: test={m:.4f}", flush=True)
    pd.DataFrame(poda).to_csv(d3 / "poda.csv", index=False)
