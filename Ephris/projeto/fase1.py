"""Fase 1: modelo inteiro do Ephris nos datasets de classificação do protocolo comum, 3 sementes.

Uso: python projeto/fase1.py [--datasets a,b]
Saída: Block_Influence_results/ephris_fase1/modelo_inteiro.csv
"""
import argparse
import time

import pandas as pd
import torch

from comum import DATASETS, R, SEEDS, Tarefa, acaso

OUT = R / "ephris_fase1" / "modelo_inteiro.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feitos = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["dataset", "seed"])
    for name in a.datasets.split(","):
        for seed in SEEDS:
            if ((feitos.dataset == name) & (feitos.seed == seed)).any():
                continue
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                t = Tarefa(name, seed)
                r = t.avaliar()
                linha = dict(dataset=name, seed=seed, val=r["val"], test=r["test"],
                             acaso=acaso(name, t.y[t.masks["test"]]), n_nos=len(t.y), n_feat=t.x.shape[1],
                             n_arestas=t.e.shape[1], n_ctx=len(t.context_idx), tempo_s=time.time() - t0,
                             pico_gb=torch.cuda.max_memory_allocated() / 2**30, erro="")
                del t
            except torch.OutOfMemoryError as ex:
                linha = dict(dataset=name, seed=seed, erro=f"OOM: {ex}"[:200])
            except RuntimeError as ex:
                linha = dict(dataset=name, seed=seed, erro=str(ex)[:200])
            torch.cuda.empty_cache()
            print(linha, flush=True)
            feitos = pd.concat([feitos, pd.DataFrame([linha])], ignore_index=True)
            feitos.to_csv(OUT, index=False)
