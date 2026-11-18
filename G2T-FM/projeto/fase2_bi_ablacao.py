"""
Fase 2 do projeto para o G2T-LimiX: Block Influence (BI) por camada e ablação individual (remover cada camada sozinha).

Roda o pipeline oficial (bin.g2t.g2t_fm.main, ICL, ensemble de 10) com duas interceptações:
  - lib.tfm.load_tfm: registra hooks de BI nas 12 camadas (EncoderBaseLayer) e/ou remove camadas de
    transformer_encoder.layers (a saída da camada anterior passa direto para a seguinte, como no GraphPFN);
  - _batched_forward: marca os forwards de nós de treino, que ficam fora do BI.
BI medido como em GraphFPN/analise_camadas_e_poda_graphpfn.ipynb: 1 − cos e arccos(cos)/π entre entrada e saída da
camada, média sobre todos os tokens ("all") e sobre o token-alvo (último token) dos nós de consulta ("tgt").

Uso (da raiz do repositório, ~/venvs/g2tfm):
    python G2T-FM/projeto/fase2_bi_ablacao.py --dataset tolokers-2 --modo bi        # baseline + BI, sementes 0 1 2
    python G2T-FM/projeto/fase2_bi_ablacao.py --dataset tolokers-2 --modo ablacao   # remove cada camada sozinha
Saída: ../Block_Influence_results/g2t_fase2/<dataset>/{baseline,block_influence,ablacao_individual}.csv
"""
import argparse
import copy
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parents[1]  # G2T-FM/
sys.path.insert(0, str(AQUI))
os.chdir(AQUI)  # caminhos de dados das configs são relativos a G2T-FM/

import lib  # noqa: E402
import bin.g2t.g2t_fm as g2t_fm  # noqa: E402

RESULTS = AQUI.parent / "Block_Influence_results" / "g2t_fase2"
N_LAYERS = 12

ESTADO = dict(remover=(), medir=False, eh_treino=False, eval_pos=None, acc=None)


def bi_reduce(h_in, h_out, eval_pos, chunk=512):
    """Igual a bi_reduce do notebook do GraphPFN: tensores [1, nós, tokens, d]. Blocos de 512 nós (o notebook usa
    4096) porque com 4096 a cópia em float32 estourava os 16 GB da GPU no amazon-ratings; a soma é a mesma."""
    a, b = h_in[0], h_out[0]
    acc = np.zeros(6)  # cos_all, ang_all, n_all, cos_tgt, ang_tgt, n_tgt
    for s in range(0, a.shape[0], chunk):
        cos = F.cosine_similarity(a[s:s + chunk].float(), b[s:s + chunk].float(), dim=-1).clamp(-1, 1)
        ang = torch.arccos(cos) / np.pi
        acc[:3] += [(1 - cos).sum().item(), ang.sum().item(), cos.numel()]
        is_q = torch.arange(s, s + cos.shape[0], device=cos.device) >= eval_pos
        ct, at = cos[is_q, -1], ang[is_q, -1]
        acc[3:] += [(1 - ct).sum().item(), at.sum().item(), ct.numel()]
    return acc


_load_tfm_original = g2t_fm.load_tfm


def load_tfm_projeto(*args, **kwargs):
    tfm = _load_tfm_original(*args, **kwargs)
    stack = tfm.module.transformer_encoder
    assert len(stack.layers) == N_LAYERS
    if ESTADO["medir"]:
        ESTADO["acc"] = np.zeros((N_LAYERS, 6))
        for i, layer in enumerate(stack.layers):
            def hook(m, a, kw, out, i=i):
                if not ESTADO["eh_treino"]:
                    ESTADO["acc"][i] += bi_reduce(a[0], out[0], kw["eval_pos"])
            layer.register_forward_hook(hook, with_kwargs=True)
    if ESTADO["remover"]:
        stack.layers = nn.ModuleList([l for i, l in enumerate(stack.layers) if i not in ESTADO["remover"]])
    return tfm


_batched_forward_original = g2t_fm._batched_forward


def batched_forward_projeto(g2t, features, y_train, train_mask, part_mask, *args, **kwargs):
    ESTADO["eh_treino"] = bool((part_mask & train_mask).any())
    try:
        return _batched_forward_original(g2t, features, y_train, train_mask, part_mask, *args, **kwargs)
    finally:
        ESTADO["eh_treino"] = False


g2t_fm.load_tfm = load_tfm_projeto
g2t_fm._batched_forward = batched_forward_projeto


def rodar(dataset, seed, remover=(), medir=False):
    cfg = lib.load_config(AQUI / "exp" / "projeto" / "fase1" / "limix-icl" / dataset / "evaluation.toml")
    config = copy.deepcopy(cfg["base_config"])
    config["seed"] = seed
    ESTADO.update(remover=tuple(remover), medir=medir, acc=None)
    saida = Path(tempfile.mkdtemp(prefix=f"g2t_{dataset}_"))
    try:
        report = g2t_fm.main(config, saida / "run", force=True)
    finally:
        shutil.rmtree(saida, ignore_errors=True)
    m = report["metrics"]
    return m["val"]["score"], m["test"]["score"], ESTADO["acc"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--modo", choices=["bi", "ablacao"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    out = RESULTS / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    lib.init()  # como em bin/go.py
    lib.configure_torch()

    if a.modo == "bi":
        base, bi = [], []
        for seed in a.seeds:
            val, test, acc = rodar(a.dataset, seed, medir=True)
            base.append(dict(dataset=a.dataset, seed=seed, val=val, test=test))
            for i in range(N_LAYERS):
                c = acc[i]
                bi.append(dict(dataset=a.dataset, seed=seed, layer=i, cos_all=c[0] / c[2], ang_all=c[1] / c[2],
                               cos_tgt=c[3] / c[5], ang_tgt=c[4] / c[5]))
            print(f"{a.dataset} seed={seed}: val={val:.4f} test={test:.4f}", flush=True)
        pd.DataFrame(base).to_csv(out / "baseline.csv", index=False)
        pd.DataFrame(bi).to_csv(out / "block_influence.csv", index=False)
    else:
        caminho = out / "ablacao_individual.csv"
        feitos = pd.read_csv(caminho) if caminho.exists() else pd.DataFrame(columns=["dataset", "seed", "removida", "val", "test"])
        for seed in a.seeds:
            for i in range(N_LAYERS):
                if ((feitos["seed"] == seed) & (feitos["removida"] == i)).any():
                    continue
                val, test, _ = rodar(a.dataset, seed, remover=(i,))
                feitos = pd.concat([feitos, pd.DataFrame([dict(dataset=a.dataset, seed=seed, removida=i, val=val, test=test)])],
                                   ignore_index=True)
                feitos.to_csv(caminho, index=False)  # grava a cada camada: retoma se for interrompido
                print(f"{a.dataset} seed={seed} sem camada {i}: val={val:.4f} test={test:.4f}", flush=True)
