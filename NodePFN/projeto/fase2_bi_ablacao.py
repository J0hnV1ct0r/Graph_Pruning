"""
Fase 2 do projeto para o NodePFN: Block Influence (BI) por camada e ablação individual (remover cada camada sozinha).

Usa avaliar_projeto.avaliar (mesmo pipeline da Fase 1) com os hiperparâmetros escolhidos pela validação na Fase 1
(Block_Influence_results/nodepfn_fase1/<dataset>.json). Camadas: model.transformer_encoder.layers (12
TransformerEncoderLayer). A entrada de cada camada tem forma [sequência, configurações do ensemble, 256], com a
sequência = [tokens globais, estilo, contexto (treino), consulta (validação + teste)]. BI como no GraphPFN:
1 − cos e arccos(cos)/π entre entrada e saída da camada, média sobre todos os tokens ("all") e sobre os tokens dos
nós de consulta ("tgt", o equivalente ao token-alvo: no NodePFN cada nó é um token e a predição sai deles).

Uso (da raiz do repositório, ~/venvs/nodepfn): python NodePFN/projeto/fase2_bi_ablacao.py --dataset tolokers-2 [--seeds 0 1 2]
Saída: ../Block_Influence_results/nodepfn_fase2/<dataset>/{baseline,block_influence,ablacao_individual}.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import avaliar_projeto as ap_

RESULTS = ap_.AQUI.parent / "Block_Influence_results" / "nodepfn_fase2"
N_LAYERS = 12


def medidor(acc):
    """Devolve preparar_modelo que registra hooks de BI e acumula em acc[camada] = [cos_all, ang_all, n, cos_tgt, ang_tgt, n]."""
    def preparar(model):
        n_query = {}

        def pre(m, args, kwargs):
            src = args[0] if args else kwargs["src"]
            sep = kwargs.get("single_eval_pos", args[2] if len(args) > 2 else None)
            x_src = src[1] if len(src) == 4 else src[0]
            n_query["n"] = x_src.shape[0] - sep

        hooks = [model.register_forward_pre_hook(pre, with_kwargs=True)]
        for i, layer in enumerate(model.transformer_encoder.layers):
            def hook(m, args, kwargs, out, i=i):
                h_in = args[0] if args else kwargs["src"]
                cos = F.cosine_similarity(h_in.float(), out.float(), dim=-1).clamp(-1, 1)  # [S, B]
                ang = torch.arccos(cos) / np.pi
                q = cos[-n_query["n"]:], ang[-n_query["n"]:]
                acc[i] += [(1 - cos).sum().item(), ang.sum().item(), cos.numel(),
                           (1 - q[0]).sum().item(), q[1].sum().item(), q[0].numel()]
            hooks.append(layer.register_forward_hook(hook, with_kwargs=True))
        return lambda: [h.remove() for h in hooks]
    return preparar


def removedor(camada):
    def preparar(model):
        stack = model.transformer_encoder
        original = stack.layers
        stack.layers = nn.ModuleList([l for i, l in enumerate(original) if i != camada])

        def desfazer():
            stack.layers = original
        return desfazer
    return preparar


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(ap_.HP))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--so-bi", action="store_true", help="só baseline + BI (para teste)")
    a = p.parse_args()
    device = "cpu" if a.cpu else "cuda"
    hp = json.load(open(ap_.RESULTS / f"{a.dataset}.json"))["hp"]
    out = RESULTS / a.dataset
    out.mkdir(parents=True, exist_ok=True)

    base, bi = [], []
    for seed in a.seeds:
        acc = np.zeros((N_LAYERS, 6))
        r = ap_.avaliar(a.dataset, seed, device, hp, preparar_modelo=medidor(acc))
        base.append(dict(dataset=a.dataset, seed=seed, val=r["metrica_artigo_val"], test=r["metrica_artigo_test"]))
        for i in range(N_LAYERS):
            c = acc[i]
            bi.append(dict(dataset=a.dataset, seed=seed, layer=i, cos_all=c[0] / c[2], ang_all=c[1] / c[2],
                           cos_tgt=c[3] / c[5], ang_tgt=c[4] / c[5]))
        print(f"{a.dataset} seed={seed}: val={base[-1]['val']:.4f} test={base[-1]['test']:.4f}", flush=True)
    pd.DataFrame(base).to_csv(out / "baseline.csv", index=False)
    pd.DataFrame(bi).to_csv(out / "block_influence.csv", index=False)

    if a.so_bi:
        raise SystemExit(0)
    abl = []
    for seed in a.seeds:
        for i in range(N_LAYERS):
            r = ap_.avaliar(a.dataset, seed, device, hp, preparar_modelo=removedor(i))
            abl.append(dict(dataset=a.dataset, seed=seed, removida=i, val=r["metrica_artigo_val"], test=r["metrica_artigo_test"]))
            print(f"{a.dataset} seed={seed} sem camada {i}: val={abl[-1]['val']:.4f} test={abl[-1]['test']:.4f}", flush=True)
        pd.DataFrame(abl).to_csv(out / "ablacao_individual.csv", index=False)
