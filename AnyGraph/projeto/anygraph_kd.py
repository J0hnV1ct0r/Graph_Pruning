"""
KD do AnyGraph (pretrain_link2, Tier 2 = camadas [4, 5, 6, 7] removidas, −50%), no mesmo desenho do KD dos PFNs
(EXP-004 v2 / EXP-008): student podado × student de mesma arquitetura com pesos aleatórios.

  - Professor: o expert que o modelo inteiro escolhe para o dataset (roteamento do AnyGraph), congelado.
  - Students: (a) o mesmo expert sem as camadas [4, 5, 6, 7]; (b) uma MLP de 4 camadas com pesos aleatórios.
    O expert fica fixo (sem roteamento) para isolar o efeito das camadas. "Podado antes do KD" também é medido com o
    expert fixo; por isso difere do Tier 2 do EXP-002, em que o roteador pode trocar de expert.
  - Loss sem rótulos: MSE entre as embeddings de saída do student e do professor (são elas que, por produto
    interno com os nós de classe, dão a predição). Só nós FORA do teste entram no treino (análogo aos nós de treino
    dos PFNs); as entradas são os projetores do dataset (fixos para a semente).
  - Protocolo do KD dos PFNs: Adam; lr por student numa varredura de 20 passos em {1e-4, 3e-4, 1e-3} pela loss
    de KD na metade A dos nós de teste; até 200 passos; lote de 1024 nós; avaliação a cada 10 passos; melhor ponto
    pela acurácia na metade A; parada após 4 avaliações sem melhora. Reporta-se a metade B (como em anygraph_fases).
  - Sementes 0, 1, 2 (projetores sorteados com a semente; mesma divisão A/B de anygraph_fases.py).

Uso: python AnyGraph/projeto/anygraph_kd.py --dataset cora --seed 0 [--steps 200 --patience 4]
Saída: Block_Influence_results/anygraph_kd/kd_<ds>_s<seed>_<student>_<steps>p<patience>.json (+ log .csv)
"""
import argparse
import copy
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anygraph_fases as af  # noqa: E402

REMOVER = (4, 5, 6, 7)
LR_GRID, SWEEP_STEPS, BATCH, EVAL_EVERY = [1e-4, 3e-4, 1e-3], 20, 1024, 10
OUT = af.R / "anygraph_kd"


def main(dataset, seed, n_steps, patience, ckpt="pretrain_link2"):
    import torch as t
    import torch.nn as nn
    exp, handler = af.preparar(ckpt, dataset)
    full = exp.model
    af.fixar_semente(seed)
    handler.make_projectors()
    af.fixar_semente(10_000 + seed)
    full.assign_experts([handler], reca=False, log_assignment=False)
    e_idx = int(full.assignment[0])
    teacher = full.experts[e_idx].eval()
    proj = handler.projectors.to("cuda")
    with t.no_grad():
        alvo = teacher(proj).detach()

    p0, y = af.predizer(exp, handler, full)
    tst_nodes = t.cat([b[0] for b in handler.tst_loader]).long().numpy()
    a = af.metade_a(len(y))
    treino = np.setdiff1d(np.arange(proj.shape[0]), tst_nodes)          # nós fora do teste (inclui nós de classe)
    nos_a = t.tensor(tst_nodes[a], device="cuda")
    acc = lambda p, m: float((p[m] == y[m]).mean())  # noqa: E731
    prof = dict(val=acc(p0, a), test=acc(p0, ~a), acaso_test=float(np.bincount(y[~a]).max() / (~a).sum()))
    print(f"{dataset} s{seed}: expert {e_idx}; professor val={prof['val']:.4f} test={prof['test']:.4f}", flush=True)

    teacher_mlp = copy.deepcopy(teacher.trainable_nn).eval()  # cópia intocável do professor

    def predizer_com(mlp):
        """Troca a MLP do expert e_idx pela do student, prediz com o expert fixo e restaura o professor."""
        orig = teacher.trainable_nn
        teacher.trainable_nn = mlp
        full.assignment = [e_idx]
        try:
            return af.predizer(exp, handler, full)[0]
        finally:
            teacher.trainable_nn = orig

    def avaliar(mlp):
        mlp.eval()
        p = predizer_com(mlp)
        with t.no_grad():
            kd_a = float(((mlp(proj[nos_a]) - alvo[nos_a]) ** 2).mean())
        return dict(val=acc(p, a), test=acc(p, ~a), kd_val=kd_a)

    def novo_student(tipo):
        if tipo == "Podado":
            mlp = copy.deepcopy(teacher_mlp)
            mlp.dense_layers = nn.Sequential(*[l for i, l in enumerate(mlp.dense_layers) if i not in REMOVER])
            mlp.layer_norms = nn.Sequential(*[l for i, l in enumerate(mlp.layer_norms) if i not in REMOVER])
        else:
            t.manual_seed(seed)
            mlp = af_mlp_aleatoria()
        return mlp.cuda()

    def af_mlp_aleatoria():
        from model import MLP
        m = MLP()
        keep = [i for i in range(af.N_LAYERS) if i not in REMOVER]
        m.dense_layers = nn.Sequential(*[m.dense_layers[i] for i in keep])
        m.layer_norms = nn.Sequential(*[m.layer_norms[i] for i in keep])
        return m

    def treinar(mlp, lr, passos, pat):
        opt = t.optim.Adam(mlp.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        log, melhor, estado, sem = [], None, None, 0
        r0 = avaliar(mlp); log.append(dict(passo=0, **r0)); melhor, estado = r0, copy.deepcopy(mlp.state_dict())
        for passo in range(1, passos + 1):
            mlp.train()
            idx = t.tensor(rng.choice(treino, size=min(BATCH, len(treino)), replace=False), device="cuda")
            loss = ((mlp(proj[idx]) - alvo[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if passo % EVAL_EVERY == 0:
                r = avaliar(mlp); log.append(dict(passo=passo, loss_treino=float(loss), **r))
                if r["val"] > melhor["val"]:
                    melhor, estado, sem = r, copy.deepcopy(mlp.state_dict()), 0
                else:
                    sem += 1
                if pat and sem >= pat:
                    break
        mlp.load_state_dict(estado)
        return melhor, log

    OUT.mkdir(parents=True, exist_ok=True)
    for tipo in ["Podado", "Aleatório"]:
        tag = f"{dataset}_s{seed}_{tipo}_{n_steps}p{patience}"
        if (OUT / f"kd_{tag}.json").exists():
            continue
        antes = avaliar(novo_student(tipo))
        varredura = {}
        for lr in LR_GRID:
            mlp = novo_student(tipo)
            opt = t.optim.Adam(mlp.parameters(), lr=lr)
            rng = np.random.default_rng(seed)
            mlp.train()
            for _ in range(SWEEP_STEPS):
                idx = t.tensor(rng.choice(treino, size=min(BATCH, len(treino)), replace=False), device="cuda")
                loss = ((mlp(proj[idx]) - alvo[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            varredura[lr] = avaliar(mlp)["kd_val"]
        lr = min(varredura, key=varredura.get)
        tic = time.time()
        melhor, log = treinar(novo_student(tipo), lr, n_steps, patience)
        res = dict(dataset=dataset, seed=seed, student=tipo, expert=e_idx, lr=lr, varredura=varredura,
                   professor=prof, antes=antes, melhor=melhor, passo_melhor=int(max(l["passo"] for l in log if l["val"] == melhor["val"])),
                   passos=int(log[-1]["passo"]), tempo_s=time.time() - tic,
                   params_student_M=sum(p.numel() for p in novo_student(tipo).parameters()) / 1e6,
                   params_professor_M=sum(p.numel() for p in teacher_mlp.parameters()) / 1e6)
        pd.DataFrame(log).to_csv(OUT / f"kd_log_{tag}.csv", index=False)
        json.dump(res, open(OUT / f"kd_{tag}.json", "w"), indent=1, ensure_ascii=False)
        print(f"  {tipo}: lr={lr:g} antes test={antes['test']:.4f} → melhor test={melhor['test']:.4f} "
              f"(passo {res['passo_melhor']}, professor {prof['test']:.4f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=af.DATASETS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()
    threading.Thread(target=af.vigia_memoria, args=(4.0,), daemon=True).start()
    main(a.dataset, a.seed, a.steps, a.patience)
