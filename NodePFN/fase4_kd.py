"""
Fase 4 do projeto para o NodePFN: KD do modelo inteiro (professor congelado) para um student de 6 camadas,
student podado × student aleatório, com o mesmo protocolo do KD v2 do GraphPFN e do G2T-LimiX (G2T-FM/fase4_kd.py):
  - loss sem rótulos: T²·KL(p_T‖p_S) com T = 2 sobre os logits;
  - Adam; lr escolhido por student em varredura de 20 passos na grade {1e-4, 3e-4, 1e-3} pela loss de KD na validação;
  - até 200 passos; consulta de até 1024 nós de treino por passo e o resto do treino como contexto (≥ 75%);
  - a cada 10 passos, avaliação na validação (1 configuração do ensemble); guarda-se o estado com a melhor métrica de
    validação e para-se após 4 avaliações sem melhora; avaliação final com o ensemble da Fase 1.

Entrada do modelo igual à de avaliar_projeto.avaliar (suavização, TSVD, hiperparâmetros da Fase 1) e a mesma função
transformer_predict do NodePFNClassifier, com no_grad=False no student. Em cada passo a sequência é
[contexto, consulta de treino, nós de validação e teste]: os nós fora do treino entram só como consulta (não são
vistos pela atenção de ninguém, mas participam da propagação no grafo, como na inferência ICL); a loss usa só a
consulta de treino. Uma configuração do ensemble sorteada por passo (pela semente do passo).

Student podado: remove as k camadas de menor BI cosseno de todos os tokens (média da Fase 2) entre 1–11 (camada 0
protegida). Student aleatório: mesma arquitetura das 12 − k primeiras camadas, com os parâmetros reinicializados
(reset_parameters de cada módulo + init_weights do TransformerModel, como na construção).

Uso (de dentro de NodePFN/): python fase4_kd.py --dataset tolokers-2 --seed 0 [--prune-k 6] [--students Podado,Aleatório]
Saída: ../Block_Influence_results/nodepfn_fase4_kd/{kd_log,kd_resumo}_<dataset>_k<k>_s<seed>_<student>_<passos>p<pac>.{csv,json}
"""
import argparse
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, average_precision_score
from torch_geometric.nn import SimpleConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected

import avaliar_projeto as ap_
from node_classification import fix_seed, update_edge_index
from scripts.transformer_prediction_interface import NodePFNClassifier, get_params_from_config, transformer_predict

R = ap_.AQUI.parent / "Block_Influence_results"
N_LAYERS = 12
PROTECTED = {0}
TRAIN_BATCH = 1024
MIN_CONTEXT_RATIO = 0.75
TEMPERATURE = 2.0
EVAL_EVERY = 10
LR_GRID = [1e-4, 3e-4, 1e-3]
SWEEP_STEPS = 20
DEV = "cuda"


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def preparar(name, seed, hp):
    """Features e grafo exatamente como em avaliar_projeto.avaliar."""
    x, y, edge_index, masks = ap_.carregar(name)
    n = x.shape[0]
    fix_seed(seed)
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes=n, add_self_loops=True)
    X = x
    for _ in range(hp["smoothing_steps"]):
        X = SimpleConv(aggr="sum")(X, edge_index, edge_weight)
    if hp["n_components"] is not None:
        k = min(hp["n_components"], X.shape[1], X.shape[0] - 1)
        X = TruncatedSVD(n_components=k, algorithm=hp["svd_algorithm"], random_state=seed).fit_transform(X)
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    clf = NodePFNClassifier(device=DEV, base_path=str(ap_.AQUI / "models_ckpts" / "nodepfn"),
                            N_ensemble_configurations=hp["n_ensemble"], seed=seed, batch_size_inference=1,
                            subsample_features=True, i=0, e=30)
    idx = {k: torch.from_numpy(np.where(v)[0]) for k, v in masks.items()}
    nao_treino = torch.from_numpy(np.where(~masks["train"])[0])
    return dict(name=name, X=X, y=y, edge_index=edge_index, masks=masks, idx=idx, nao_treino=nao_treino, clf=clf,
                hp=hp, seed=seed, binaria=len(np.unique(y)) == 2)


def logits(model, t, ctx, query, n_ens, seed, grad=False):
    """Logits (sequência de consulta × classes) com contexto = nós ctx e consulta = nós query (índices globais)."""
    clf = t["clf"]
    X_full = torch.cat([t["X"][ctx], t["X"][query]]).unsqueeze(1).to(DEV)
    y_full = torch.cat([torch.from_numpy(t["y"][ctx.numpy()]).float(), torch.zeros(len(query))]).unsqueeze(1).to(DEV)
    ei = update_edge_index(t["edge_index"], ctx, query)
    with torch.set_grad_enabled(grad):
        out = transformer_predict(model, X_full, y_full, len(ctx), ei, device=DEV, style=clf.style, inference_mode=True,
                                  preprocess_transform="mix", extend_features=True, normalize_with_test=True,
                                  N_ensemble_configurations=n_ens, softmax_temperature=clf.temperature,
                                  multiclass_decoder=clf.multiclass_decoder, feature_shift_decoder=clf.feature_shift_decoder,
                                  differentiable_hps_as_style=clf.differentiable_hps_as_style, seed=seed,
                                  return_logits=True, no_grad=not grad, batch_size_inference=1,
                                  **get_params_from_config(clf.c))
    return out.squeeze(0).float()


def kd_loss(s, te):
    zs, zt = s / TEMPERATURE, te / TEMPERATURE
    return F.kl_div(F.log_softmax(zs, -1), F.log_softmax(zt, -1), log_target=True, reduction="batchmean") * TEMPERATURE ** 2


def metrica(t, lg, nodes):
    y = t["y"][nodes.numpy()]
    p = torch.softmax(lg, -1).cpu().numpy()
    return float(average_precision_score(y, p[:, 1])) if t["binaria"] else float(accuracy_score(y, p.argmax(1)))


def icl(model, t, n_ens):
    """ICL como na Fase 1: contexto = treino, consulta = validação + teste. Devolve logits por parte."""
    lg = logits(model, t, t["idx"]["train"], t["nao_treino"], n_ens, seed=t["seed"])
    pos = {part: torch.isin(t["nao_treino"], t["idx"][part]) for part in ["val", "test"]}
    return {part: lg[pos[part].to(lg.device)] for part in pos}


@torch.no_grad()
def avaliar(model, t, prof_raw):
    model.eval()
    lg = icl(model, t, 1)
    out = {}
    for part in ["val", "test"]:
        out[f"kd_{part}"] = kd_loss(lg[part], prof_raw[part]).item()
        out[f"metric_{part}"] = metrica(t, lg[part], t["idx"][part])
    return out


@torch.no_grad()
def avaliar_ensemble(model, t):
    lg = icl(model, t, t["hp"]["n_ensemble"])
    return metrica(t, lg["test"], t["idx"]["test"])


def kd_train(student, prof, t, lr, n_steps, prof_raw, label, patience=None, estado_path=None, extra=None):
    """Igual a kd_train_v2 do GraphPFN: melhor ponto pela métrica de validação, parada antecipada e retomada."""
    seed = t["seed"]
    torch.manual_seed(seed); random.seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    train_idx = t["idx"]["train"]
    n_train = len(train_idx)
    n_q = min(TRAIN_BATCH, int(n_train * (1 - MIN_CONTEXT_RATIO)))
    g = torch.Generator().manual_seed(seed)

    inicio, retomadas = 0, []
    if estado_path is not None and estado_path.exists():
        e = torch.load(estado_path, map_location="cpu", weights_only=False)
        student.load_state_dict(e["student"]); opt.load_state_dict(e["opt"])
        log, best, sem_melhora, t_kd, losses = e["log"], e["best"], e["sem_melhora"], e["t_kd"], e["losses"]
        inicio, retomadas = e["step"], e["retomadas"] + [e["step"]]
        g.manual_seed(seed * 1000 + inicio)
        print(f"  {label}: retomando do passo {inicio}", flush=True)
    else:
        log = [dict(step=0, t_kd=0.0, train_loss=np.nan, **avaliar(student, t, prof_raw))]
        best = dict(step=0, metric_val=log[0]["metric_val"],
                    state={k: v.detach().cpu().clone() for k, v in student.state_dict().items()})
        sem_melhora, t_kd, losses = 0, 0.0, []
    fila = torch.tensor([], dtype=torch.long)
    for step in range(inicio + 1, n_steps + 1):
        sync(); tic = time.time()
        if len(fila) < n_q:  # como o CandidateQueue do GraphPFN/G2T-FM: percorre o treino em permutações
            fila = torch.cat([fila, torch.randperm(n_train, generator=g)])
        batch, fila = fila[:n_q], fila[n_q:]
        ctx = torch.ones(n_train, dtype=torch.bool); ctx[batch] = False
        ctx_nodes, q_nodes = train_idx[ctx], train_idx[batch]
        query = torch.cat([q_nodes, t["nao_treino"]])
        s_seed = seed * 100_000 + step
        te = logits(prof, t, ctx_nodes, query, 1, s_seed)[:len(q_nodes)]
        student.train()
        s = logits(student, t, ctx_nodes, query, 1, s_seed, grad=True)[:len(q_nodes)]
        loss = kd_loss(s, te)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(); t_kd += time.time() - tic
        losses.append(loss.item())
        if not np.isfinite(losses[-1]):
            print(f"  {label}: loss não finita no passo {step}, interrompido", flush=True)
            break
        if step % EVAL_EVERY == 0 or step == n_steps:
            log.append(dict(step=step, t_kd=t_kd, train_loss=float(np.mean(losses[-EVAL_EVERY:])),
                            **avaliar(student, t, prof_raw)))
            mv = log[-1]["metric_val"]
            print(f"  {label} passo {step}: kd_val={log[-1]['kd_val']:.4f} metric_val={mv:.4f}", flush=True)
            if mv > best["metric_val"]:
                best = dict(step=step, metric_val=mv,
                            state={k: v.detach().cpu().clone() for k, v in student.state_dict().items()})
                sem_melhora = 0
            else:
                sem_melhora += 1
                if patience is not None and sem_melhora >= patience:
                    print(f"  {label}: parada antecipada no passo {step} (melhor: {best['step']})", flush=True)
                    break
            if estado_path is not None:
                tmp = estado_path.with_suffix(".tmp")
                torch.save(dict(student=student.state_dict(), opt=opt.state_dict(), log=log, best=best,
                                sem_melhora=sem_melhora, t_kd=t_kd, losses=losses[-EVAL_EVERY:], step=step,
                                retomadas=retomadas, extra=extra), tmp)
                tmp.replace(estado_path)
    best["retomadas"] = retomadas
    return pd.DataFrame(log), best


def camadas_student(kind, dataset, prune_k):
    if kind == "Podado":
        bi = pd.read_csv(R / "nodepfn_fase2" / dataset / "block_influence.csv").groupby("layer").cos_all.mean()
        removidas = [l for l in bi.sort_values(kind="stable").index if l not in PROTECTED][:prune_k]
        return [i for i in range(N_LAYERS) if i not in removidas]
    return list(range(N_LAYERS - prune_k))


def montar(prof, kind, manter, seed):
    st = copy.deepcopy(prof)
    st.transformer_encoder.layers = nn.ModuleList([st.transformer_encoder.layers[i] for i in manter])
    if kind == "Aleatório":
        torch.manual_seed(seed)
        for m in st.modules():
            if m is not st and hasattr(m, "reset_parameters"):
                m.reset_parameters()
        st.init_weights()
    for p in st.parameters():
        p.requires_grad_(True)
    return st


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(ap_.HP))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prune-k", type=int, default=6)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--patience", type=int, default=4, help="avaliações sem melhora na validação; 0 = sem parada")
    p.add_argument("--students", default="Podado,Aleatório")
    p.add_argument("--salvar", action="store_true", help="salva o melhor student em student_<tag>.pt")
    p.add_argument("--lr", type=float, default=None, help="pula a varredura e usa este lr")
    p.add_argument("--out", default="nodepfn_fase4_kd")
    p.add_argument("--cpu", action="store_true", help="para testes")
    a = p.parse_args()
    if a.cpu:
        DEV = "cpu"
    hp = json.load(open(ap_.RESULTS / f"{a.dataset}.json"))["hp"]
    out_dir = R / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    patience = a.patience or None

    t = preparar(a.dataset, a.seed, hp)
    prof = t["clf"].model[2].to(DEV).eval()
    for q in prof.parameters():
        q.requires_grad_(False)
    with torch.no_grad():
        prof_raw = icl(prof, t, 1)
    prof_ens = avaliar_ensemble(prof, t)
    print(f"{a.dataset} seed={a.seed}: professor (ensemble {hp['n_ensemble']}) = {prof_ens:.4f}", flush=True)

    for kind in a.students.split(","):
        tag = f"{a.dataset}_k{a.prune_k}_s{a.seed}_{kind}_{a.steps}p{a.patience}" + (f"_lr{a.lr:g}" if a.lr else "")
        out_log, out_sum = out_dir / f"kd_log_{tag}.csv", out_dir / f"kd_resumo_{tag}.json"
        if out_log.exists() and out_sum.exists():
            print(f"(já feito: {tag})", flush=True)
            continue
        manter = camadas_student(kind, a.dataset, a.prune_k)
        estado = out_dir / f"estado_{tag}.pt"
        salvo = torch.load(estado, map_location="cpu", weights_only=False)["extra"] if estado.exists() else None
        if salvo:
            best_lr, sweep = salvo["lr"], salvo["sweep"]
        elif a.lr:
            best_lr, sweep = a.lr, {}
        else:
            sweep = {}
            for lr in LR_GRID:
                st = montar(prof, kind, manter, a.seed)
                try:
                    d, _ = kd_train(st, prof, t, lr, SWEEP_STEPS, prof_raw, f"{kind} lr={lr:g}")
                    v = d.kd_val.iloc[-1]
                    sweep[lr] = v if np.isfinite(v) else np.inf
                except torch.OutOfMemoryError:
                    sweep[lr] = np.inf
                del st; torch.cuda.empty_cache()
            best_lr = min(sweep, key=sweep.get)
            print(f"{tag}: varredura {sweep} → lr = {best_lr:g}", flush=True)
        st = montar(prof, kind, manter, a.seed)
        antes = salvo["antes"] if salvo else avaliar_ensemble(st, t)
        d, best = kd_train(st, prof, t, best_lr, a.steps, prof_raw, kind, patience=patience, estado_path=estado,
                           extra=dict(lr=best_lr, sweep=sweep, antes=antes))
        ultimo = avaliar_ensemble(st, t)
        st.load_state_dict(best["state"])
        melhor = avaliar_ensemble(st, t)
        if a.salvar:
            torch.save(dict(state=best["state"], manter=manter, kind=kind, lr=best_lr, dataset=a.dataset, seed=a.seed,
                            melhor_passo=best["step"]), out_dir / f"student_{tag}.pt")
        info = dict(dataset=a.dataset, task="binclass" if t["binaria"] else "multiclass",
                    metric="AP" if t["binaria"] else "Acurácia", seed=a.seed, prune_k=a.prune_k, student=kind,
                    camadas=manter, params=n_params(st), params_teacher=n_params(prof), lr=best_lr,
                    sweep={f"{k:g}": v for k, v in sweep.items()}, max_steps=a.steps, patience=a.patience,
                    passos_feitos=int(d.step.iloc[-1]), melhor_passo=best["step"], metric_val_melhor=best["metric_val"],
                    teacher_test_ens=prof_ens, test_ens_antes=antes, test_ens_ultimo=ultimo, test_ens_melhor=melhor,
                    retomado_nos_passos=best["retomadas"])
        d.assign(dataset=a.dataset, student=kind, lr=best_lr, seed=a.seed, prune_k=a.prune_k).to_csv(out_log, index=False)
        json.dump(info, open(out_sum, "w"), default=float, indent=1)
        estado.unlink(missing_ok=True)
        print(f"{tag}: antes {antes:.4f} | último {ultimo:.4f} | melhor (passo {best['step']}) {melhor:.4f} | "
              f"professor {prof_ens:.4f}", flush=True)
        del st; torch.cuda.empty_cache()
