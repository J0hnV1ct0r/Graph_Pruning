"""
Fase 4 do projeto para o G2T-LimiX: KD do modelo inteiro (professor congelado) para um student de 6 camadas,
student podado × student aleatório, com o mesmo protocolo do KD v2 do GraphPFN (GraphFPN/kd_v2_graphpfn.py):
  - loss sem rótulos: classificação T²·KL(p_T‖p_S) com T = 2; regressão MSE entre saídas padronizadas;
  - Adam; lr escolhido por student em varredura de 20 passos na grade {1e-4, 3e-4, 1e-3} pela loss de KD na validação;
  - até 200 passos; consulta de até 1024 nós de treino por passo (resto do treino é contexto, ≥ 75%), como o
    fine-tuning oficial do G2T-FM (compute_loss/CandidateQueue); membro do ensemble sorteado a cada passo;
  - a cada 10 passos, avaliação na validação (1 membro do ensemble); guarda-se o estado com a melhor métrica de
    validação e para-se após 4 avaliações sem melhora; avaliação final com o ensemble de 10 do pipeline oficial.
Student podado: remove as k camadas de menor BI cosseno de todos os tokens (média da Fase 2) entre 1–11 (camada 0
protegida), como no GraphPFN. Student aleatório: LimiX sem pesos carregados, as 12 − k primeiras camadas.
O PEARL (pesos fixos de checkpoints/pearl_random.pt) é compartilhado e não é treinado; a mesma semente antes de cada
forward dá ao professor e ao student os mesmos vetores aleatórios do PEARL.

Uso (da raiz do repositório, ~/venvs/g2tfm): python G2T-FM/projeto/fase4_kd.py --dataset tolokers-2 --seed 0 [--prune-k 6] [--students Podado,Aleatório]
Saída: ../Block_Influence_results/g2t_fase4_kd/{kd_log,kd_resumo}_<dataset>_k<k>_s<seed>_<student>_<passos>p<pac>.{csv,json}
"""
import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
AQUI = Path(__file__).resolve().parents[1]  # G2T-FM/
sys.path.insert(0, str(AQUI))
os.chdir(AQUI)  # caminhos de dados das configs são relativos a G2T-FM/

import numpy as np
import pandas as pd
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib  # noqa: E402
import bin.g2t.g2t_fm as g2t_fm  # noqa: E402
from lib.ensemble import apply_feature_transform, apply_target_transform, create_ensemble, reverse_target_perm  # noqa: E402
from lib.graph.pearl import PEARL  # noqa: E402
from lib.tfm.limix import LimiXWrapper  # noqa: E402

R = AQUI.parent / "Block_Influence_results"
N_LAYERS = 12
PROTECTED = {0}
TRAIN_BATCH = 1024
MIN_CONTEXT_RATIO = 0.75
TEMPERATURE = 2.0
EVAL_EVERY = 10
LR_GRID = [1e-4, 3e-4, 1e-3]
SWEEP_STEPS = 20
EVAL_BATCH = 2**15
MAX_CTX = None  # máximo de nós de contexto por passo de treino (None = todo o treino restante, como o oficial)
TRAIN_CHUNKS = 1  # partes da consulta de treino por passo (acumulação de gradiente; mesmo gradiente, menos memória)
METRICA = {"binclass": "AP", "multiclass": "Acurácia", "regression": "R²"}


def preparar(dataset_name, seed, device):
    """Dados, features e ensemble exatamente como em g2t_fm.main (config da Fase 1)."""
    cfg = lib.load_config(AQUI / "exp" / "projeto" / "fase1" / "limix-icl" / dataset_name / "evaluation.toml")
    config = copy.deepcopy(cfg["base_config"])
    config["seed"] = seed
    import delu
    delu.random.seed(seed)
    dataset = lib.graph.data.GraphDataset.from_dir(**config["data"])
    d_red = config.get("d_reduction", {})

    def transform_and_reduce(features, key):
        features = lib.graph.data.transform_features(features, dataset.task, **config["transform"].get(key, {}))
        return lib.graph.data.apply_d_reduction(features, dataset.task, **d_red)

    sources = [transform_and_reduce(dataset.features, "features")]
    if "nfa" in config:
        sources.append(transform_and_reduce(lib.graph.data.apply_nfa(dataset, **config["nfa"]), "nfa"))
    for name, params in config.get("encodings", {}).items():
        enc = dict.fromkeys(lib.graph.data.GRAPH_FEATURE_KEYS)
        enc["num_features"] = lib.graph.data.get_structural_encodings(dataset.data["graph"], name=name, params=params)
        sources.append(transform_and_reduce(enc, "encodings"))
    stats = lib.graph.data.prepare_labels(dataset, config["transform"]["labels"])
    for key in lib.graph.data.GRAPH_FEATURE_KEYS:
        dataset.data[key] = None
    dataset = dataset.to_torch(device)
    train_mask = dataset.data["masks"]["train"]
    groups = []
    for s in sources:
        g = lib.graph.data.flatten_features(s)
        if g is None:
            continue
        g = lib.graph.data.drop_constant_features(torch.tensor(g, device=device), train_mask)
        if g.shape[1]:
            groups.append(g)
    y_train = dataset.data["labels"][train_mask].to(dtype=torch.float32, device=device)
    sizes = [g.shape[1] for g in groups]
    ens = create_ensemble(n_members=config["ensemble"]["n_members"], n_features=sum(sizes),
                          n_classes=dataset.task.try_compute_n_classes(), feature_group_sizes=sizes,
                          shuffle_features=config["ensemble"]["shuffle_features"],
                          shuffle_targets=config["ensemble"]["shuffle_targets"],
                          max_columns=config["ensemble"].get("max_columns"), device=device, seed=seed)
    assert not ens.uses_ecoc
    pearl = None
    if "pearl" in config:
        pearl = PEARL(**config["pearl"])
        pearl.load_state_dict(torch.load(AQUI / "checkpoints" / "pearl_random.pt"))
        pearl = pearl.to(device).eval()
        for p in pearl.parameters():
            p.requires_grad_(False)
    task = dataset.task
    tipo = "regression" if task.is_regression else ("binclass" if task.is_binclass else "multiclass")
    return dict(name=dataset_name, dataset=dataset, groups=groups, y_train=y_train, ens=ens, stats=stats,
                pearl=pearl, task=task, tipo=tipo, n_classes=task.try_compute_n_classes(), device=device,
                prediction_type="labels" if task.is_regression else "probs", seed=seed)


def montar(t, pesos=True, manter=None):
    torch.manual_seed(t["seed"])
    tfm = LimiXWrapper(load_weights=pesos, checkpointing=True)
    # feature_decoder só alimenta a perda auxiliar de reconstrução de features do pré-treino; a predição sai de
    # y_decoder. É o maior consumo de memória no treino, então é desligado (como no KD do GraphPFN).
    tfm.module.feature_decoder.forward = lambda x: None
    if manter is not None:
        stack = tfm.module.transformer_encoder
        stack.layers = nn.ModuleList([stack.layers[i] for i in manter])
    return g2t_fm.G2T(tfm=tfm, pearl=t["pearl"]).to(t["device"])


def forward(g2t, t, member, ctx_mask, y_ctx, eval_mask, seed):
    """Saída do G2T (log-probabilidades ou alvo padronizado) para os nós de eval_mask, em blocos."""
    feats = apply_feature_transform(t["groups"], member)
    y = apply_target_transform(y_ctx, member)
    idx = torch.where(eval_mask)[0]
    outs = []
    for chunk in idx.split(EVAL_BATCH):
        m = torch.zeros_like(eval_mask)
        m[chunk] = True
        torch.manual_seed(seed)  # mesmos vetores aleatórios do PEARL para professor e student
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(g2t(features=feats, y_train=y, train_mask=ctx_mask, eval_mask=m, graph=t["dataset"].data["graph"],
                            task_type=t["task"].type_))
    return reverse_target_perm(torch.cat(outs), member).float()


def kd_loss(s, te, t):
    if t["tipo"] == "regression":
        return F.mse_loss(s, te)
    n = t["n_classes"]
    zs, zt = s[..., :n] / TEMPERATURE, te[..., :n] / TEMPERATURE
    return F.kl_div(F.log_softmax(zs, -1), F.log_softmax(zt, -1), log_target=True, reduction="batchmean") * TEMPERATURE ** 2


def metrica(t, raw, part):
    """Métrica do artigo a partir das saídas de 1 forward, com a mesma conversão de g2t_fm.evaluate."""
    if t["stats"] is not None:
        pred = (raw * t["stats"].std + t["stats"].mean).cpu().numpy()
    elif t["tipo"] == "regression":
        pred = raw.cpu().numpy()
    else:
        pred = scipy.special.softmax(raw.cpu().numpy()[..., :t["n_classes"]], axis=-1)
        if t["tipo"] == "binclass":
            pred = pred[..., 1]
    return t["task"].calculate_metrics({part: pred}, t["prediction_type"])[part]["score"]


@torch.no_grad()
def avaliar(g2t, t, prof_raw):
    """Loss de KD (val/teste) contra o professor e métrica do artigo, ICL com todo o treino como contexto, membro 0."""
    g2t.eval()
    masks = t["dataset"].data["masks"]
    out = {}
    for part in ["val", "test"]:
        raw = forward(g2t, t, t["ens"].get_member(0), masks["train"], t["y_train"], masks[part], seed=t["seed"])
        out[f"kd_{part}"] = kd_loss(raw, prof_raw[part], t).item()
        out[f"metric_{part}"] = metrica(t, raw, part)
    return out


@torch.no_grad()
def avaliar_ensemble(g2t, t):
    """Métrica no teste com o ensemble de 10 do pipeline oficial (g2t_fm.evaluate)."""
    m, _, _ = g2t_fm.evaluate(g2t, t["dataset"], t["groups"], t["y_train"], t["prediction_type"], t["stats"],
                              t["device"], True, t["ens"], EVAL_BATCH, parts=["test"])
    return m["test"]["score"]


def kd_train(student, prof, t, lr, n_steps, prof_raw, label, patience=None, estado_path=None, extra=None):
    """Igual a kd_train_v2 do GraphPFN: melhor ponto pela métrica de validação, parada antecipada e retomada."""
    seed = t["seed"]
    torch.manual_seed(seed); random.seed(seed)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=lr)
    masks = t["dataset"].data["masks"]
    train_idx = torch.where(masks["train"])[0]
    n_train, dev = len(train_idx), t["device"]
    queue = g2t_fm.CandidateQueue(n_train, n_candidates=min(TRAIN_BATCH, int(n_train * (1 - MIN_CONTEXT_RATIO))), device=dev)

    inicio, retomadas = 0, []
    if estado_path is not None and estado_path.exists():
        e = torch.load(estado_path, map_location="cpu", weights_only=False)
        student.load_state_dict(e["student"]); opt.load_state_dict(e["opt"])
        log, best, sem_melhora, t_kd, losses = e["log"], e["best"], e["sem_melhora"], e["t_kd"], e["losses"]
        inicio, retomadas = e["step"], e["retomadas"] + [e["step"]]
        torch.manual_seed(seed * 1000 + inicio); random.seed(seed * 1000 + inicio)
        print(f"  {label}: retomando do passo {inicio}", flush=True)
    else:
        log = [dict(step=0, t_kd=0.0, train_loss=np.nan, **avaliar(student, t, prof_raw))]
        best = dict(step=0, metric_val=log[0]["metric_val"],
                    state={k: v.detach().cpu().clone() for k, v in student.state_dict().items()})
        sem_melhora, t_kd, losses = 0, 0.0, []
    for step in range(inicio + 1, n_steps + 1):
        torch.cuda.synchronize(); tic = time.time()
        student.train()
        batch = torch.unique(next(queue))
        ctx = torch.ones(n_train, dtype=torch.bool, device=dev); ctx[batch] = False
        if MAX_CTX is not None and int(ctx.sum()) > MAX_CTX:
            # Sorteia MAX_CTX nós de contexto neste passo (professor e student recebem o mesmo contexto).
            livres = torch.where(ctx)[0]
            ctx[:] = False
            ctx[livres[torch.randperm(len(livres), device=dev)[:MAX_CTX]]] = True
        ctx_mask = torch.zeros_like(masks["train"]); ctx_mask[train_idx[ctx]] = True
        q_mask = torch.zeros_like(masks["train"]); q_mask[train_idx[batch]] = True
        _, member = t["ens"].sample_member()
        y_ctx = t["y_train"][ctx]
        # Os nós de consulta só atendem ao contexto, então dividir a consulta em partes com o mesmo contexto e somar
        # os gradientes (ponderados pelo tamanho da parte) dá o mesmo gradiente da consulta inteira.
        q_idx = torch.where(q_mask)[0]
        opt.zero_grad(set_to_none=True)
        loss_passo = 0.0
        for parte in q_idx.tensor_split(TRAIN_CHUNKS):
            q_parte = torch.zeros_like(q_mask); q_parte[parte] = True
            with torch.no_grad():
                te = forward(prof, t, member, ctx_mask, y_ctx, q_parte, seed=seed * 100_000 + step)
            s = forward(student, t, member, ctx_mask, y_ctx, q_parte, seed=seed * 100_000 + step)
            loss = kd_loss(s, te, t) * (len(parte) / len(q_idx))
            loss.backward()
            loss_passo += loss.item()
            del s, te, loss
        opt.step()
        torch.cuda.synchronize(); t_kd += time.time() - tic
        losses.append(loss_passo)
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
        bi = pd.read_csv(R / "g2t_fase2" / dataset / "block_influence.csv").groupby("layer").cos_all.mean()
        removidas = [l for l in bi.sort_values(kind="stable").index if l not in PROTECTED][:prune_k]
        return [i for i in range(N_LAYERS) if i not in removidas]
    return list(range(N_LAYERS - prune_k))


def n_params(m):
    return sum(p.numel() for p in m.tfm.parameters())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prune-k", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=4, help="avaliações sem melhora na validação; 0 = sem parada")
    ap.add_argument("--students", default="Podado,Aleatório")
    ap.add_argument("--salvar", action="store_true", help="salva o melhor student em student_<tag>.pt")
    ap.add_argument("--lr", type=float, default=None, help="pula a varredura e usa este lr")
    ap.add_argument("--out", default="g2t_fase4_kd")
    ap.add_argument("--eval-batch", type=int, default=EVAL_BATCH,
                    help="nós de consulta por forward de avaliação (menor = menos memória; a saída não muda)")
    ap.add_argument("--max-ctx", type=int, default=None,
                    help="máximo de nós de contexto por passo de treino, sorteados a cada passo (avaliação usa todo o treino)")
    ap.add_argument("--train-chunks", type=int, default=TRAIN_CHUNKS,
                    help="partes da consulta de treino por passo, com acumulação de gradiente (mais = menos memória)")
    a = ap.parse_args()
    EVAL_BATCH = a.eval_batch
    TRAIN_CHUNKS = a.train_chunks
    MAX_CTX = a.max_ctx
    lib.init()
    lib.configure_torch()
    dev = torch.device("cuda")
    out_dir = R / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    patience = a.patience or None

    t = preparar(a.dataset, a.seed, dev)
    prof = montar(t).eval()
    for p in prof.parameters():
        p.requires_grad_(False)
    masks = t["dataset"].data["masks"]
    with torch.no_grad():
        prof_raw = {part: forward(prof, t, t["ens"].get_member(0), masks["train"], t["y_train"], masks[part], seed=a.seed)
                    for part in ["val", "test"]}
    prof_ens = avaliar_ensemble(prof, t)
    print(f"{a.dataset} seed={a.seed}: professor (ensemble 10) = {prof_ens:.4f}", flush=True)

    for kind in a.students.split(","):
        tag = f"{a.dataset}_k{a.prune_k}_s{a.seed}_{kind}_{a.steps}p{a.patience}" + (f"_lr{a.lr:g}" if a.lr else "") + (f"_ctx{a.max_ctx}" if a.max_ctx else "")
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
                st = montar(t, pesos=kind == "Podado", manter=manter)
                try:
                    d, _ = kd_train(st, prof, t, lr, SWEEP_STEPS, prof_raw, f"{kind} lr={lr:g}")
                    v = d.kd_val.iloc[-1]
                    sweep[lr] = v if np.isfinite(v) else np.inf
                except torch.OutOfMemoryError:
                    sweep[lr] = np.inf
                del st; torch.cuda.empty_cache()
            best_lr = min(sweep, key=sweep.get)
            print(f"{tag}: varredura {sweep} → lr = {best_lr:g}", flush=True)
        st = montar(t, pesos=kind == "Podado", manter=manter)
        antes = salvo["antes"] if salvo else avaliar_ensemble(st, t)
        d, best = kd_train(st, prof, t, best_lr, a.steps, prof_raw, kind, patience=patience, estado_path=estado,
                           extra=dict(lr=best_lr, sweep=sweep, antes=antes))
        ultimo = avaliar_ensemble(st, t)
        st.load_state_dict(best["state"])
        melhor = avaliar_ensemble(st, t)
        if a.salvar:
            torch.save(dict(state=best["state"], manter=manter, kind=kind, lr=best_lr, dataset=a.dataset, seed=a.seed,
                            melhor_passo=best["step"]), out_dir / f"student_{tag}.pt")
        info = dict(dataset=a.dataset, task=t["tipo"], metric=METRICA[t["tipo"]], seed=a.seed, prune_k=a.prune_k,
                    student=kind, camadas=manter, params=n_params(st), params_teacher=n_params(prof), lr=best_lr,
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
