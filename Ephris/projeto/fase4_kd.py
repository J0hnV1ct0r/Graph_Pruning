"""
Fase 4 do Ephris: KD do student podado × student aleatório, no protocolo do KD v2 do GraphPFN / fase4_kd.py do
G2T-LimiX e do NodePFN.

Protocolo (igual aos PFNs):
  - professor = modelo inteiro congelado; loss sem rótulos: T²·KL(student ‖ professor), T = 2, nos nós de consulta;
  - a cada passo, até 1024 nós de treino viram consulta (CandidateQueue: ciclo sem reposição) e o resto do treino
    é o contexto (≥ 75%); professor e student recebem o mesmo contexto;
  - Adam; lr por student numa varredura de 20 passos em {1e-4, 3e-4, 1e-3} pela loss de KD na validação;
  - até 200 passos; avaliação a cada 10 na validação (ICL com todo o treino como contexto); melhor ponto pela
    métrica de validação; parada após 4 avaliações sem melhora; o teste só é reportado.

Adaptação ao Ephris (o que é podado é a pilha de 10 camadas MP_ICL, 58% dos parâmetros):
  - student podado: as 5 camadas MP_ICL de menor BI de todos os nós (média da Fase 2) removidas, camada 0 protegida
    (−50% das camadas ICL; mesma regra do KD do G2T-LimiX e do NodePFN, que usam o BI de todos os tokens);
  - student aleatório: mesma arquitetura (5 camadas MP_ICL) com a pilha ICL (camadas e nós globais) reinicializada;
  - nos dois, o codificador de features (tokenizador, 3 blocos de refinamento, compressor) é o pré-treinado e fica
    congelado: ele não é podado e é idêntico ao do professor, então a sua saída é calculada uma vez por passo e
    compartilhada. Treinam-se a pilha ICL, a embedding de rótulos, as normas e a cabeça.

Uso: python projeto/fase4_kd.py --dataset tolokers-2 --seed 0 [--steps 200 --patience 4]
Saída: Block_Influence_results/ephris_fase4_kd/kd_{log,resumo}_<ds>_k5_s<seed>_<student>_<steps>p<patience>.{csv,json}
"""
import argparse
import copy
import functools
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from comum import R, Tarefa, acaso, metrica, salvar_json
from ephris.data.permutation import make_inference_permutation
from ephris.execution.feature_encoding import encode_cached_or_replay
from ephris.execution.precision import bf16_autocast
from ephris.execution.runner import _feature_type_ids, _make_chunk_budget, _permute_labels
from ephris.nn.graph import GraphBlock
from ephris.nn.layers.fused_linear import fused_linear
from ephris.nn.layers.weights import LayerNorm, Linear
import torch.utils.checkpoint

PROTECTED = {0}
TRAIN_BATCH = 1024
MIN_CONTEXT_RATIO = 0.75
TEMPERATURE = 2.0
EVAL_EVERY = 10
LR_GRID = [1e-4, 3e-4, 1e-3]
SWEEP_STEPS = 20
EDGE_CHUNK = 131072  # arestas por bloco na atenção de treino (só memória; o resultado não muda)
OUT = R / "ephris_fase4_kd"


class CandidateQueue:
    """Igual à do GraphPFN/G2T-FM: percorre os nós de treino em ordem aleatória, sem reposição dentro do ciclo."""

    def __init__(self, n, n_candidates, gen):
        self.n, self.k, self.gen, self.buf = n, n_candidates, gen, torch.empty(0, dtype=torch.long)

    def __next__(self):
        while len(self.buf) < self.k:
            self.buf = torch.cat([self.buf, torch.randperm(self.n, generator=self.gen)])
        out, self.buf = self.buf[: self.k], self.buf[self.k:]
        return out


class Partes:
    """Forward do Ephris em duas partes (codificador congelado | pilha ICL + cabeça), igual a
    ephris.execution.runner.infer_node_logits, mas com gradiente na segunda parte."""

    def __init__(self, t):
        self.t, self.m = t, t.model
        self.dev = t.prepared.device
        self.x = t.prepared.features
        self.N, self.C = self.x.shape[0], t.n_classes
        perm = make_inference_permutation(x=self.x, num_outputs=self.C)
        self.cperm = perm.class_permutation.to(self.dev)
        self.forder = perm.feature_order
        self.ftype = _feature_type_ids(self.m, feature_dim=int(self.x.size(1)))
        self.budget = _make_chunk_budget(self.m, self.dev)
        self.edges = self.m.edge_cache.prepare(t.context.data.edge_index, num_nodes=self.N, device=self.dev)
        self.y = torch.from_numpy(t.y).to(self.dev)
        self.stream = self.m.should_stream_feature_tokens(self.x, chunk_budget=self.budget)

    def rotulos(self, ctx_mask):
        lab = torch.where(ctx_mask, self.y, torch.full_like(self.y, -1))
        return _permute_labels(lab, self.cperm, num_outputs=self.C)

    @torch.no_grad()
    def codificar(self, ctx_mask):
        lab = self.rotulos(ctx_mask)
        m = self.m
        with bf16_autocast(self.dev):
            if self.stream:
                xp, ft = m.prepare_streaming_features(x=self.x, x_is_preprocessed=True, feature_type_ids=self.ftype)
                nts = encode_cached_or_replay(m, xp, lab, ctx_mask, ft, self.edges, chunk_budget=self.budget,
                                              feature_order=self.forder)
            else:
                fs = m.tokenize_features(x=self.x, y=lab, context_mask=ctx_mask, x_is_preprocessed=True,
                                         feature_type_ids=self.ftype, chunk_budget=self.budget,
                                         feature_order=self.forder)
                nts = m.encode_features(fs, self.edges)
            h = m.flatten_node_state(nts)
        return h.detach(), lab

    def cabeca(self, model, h, lab, ctx_mask, nodes):
        """Log-probabilidades [len(nodes), C] na ordem original das classes."""
        with bf16_autocast(self.dev):
            s = model.inject_labels(h, lab, ctx_mask)
            s = model.propagate(s, self.edges)
            logits = model.project_logits(s[nodes])
        logits = logits[:, : self.C].float().index_select(1, self.cperm)
        return F.log_softmax(logits, -1)


def atencao_treino(att, x, edge_index, *, num_real_nodes, num_globals, edge_chunk_size=None):
    """Mesma conta de GraphAttention.forward (softmax por destino sobre as arestas do grafo + nós globais; cada nó
    global atende a si mesmo e a todos os nós reais), mas sem o acumulador in-place da inferência, para permitir o
    backward. Conferida contra a implementação original em testar_atencao()."""
    with torch.autocast(device_type="cuda", enabled=False):
        return _atencao_fp32(att, x, edge_index, int(num_real_nodes), int(num_globals)).to(x.dtype)


def _atencao_fp32(att, x, edge_index, R_, G):
    x = x.float()
    H, hw = att.heads, att.head_width
    N = int(x.size(0))
    q, k, v = fused_linear(x, (att.q_proj, att.k_proj, att.v_proj)).split(H * hw, dim=-1)
    q, k, v = (t_.view(N, H, hw) for t_ in (q, k, v))
    src, dst = edge_index[0].long(), edge_index[1].long()
    degree = torch.bincount(dst, minlength=N)
    degree[:R_] += G
    degree[R_:] += R_ + 1
    q = att._scale_query_by_lengths(q, degree.clamp_min(1))
    esc = math.sqrt(hw)
    # nós reais: arestas + globais
    sg = torch.einsum("rhd,ghd->rgh", q[:R_].float(), k[R_:].float()) / esc  # [R, G, H]
    with torch.no_grad():  # máximo por destino (só estabilidade numérica; o softmax não depende dele)
        mx = sg.amax(1).clone()
        for i in range(0, len(src), EDGE_CHUNK):
            s_c, d_c = src[i:i + EDGE_CHUNK], dst[i:i + EDGE_CHUNK]
            se = (q.index_select(0, d_c) * k.index_select(0, s_c)).sum(-1).float() / esc
            mx.scatter_reduce_(0, d_c.view(-1, 1).expand_as(se), se, reduce="amax", include_self=True)
        mx = mx.detach()

    def bloco(q_, k_, v_, mx_, s_, d_):
        sc = (q_.index_select(0, d_) * k_.index_select(0, s_)).sum(-1).float() / esc
        w_ = torch.exp(sc - mx_.index_select(0, d_))
        den_ = torch.zeros((R_, H), device=x.device).index_add(0, d_, w_)
        num_ = torch.zeros((R_, H, hw), device=x.device).index_add(
            0, d_, w_.unsqueeze(-1) * v_.index_select(0, s_).float())
        return den_, num_

    wg = torch.exp(sg - mx.unsqueeze(1))
    den = wg.sum(1)
    num = torch.einsum("rgh,ghd->rhd", wg, v[R_:].float())
    passo = EDGE_CHUNK
    for i in range(0, len(src), passo):
        s_c, d_c = src[i:i + passo], dst[i:i + passo]
        if torch.is_grad_enabled():
            dc, nc = torch.utils.checkpoint.checkpoint(bloco, q, k, v, mx, s_c, d_c, use_reentrant=False)
        else:
            dc, nc = bloco(q, k, v, mx, s_c, d_c)
        den, num = den + dc, num + nc
    out_r = num / den.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    # nós globais: si mesmo + todos os reais
    s_self = (q[R_:].float() * k[R_:].float()).sum(-1) / esc  # [G, H]
    s_gr = torch.einsum("ghd,rhd->grh", q[R_:].float(), k[:R_].float()) / esc  # [G, R, H]
    w = torch.softmax(torch.cat([s_self.unsqueeze(1), s_gr], 1), dim=1)
    out_g = w[:, 0].unsqueeze(-1) * v[R_:].float() + torch.einsum("grh,rhd->ghd", w[:, 1:], v[:R_].float())
    return torch.cat([out_r, out_g]).reshape(N, H * hw)


def preparar_treino(model):
    """Troca a atenção das camadas MP_ICL pela versão diferenciável e recalcula cada camada no backward
    (gradient checkpointing; mesmo resultado, menos memória)."""
    for layer in model.icl.layers:
        att = layer.attention
        att.forward = functools.partial(atencao_treino, att)
        orig = layer.forward
        if getattr(layer, "_ck", False):
            continue

        def fwd(*a, _orig=orig, **k):
            if torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(_orig, *a, use_reentrant=False, **k)
            return _orig(*a, **k)
        layer.forward = fwd
        layer._ck = True
    return model


@torch.no_grad()
def testar_atencao(t):
    """Diferença máxima entre a atenção original e a de treino numa camada do professor."""
    parts = Partes(t)
    ctx = torch.from_numpy(t.masks["train"]).to(parts.dev)
    h, lab = parts.codificar(ctx)
    m = t.model
    s = m.inject_labels(h, lab, ctx).float()
    g = torch.cat([s, m.icl.global_nodes.float()])
    att = m.icl.layers[0].attention
    kw = dict(num_real_nodes=parts.N, num_globals=m.icl.num_globals)
    a_ = att(g, parts.edges, **kw)
    b_ = atencao_treino(att, g, parts.edges, **kw)
    return float((a_ - b_).abs().max()), float(a_.abs().max())


def treinaveis(model):
    return [model.icl, model.icl_label_embedding, model.label_norm, model.final_norm, model.head]


def montar_student(kind, teacher, keep, seed):
    st = copy.copy(teacher)  # módulos do codificador compartilhados (congelados); os treináveis são copiados
    st._modules = dict(teacher._modules)
    for nome in ["icl", "icl_label_embedding", "label_norm", "final_norm", "head"]:
        st._modules[nome] = copy.deepcopy(teacher._modules[nome])
    if kind == "Podado":
        st.icl.layers = nn.ModuleList([st.icl.layers[i] for i in keep])
    else:
        g = torch.Generator().manual_seed(seed)
        icl = GraphBlock(width=teacher.node_width, heads=teacher.graph_heads, num_layers=len(keep),
                         num_globals=teacher.num_globals, scaling_hidden_width=teacher.scaling_hidden_width,
                         expansion=teacher.expansion)
        icl.global_nodes = nn.Parameter(torch.randn(icl.global_nodes.shape, generator=g) * 0.02)
        for mod in icl.modules():
            if isinstance(mod, LayerNorm):
                mod.weight = nn.Parameter(torch.ones(mod.weight.shape))
                mod.bias = nn.Parameter(torch.zeros(mod.bias.shape))
            elif isinstance(mod, Linear):  # como nn.Linear: U(−1/√fan_in, 1/√fan_in) em pesos e vieses
                b = 1 / math.sqrt(mod.in_features)
                mod.weight = nn.Parameter((torch.rand(mod.weight.shape, generator=g) * 2 - 1) * b)
                if mod.bias is not None:
                    mod.bias = nn.Parameter((torch.rand(mod.bias.shape, generator=g) * 2 - 1) * b)
        faltando = [n for n, p in icl.named_parameters() if p.is_meta]
        assert not faltando, f"parâmetros sem inicialização: {faltando}"
        st._modules["icl"] = icl.to(teacher.head.weight.device)
    for mod in treinaveis(st):
        for p in mod.parameters():
            p.data = p.data.float()
            p.requires_grad_(True)
    return preparar_treino(st)


def kd_loss(s_logp, t_logp):
    """T²·KL com temperatura sobre log-probabilidades (equivale a logits/T)."""
    zs, zt = s_logp / TEMPERATURE, t_logp / TEMPERATURE
    return F.kl_div(F.log_softmax(zs, -1), F.log_softmax(zt, -1), log_target=True,
                    reduction="batchmean") * TEMPERATURE ** 2


def run(parts, teacher, student, lr, n_steps, patience, seed, h_full, lab_full, ctx_full, t_full, log_every):
    t = parts.t
    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.from_numpy(np.flatnonzero(t.masks["train"])).to(parts.dev)
    n_train = len(train_idx)
    queue = CandidateQueue(n_train, min(TRAIN_BATCH, int(n_train * (1 - MIN_CONTEXT_RATIO))), gen)
    params = [p for mod in treinaveis(student) for p in mod.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    val_m = torch.from_numpy(t.masks["val"]).to(parts.dev)

    @torch.no_grad()
    def avaliar():
        for mod in treinaveis(student):
            mod.eval()
        lp = parts.cabeca(student, h_full, lab_full, ctx_full, torch.arange(parts.N, device=parts.dev))
        kd_val = float(kd_loss(lp[val_m], t_full[val_m]))
        lp = lp.cpu().numpy()
        return dict(kd_val=kd_val, metric_val=metrica(t.name, t.y[t.masks["val"]], lp[t.masks["val"]]),
                    metric_test=metrica(t.name, t.y[t.masks["test"]], lp[t.masks["test"]])), lp

    r0, lp0 = avaliar()
    log = [dict(step=0, t_kd=0.0, train_loss=np.nan, **r0)]
    best = dict(step=0, metric_val=r0["metric_val"], lp=lp0,
                state=[p.detach().clone() for p in params])
    sem, losses, t_kd = 0, [], 0.0
    for step in range(1, n_steps + 1):
        torch.cuda.synchronize(); tic = time.time()
        q = next(queue).to(parts.dev)
        qn = train_idx[q]
        ctx = torch.zeros(parts.N, dtype=torch.bool, device=parts.dev)
        ctx[train_idx] = True
        ctx[qn] = False
        h, lab = parts.codificar(ctx)
        with torch.no_grad():
            t_lp = parts.cabeca(teacher, h, lab, ctx, qn)
        for mod in treinaveis(student):
            mod.train()
        s_lp = parts.cabeca(student, h, lab, ctx, qn)
        loss = kd_loss(s_lp, t_lp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(); t_kd += time.time() - tic
        losses.append(float(loss))
        if not np.isfinite(losses[-1]):
            print(f"  loss não finita no passo {step}", flush=True)
            break
        if step % log_every == 0 or step == n_steps:
            r, lp = avaliar()
            log.append(dict(step=step, t_kd=t_kd, train_loss=float(np.mean(losses[-log_every:])), **r))
            if log_every == EVAL_EVERY:
                print(f"  passo {step}: kd_val={r['kd_val']:.4f} metric_val={r['metric_val']:.4f}", flush=True)
            if r["metric_val"] > best["metric_val"]:
                best = dict(step=step, metric_val=r["metric_val"], lp=lp,
                            state=[p.detach().clone() for p in params])
                sem = 0
            else:
                sem += 1
                if patience and sem >= patience:
                    print(f"  parada antecipada no passo {step} (melhor: {best['step']})", flush=True)
                    break
    return pd.DataFrame(log), best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prune-k", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=4, help="0 = sem parada antecipada")
    ap.add_argument("--students", default="Podado,Aleatório")
    ap.add_argument("--lr", type=float, default=None, help="pula a varredura e usa este lr")
    ap.add_argument("--out", default=None, help="pasta de saída em Block_Influence_results (padrão: ephris_fase4_kd)")
    ap.add_argument("--salvar", action="store_true", help="salva o melhor student em student_<tag>.pt")
    a = ap.parse_args()
    if a.out:
        OUT = R / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    name, seed = a.dataset, a.seed

    bi = pd.read_csv(R / "ephris_fase2" / name / "block_influence.csv").groupby("layer").cos_all.mean()
    removidas = [l for l in bi.sort_values(kind="stable").index if l not in PROTECTED][: a.prune_k]
    keep = [i for i in range(len(bi)) if i not in removidas]

    t = Tarefa(name, seed)
    torch.manual_seed(seed)
    teacher = t.model.eval()
    parts = Partes(t)
    ctx_full = torch.from_numpy(t.masks["train"]).to(parts.dev)
    h_full, lab_full = parts.codificar(ctx_full)
    with torch.no_grad():
        t_full = parts.cabeca(teacher, h_full, lab_full, ctx_full, torch.arange(parts.N, device=parts.dev))
    tl = t_full.cpu().numpy()
    prof = {p: metrica(name, t.y[t.masks[p]], tl[t.masks[p]]) for p in ["val", "test"]}
    ac = acaso(name, t.y[t.masks["test"]])
    print(f"{name} s{seed}: professor val={prof['val']:.4f} test={prof['test']:.4f}; mantidas {keep}", flush=True)

    for kind in a.students.split(","):
        tag = f"{name}_k{a.prune_k}_s{seed}_{kind}_{a.steps}p{a.patience}"
        if (OUT / f"kd_resumo_{tag}.json").exists():
            print(f"(já feito: {tag})")
            continue
        sweep = {}
        for lr in ([] if a.lr else LR_GRID):
            st = montar_student(kind, teacher, keep, seed)
            try:
                d, _ = run(parts, teacher, st, lr, SWEEP_STEPS, 0, seed, h_full, lab_full, ctx_full, t_full, SWEEP_STEPS)
                v = d.kd_val.iloc[-1]
                sweep[lr] = v if np.isfinite(v) else np.inf
            except torch.OutOfMemoryError:
                sweep[lr] = np.inf
            del st
            torch.cuda.empty_cache()
        lr = a.lr or min(sweep, key=sweep.get)
        print(f"{tag}: varredura {sweep} → lr={lr:g}", flush=True)
        st = montar_student(kind, teacher, keep, seed)
        d, best = run(parts, teacher, st, lr, a.steps, a.patience, seed, h_full, lab_full, ctx_full, t_full, EVAL_EVERY)
        lpb = best["lp"]
        if a.salvar:
            params = [p for mod in treinaveis(st) for p in mod.parameters()]
            with torch.no_grad():
                for p, v in zip(params, best["state"]):
                    p.copy_(v)
            torch.save(dict(modulos={n: st._modules[n].state_dict() for n in ["icl", "icl_label_embedding", "label_norm",
                                                                              "final_norm", "head"]},
                            keep=keep, kind=kind, lr=lr, dataset=name, seed=seed, melhor_passo=best["step"]),
                       OUT / f"student_{tag}.pt")
        info = dict(dataset=name, seed=seed, student=kind, prune_k=a.prune_k, camadas_icl=keep, lr=lr,
                    sweep={f"{k:g}": v for k, v in sweep.items()}, max_steps=a.steps, patience=a.patience,
                    passos_feitos=int(d.step.iloc[-1]), melhor_passo=best["step"],
                    params_student=sum(p.numel() for p in st.parameters()),
                    params_teacher=sum(p.numel() for p in teacher.parameters()),
                    teacher_val=prof["val"], teacher_test=prof["test"], acaso_test=ac,
                    test_antes=d.metric_test.iloc[0], val_antes=d.metric_val.iloc[0],
                    test_melhor=metrica(name, t.y[t.masks["test"]], lpb[t.masks["test"]]),
                    val_melhor=best["metric_val"], test_ultimo=d.metric_test.iloc[-1],
                    t_kd=d.t_kd.iloc[-1])
        d.assign(dataset=name, student=kind, seed=seed, lr=lr).to_csv(OUT / f"kd_log_{tag}.csv", index=False)
        salvar_json(info, OUT / f"kd_resumo_{tag}.json")
        ret = lambda s: 100 * (s - ac) / (prof["test"] - ac)  # noqa: E731
        print(f"{tag}: antes {info['test_antes']:.4f} ({ret(info['test_antes']):.0f}%) → melhor "
              f"{info['test_melhor']:.4f} ({ret(info['test_melhor']):.0f}%, passo {best['step']}) | "
              f"professor {prof['test']:.4f}", flush=True)
        del st
        torch.cuda.empty_cache()
