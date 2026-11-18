"""
Transferência do KD (GraphPFN, G2T-LimiX, NodePFN): o student podado destilado no dataset ORIGEM é avaliado, sem
novo treino, nos outros datasets (ALVO), por ICL com o contexto rotulado do alvo. Responde se o KD gera um PFN menor
e geral ou um modelo específico do dataset em que foi destilado.

Para cada student salvo (fase4_kd.py / kd_v2_graphpfn.py com --salvar) e cada alvo, na mesma semente do student,
com o ensemble completo do pipeline de cada modelo (o mesmo de test_ens_melhor no KD):
  - professor: modelo inteiro;
  - sem KD: a mesma arquitetura do student antes do KD (podado: pesos originais; aleatório: inicialização);
  - student: pesos destilados na origem.
Students podado e aleatório (todos os student_*.pt da pasta).

Uso (da raiz do repositório, no ambiente de cada modelo):
    ~/venvs/g2tfm/bin/python              pipeline/kd_transferencia.py --modelo g2t
    ~/venvs/nodepfn/bin/python            pipeline/kd_transferencia.py --modelo nodepfn
    ~/miniconda3/envs/graphpfn/bin/python pipeline/kd_transferencia.py --modelo graphpfn
Saída: Block_Influence_results/<pasta do modelo>/transferencia.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "pipeline"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fase3_ordens import ACASO  # noqa: E402

R = RAIZ / "Block_Influence_results"
CFG = {
    "graphpfn": dict(nome="GraphPFN", pasta="graphfpn_kd_transfer",
                     alvos=["tolokers-2", "wiki-cs", "city-roads-M"]),
    "g2t": dict(nome="G2T-LimiX", pasta="g2t_kd_transfer",
                alvos=["tolokers-2", "wiki-cs", "city-roads-M"]),
    "nodepfn": dict(nome="NodePFN", pasta="nodepfn_kd_transfer",
                    alvos=["tolokers-2", "wiki-cs"]),
}


def adaptador(modelo):
    """(preparar(alvo, seed) -> t, professor(t), avaliar(m, t) -> métrica no teste, podado(t, prof, manter, seed))."""
    if modelo == "g2t":
        os.chdir(RAIZ / "G2T-FM")
        sys.path.insert(0, str(RAIZ / "G2T-FM" / "projeto"))
        import fase4_kd as k
        k.lib.init()
        k.lib.configure_torch()
        k.EVAL_BATCH = 2048  # o padrão (2**15) estoura 16 GB no wiki-cs; a saída não depende do bloco
        return (lambda alvo, seed: k.preparar(alvo, seed, torch.device("cuda")),
                lambda t: k.montar(t).eval(),
                k.avaliar_ensemble,
                lambda t, prof, manter, seed, kind: k.montar(t, pesos=kind == "Podado", manter=manter).eval())
    if modelo == "nodepfn":
        os.chdir(RAIZ / "NodePFN")
        sys.path.insert(0, str(RAIZ / "NodePFN" / "projeto"))
        import avaliar_projeto as ap_
        import fase4_kd as k

        def preparar(alvo, seed):
            hp = json.load(open(ap_.RESULTS / f"{alvo}.json"))["hp"]
            return k.preparar(alvo, seed, hp)
        return (preparar,
                lambda t: t["clf"].model[2].to(k.DEV).eval(),
                k.avaliar_ensemble,
                lambda t, prof, manter, seed, kind: k.montar(prof, kind, manter, seed).eval())
    if modelo == "graphpfn":
        sys.path.insert(0, str(RAIZ / "GraphFPN" / "projeto"))
        import kd_v2_graphpfn as k  # faz chdir para GraphFPN/

        def preparar(alvo, seed):
            k.SEED = seed  # evaluate_ensemble usa a semente global
            return k.prepare_task(alvo, k.ALL_DATASETS[alvo])

        def professor(t):
            m = k.enable_low_memory(k.load_pretrained()).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            return m
        return (preparar, professor, k.evaluate_ensemble,
                lambda t, prof, manter, seed, kind: k.enable_low_memory(k.keep_layers(
                    k.load_pretrained() if kind == "Podado" else k.load_random(seed), manter)).eval())
    raise ValueError(modelo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=list(CFG))
    ap.add_argument("--alvos", default=None, help="lista separada por vírgula (padrão: todos do modelo)")
    a = ap.parse_args()
    c = CFG[a.modelo]
    pasta = R / c["pasta"]
    out = pasta / "transferencia.csv"
    feitos = pd.read_csv(out) if out.exists() else pd.DataFrame()
    students = [torch.load(f, map_location="cpu", weights_only=False)
                for f in sorted(pasta.glob("student_*.pt"))]
    print(f"{len(students)} students: {sorted({(s['dataset'], s['seed'], s['kind']) for s in students})}", flush=True)
    ja = lambda alvo, s: len(feitos) and ((feitos.alvo == alvo) & (feitos.origem == s["dataset"])  # noqa: E731
                                          & (feitos.seed == s["seed"]) & (feitos.student == s["kind"])).any()
    preparar = professor = avaliar = podado = None
    for alvo in (a.alvos.split(",") if a.alvos else c["alvos"]):
        for seed in sorted({s["seed"] for s in students}):
            daqui = [s for s in students if s["seed"] == seed]
            pend = [s for s in daqui if not ja(alvo, s)]
            if not pend:
                continue
            if preparar is None:
                preparar, professor, avaliar, podado = adaptador(a.modelo)
            t = preparar(alvo, seed)
            prof = professor(t)
            torch.manual_seed(seed)
            prof_test = avaliar(prof, t)
            ac = ACASO[alvo]
            ret = lambda v: 100 * (v - ac) / (prof_test - ac)  # noqa: E731
            for s in pend:
                m = podado(t, prof, s["manter"], seed, s["kind"])
                torch.manual_seed(seed)
                sem_kd = avaliar(m, t)
                m.load_state_dict(s["state"])
                m.eval()
                torch.manual_seed(seed)
                com_kd = avaliar(m, t)
                del m
                torch.cuda.empty_cache()
                linha = dict(modelo=c["nome"], student=s["kind"], origem=s["dataset"], alvo=alvo, seed=seed,
                             camadas=" ".join(map(str, s["manter"])), acaso=ac, professor_test=prof_test,
                             sem_kd_test=sem_kd, student_test=com_kd, ret_sem_kd=ret(sem_kd), ret_student=ret(com_kd))
                feitos = pd.concat([feitos, pd.DataFrame([linha])], ignore_index=True)
                feitos.to_csv(out, index=False)
                print(f"{s['kind']} {s['dataset']} → {alvo} s{seed}: professor {prof_test:.4f} | sem KD {sem_kd:.4f} "
                      f"({linha['ret_sem_kd']:.0f}%) | student {com_kd:.4f} ({linha['ret_student']:.0f}%)", flush=True)
            del t, prof
            torch.cuda.empty_cache()
