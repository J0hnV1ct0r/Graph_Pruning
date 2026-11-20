"""
Transferência do KD (Ephris): o student podado destilado no dataset ORIGEM é avaliado, sem novo treino, nos outros
datasets (ALVO), por ICL com o contexto rotulado do alvo. Responde se o KD gera um PFN menor e geral ou um modelo
específico do dataset em que foi destilado.

Para cada student salvo (fase4_kd.py --salvar) e cada alvo, na mesma semente do student:
  - professor: modelo inteiro;
  - podado sem KD: as mesmas camadas MP_ICL do student, pesos originais;
  - student: as mesmas camadas, pesos destilados na origem.
Avaliação pelo mesmo caminho do KD (Partes.cabeca, contexto = todo o treino), como em test_melhor de fase4_kd.py.

Uso: python projeto/kd_transferencia.py [--pasta ephris_kd_transfer] [--alvos tolokers-2,wiki-cs,...]
Saída: Block_Influence_results/<pasta>/transferencia.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch

from comum import R, Tarefa, acaso, metrica
from fase4_kd import Partes, montar_student

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default="ephris_kd_transfer")
    ap.add_argument("--alvos", default="tolokers-2,wiki-cs")
    a = ap.parse_args()
    pasta = R / a.pasta
    out = pasta / "transferencia.csv"
    feitos = pd.read_csv(out) if out.exists() else pd.DataFrame()
    students = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(pasta.glob("student_*.pt"))]
    for alvo in a.alvos.split(","):
        for seed in sorted({s["seed"] for s in students}):
            daqui = [s for s in students if s["seed"] == seed]
            ja = lambda s: len(feitos) and ((feitos.alvo == alvo) & (feitos.origem == s["dataset"])  # noqa: E731
                                            & (feitos.seed == seed) & (feitos.student == s["kind"])).any()
            if all(ja(s) for s in daqui):
                continue
            t = Tarefa(alvo, seed)
            torch.manual_seed(seed)
            teacher = t.model.eval()
            parts = Partes(t)
            ctx = torch.from_numpy(t.masks["train"]).to(parts.dev)
            h, lab = parts.codificar(ctx)
            todos = torch.arange(parts.N, device=parts.dev)

            @torch.no_grad()
            def medir(m):
                lp = parts.cabeca(m, h, lab, ctx, todos).cpu().numpy()
                return {p: metrica(alvo, t.y[t.masks[p]], lp[t.masks[p]]) for p in ["val", "test"]}

            prof = medir(teacher)
            ac = acaso(alvo, t.y[t.masks["test"]])
            for s in daqui:
                if ja(s):
                    continue
                pod = montar_student(s["kind"], teacher, s["keep"], seed)
                for mod in pod._modules.values():
                    mod.eval()
                sem_kd = medir(pod)
                for nome, sd in s["modulos"].items():
                    pod._modules[nome].load_state_dict(sd)
                com_kd = medir(pod)
                del pod
                torch.cuda.empty_cache()
                ret = lambda v: 100 * (v - ac) / (prof["test"] - ac)  # noqa: E731
                linha = dict(modelo="Ephris", student=s["kind"], origem=s["dataset"], alvo=alvo, seed=seed, camadas=" ".join(map(str, s["keep"])),
                             acaso=ac, professor_val=prof["val"], professor_test=prof["test"],
                             sem_kd_val=sem_kd["val"], sem_kd_test=sem_kd["test"],
                             student_val=com_kd["val"], student_test=com_kd["test"],
                             ret_sem_kd=ret(sem_kd["test"]), ret_student=ret(com_kd["test"]))
                feitos = pd.concat([feitos, pd.DataFrame([linha])], ignore_index=True)
                feitos.to_csv(out, index=False)
                print(f"{s['kind']} {s['dataset']} → {alvo} s{seed}: professor {prof['test']:.4f} | sem KD {sem_kd['test']:.4f} "
                      f"({linha['ret_sem_kd']:.0f}%) | student {com_kd['test']:.4f} ({linha['ret_student']:.0f}%)", flush=True)
            del t, teacher, parts, h, lab
            torch.cuda.empty_cache()
