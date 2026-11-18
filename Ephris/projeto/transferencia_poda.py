"""
Transferência da ordem de poda (Ephris): as camadas removidas no dataset ALVO são escolhidas pelo BI medido em
outro dataset (ORIGEM). Responde se a ordem de BI é do modelo ou do dataset.

- ordens: bi_todos e bi_alvo de ephris_fase2/<origem> (média das 3 sementes, camada 0 protegida), k = 1..6;
- avaliação no alvo, sem re-treino, sementes 0, 1, 2 (mesmo pipeline da fase 3);
- conjuntos já avaliados na fase 3 do alvo são reaproveitados (fonte = "fase3").

Uso: python projeto/transferencia_poda.py [--datasets tolokers-2,wiki-cs]
Saída: Block_Influence_results/transferencia_poda/ephris/avaliacoes.csv
"""
import argparse

import pandas as pd

from comum import R, SEEDS, Tarefa
from fase23 import KS, Podador, ordens

OUT = R / "transferencia_poda" / "ephris" / "avaliacoes.csv"


def ordens_de(ds):
    d2 = R / "ephris_fase2" / ds
    base = pd.read_csv(d2 / "baseline.csv")
    o = ordens(pd.read_csv(d2 / "block_influence.csv"), pd.read_csv(d2 / "ablacao_individual.csv"),
               base.set_index("seed").val, float(base.acaso_val.mean()))
    return {e: o[e] for e in ["bi_todos", "bi_alvo"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="tolokers-2,wiki-cs")
    a = ap.parse_args()
    dss = a.datasets.split(",")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feitos = pd.read_csv(OUT, dtype={"removidas": str}) if OUT.exists() else pd.DataFrame()
    for alvo in dss:
        f3 = pd.read_csv(R / "ephris_fase3" / alvo / "poda.csv", dtype={"removidas": str})
        cache = {(r.removidas, r.seed): (r.val, r.test) for r in f3.itertuples()}
        tarefas = {}
        for origem in [d for d in dss if d != alvo]:
            for estr, o in ordens_de(origem).items():
                for k in KS:
                    rem = tuple(sorted(o[:k]))
                    chave = " ".join(map(str, rem))
                    for seed in SEEDS:
                        if len(feitos) and ((feitos.alvo == alvo) & (feitos.origem == origem) & (feitos.estrategia == estr)
                                            & (feitos.k == k) & (feitos.seed == seed)).any():
                            continue
                        if (chave, seed) in cache:
                            (val, test), fonte = cache[(chave, seed)], "fase3"
                        else:
                            if seed not in tarefas:
                                t = Tarefa(alvo, seed)
                                tarefas[seed] = (t, Podador(t.model))
                            t, pod = tarefas[seed]
                            pod.remover(rem)
                            r = t.avaliar()
                            pod.restaurar()
                            (val, test), fonte = (r["val"], r["test"]), "novo"
                            cache[(chave, seed)] = (val, test)
                        linha = dict(modelo="Ephris", alvo=alvo, origem=origem, estrategia=estr, k=k, removidas=chave,
                                     seed=seed, val=val, test=test, fonte=fonte)
                        feitos = pd.concat([feitos, pd.DataFrame([linha])], ignore_index=True)
                        feitos.to_csv(OUT, index=False)
                        print(linha, flush=True)
        tarefas.clear()
