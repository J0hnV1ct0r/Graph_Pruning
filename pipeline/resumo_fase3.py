"""Resumo da Fase 3 (poda cumulativa sem re-treino) com o que já estiver pronto em Block_Influence_results/.

Para cada modelo, dataset, estratégia e k: métrica bruta no teste e desempenho retido (%), média das sementes.
Retido = 100·(podado − acaso)/(inteiro − acaso), pareado por semente (inteiro = baseline.csv da Fase 2).
As 3 ordens aleatórias aparecem resumidas pela mediana.

Uso (da raiz do repositório): python pipeline/resumo_fase3.py
Saída: Block_Influence_results/{g2t,nodepfn}_fase3/<dataset>/curvas_poda.csv e resumo_fase3.csv na raiz de cada pasta.
"""
import pandas as pd

from fase3_ordens import ACASO, R

for modelo, f2, f3 in [("GraphPFN", "graphfpn_fase2", "graphfpn_fase3"), ("G2T-LimiX", "g2t_fase2", "g2t_fase3"),
                       ("NodePFN", "nodepfn_fase2", "nodepfn_fase3")]:
    linhas = []
    for ds in ["tolokers-2", "amazon-ratings", "wiki-cs", "city-roads-M"]:
        d = R / f3 / ds
        if not (d / "avaliacoes.csv").exists():
            continue
        plano = pd.read_csv(d / "plano.csv", dtype={"removidas": str})
        av = pd.read_csv(d / "avaliacoes.csv", dtype={"removidas": str})
        base = pd.read_csv(R / f2 / ds / "baseline.csv").set_index("seed")
        c = ACASO[ds]
        av["ret_test"] = 100 * (av.test - c) / (av.seed.map(base.test) - c)
        av["ret_val"] = 100 * (av.val - c) / (av.seed.map(base.val) - c)
        m = plano.merge(av, on="removidas", how="left", suffixes=("", "_av"))
        cur = (m.groupby(["estrategia", "k", "removidas"])
               .agg(test=("test", "mean"), ret_test=("ret_test", "mean"), ret_val=("ret_val", "mean"), n=("seed", "count"))
               .reset_index().sort_values(["estrategia", "k"]))
        cur.insert(0, "dataset", ds)
        cur.to_csv(d / "curvas_poda.csv", index=False)
        cur["grupo"] = cur.estrategia.where(~cur.estrategia.str.startswith("aleatoria"), "aleatoria (mediana)")
        tab = cur.groupby(["grupo", "k"]).ret_test.median().unstack("k").round(0)
        print(f"== {modelo} · {ds}: inteiro {base.test.mean():.4f}; retido no teste (%) por k"
              f"{'' if (cur.n == len(base)).all() else '  [parcial]'}")
        print(tab.to_string())
        for _, r in cur.iterrows():
            linhas.append(dict(modelo=modelo, inteiro=base.test.mean(), **r.drop("grupo").to_dict()))
    if linhas:
        pd.DataFrame(linhas).to_csv(R / f3 / "resumo_fase3.csv", index=False)
