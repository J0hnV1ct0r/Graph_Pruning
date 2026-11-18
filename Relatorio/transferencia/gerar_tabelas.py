"""
Gera as tabelas LaTeX do relatório de transferência entre datasets (EXP-013) a partir dos CSVs de resultados.

Parte 1 (ordem de poda transferida): retido no teste do ALVO, por k, para a ordem de BI do próprio alvo, a ordem de
BI de cada outro dataset (ORIGEM) e a ordem aleatória (mediana de 3). Retido = 100·(s − acaso)/(inteiro − acaso),
pareado por semente, média de 3 sementes. Ordem de BI: token-alvo no GraphPFN e no G2T-LimiX (a forma recomendada
no relatório principal); de todos os tokens/nós no NodePFN e no Ephris (as duas formas coincidem).

Parte 2 (student do KD transferido): retido no alvo do student destilado na origem, sem novo treino.

Uso: ~/miniconda3/envs/ephris/bin/python Relatorio/transferencia/gerar_tabelas.py
Saída: Relatorio/transferencia/{tab_parte1.tex, tab_parte1_resumo.tex, tab_bi_rho.tex, tab_parte2.tex, numeros.json}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
W = RAIZ / "Block_Influence_results"
RM = RAIZ / "Block_Influence_results"
ACASO = {"tolokers-2": 0.218, "wiki-cs": 0.230, "city-roads-M": 0.0}
MODELOS = {
    "GraphPFN": dict(base=W, f2="graphfpn_fase2", f3="graphfpn_fase3", tr="transferencia_poda/graphpfn",
                     ds=["tolokers-2", "wiki-cs", "city-roads-M"], estr="bi_alvo", kd="graphfpn_kd_transfer"),
    "G2T-LimiX": dict(base=W, f2="g2t_fase2", f3="g2t_fase3", tr="transferencia_poda/g2t",
                      ds=["tolokers-2", "wiki-cs", "city-roads-M"], estr="bi_alvo", kd="g2t_kd_transfer"),
    "NodePFN": dict(base=W, f2="nodepfn_fase2", f3="nodepfn_fase3", tr="transferencia_poda/nodepfn",
                    ds=["tolokers-2", "wiki-cs"], estr="bi_todos", kd="nodepfn_kd_transfer"),
    "Ephris": dict(base=RM, f2="ephris_fase2", f3="ephris_fase3", tr="transferencia_poda/ephris",
                   ds=["tolokers-2", "wiki-cs"], estr="bi_todos", kd="ephris_kd_transfer"),
}
KS = range(1, 7)
METADE = {"GraphPFN": 6, "G2T-LimiX": 6, "NodePFN": 6, "Ephris": 5}  # 50% das camadas
NOME_DS = {"tolokers-2": "tolokers-2", "wiki-cs": "wiki-cs", "city-roads-M": "city-roads-M"}


def fmt(v):
    if v is None or not np.isfinite(v):
        return "---"
    s = f"{v:.0f}".replace("-", "$-$")
    return s


def baseline(m, c, ds):
    b = pd.read_csv(c["base"] / c["f2"] / ds / "baseline.csv").set_index("seed")
    acaso = float(b.acaso_test.mean()) if "acaso_test" in b else ACASO[ds]
    return b.test, acaso


def propria_e_aleatoria(m, c, ds):
    """{estrategia: {k: retido médio}} das ordens da fase 3 do próprio alvo."""
    base, ac = baseline(m, c, ds)
    if m == "Ephris":
        p = pd.read_csv(c["base"] / c["f3"] / ds / "poda.csv", dtype={"removidas": str})
    else:
        plano = pd.read_csv(c["base"] / c["f3"] / ds / "plano.csv", dtype={"removidas": str})
        av = pd.read_csv(c["base"] / c["f3"] / ds / "avaliacoes.csv", dtype={"removidas": str})
        p = plano.merge(av[["removidas", "seed", "test"]], on="removidas")
    p["ret"] = 100 * (p.test - ac) / (p.seed.map(base) - ac)
    cam = p[p.estrategia == c["estr"]].groupby("k").removidas.first()
    return p.groupby(["estrategia", "k"]).ret.mean().unstack("k"), cam


def transferidas(m, c, ds):
    base, ac = baseline(m, c, ds)
    t = pd.read_csv(c["base"] / c["tr"] / "avaliacoes.csv")
    t = t[(t.alvo == ds) & (t.origem != ds) & (t.estrategia == c["estr"])].copy()
    t["ret"] = 100 * (t.test - ac) / (t.seed.map(base) - ac)
    n = t.groupby(["origem", "k"]).seed.nunique().unstack("k")
    r = t.groupby(["origem", "k"]).ret.mean().unstack("k")
    cam = t.groupby(["origem", "k"]).removidas.first()
    return r.where(n == 3), cam  # só células completas (3 sementes)


def rho_bi(m, c):
    """Spearman entre os rankings de BI de cada par de datasets (camadas 1..L−1)."""
    col = "cos_tgt" if c["estr"] == "bi_alvo" else "cos_all"
    bis = {ds: pd.read_csv(c["base"] / c["f2"] / ds / "block_influence.csv").groupby("layer")[col].mean().drop(index=0)
           for ds in c["ds"]}
    out = []
    for i, a in enumerate(c["ds"]):
        for b in c["ds"][i + 1:]:
            out.append((a, b, spearmanr(bis[a], bis[b]).statistic))
    return out


numeros = {}
linhas, resumo, rhos, topk = [], [], [], []
for m, c in MODELOS.items():
    numeros[m] = {}
    for ds in c["ds"]:
        pa, cam_p = propria_e_aleatoria(m, c, ds)
        tr, cam_t = transferidas(m, c, ds)
        topk.append((m, ds, c, pa, cam_p, tr, cam_t))
        prop = pa.loc[c["estr"]]
        alea = pa.loc[[e for e in pa.index if e.startswith("aleatoria")]].median()
        rotulo = "BI do alvo" if c["estr"] == "bi_alvo" else "BI"
        bloco = [("própria", prop)] + [(f"de {o}", tr.loc[o]) for o in tr.index] + [("aleatória", alea)]
        for j, (nome, serie) in enumerate(bloco):
            pre = f"\\multirow{{{len(bloco)}}}{{*}}{{{m}}} & \\multirow{{{len(bloco)}}}{{*}}{{{ds}}}" if j == 0 else " & "
            linhas.append(f"{pre} & {nome} & " + " & ".join(fmt(serie.get(k, np.nan)) for k in KS) + " \\\\")
        linhas.append("\\midrule")
        km = METADE[m]
        numeros[m][ds] = {"propria": {k: float(prop.get(k, np.nan)) for k in KS},
                          "transferida": {o: {k: float(tr.loc[o].get(k, np.nan)) for k in KS} for o in tr.index},
                          "aleatoria": {k: float(alea.get(k, np.nan)) for k in KS}}
        t3 = tr[3].mean() if 3 in tr else np.nan
        tm = tr[km].mean() if km in tr else np.nan
        resumo.append((m, ds, prop.get(3), t3, alea.get(3), prop.get(km), tm, alea.get(km)))
linhas = linhas[:-1]
(AQUI / "tab_parte1.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")

res = pd.DataFrame(resumo, columns=["m", "ds", "p3", "t3", "a3", "pm", "tm", "am"])
lin = []
for m, g in res.groupby("m", sort=False):
    for r in g.itertuples():
        lin.append(f"{m} & {r.ds} & {fmt(r.p3)} & {fmt(r.t3)} & {fmt(r.a3)} & {fmt(r.pm)} & {fmt(r.tm)} & {fmt(r.am)} \\\\")
    lin.append("\\midrule")
med = res[["p3", "t3", "a3", "pm", "tm", "am"]].mean()
(AQUI / "tab_parte1_resumo.tex").write_text("\n".join(lin[:-1]) + "\n", encoding="utf-8")
numeros["_media_parte1"] = {k: float(v) for k, v in med.items()}



# Parte 2: uma tabela por tipo de student; por alvo, primeiro o student do próprio alvo
p2 = {"Podado": [], "Aleatório": []}
for m, c in MODELOS.items():
    f = c["base"] / c["kd"] / "transferencia.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    if "student" not in d:
        d["student"] = "Podado"
    g = d.groupby(["student", "origem", "alvo"]).agg(n=("seed", "nunique"), sem=("ret_sem_kd", "mean"),
                                                     com=("ret_student", "mean")).reset_index()
    numeros.setdefault("_parte2", {})[m] = g.to_dict("records")
    for kind in p2:
        gk = g[g.student == kind]
        if gk.empty:
            continue
        bloco = []
        for alvo in c["ds"]:
            ga = gk[gk.alvo == alvo]
            if ga.empty:
                continue
            ga = pd.concat([ga[ga.origem == alvo], ga[ga.origem != alvo]])
            for j, r in enumerate(ga.itertuples()):
                tipo = "próprio" if r.origem == alvo else f"de {r.origem}"
                nota = "" if r.n == 3 else f"$^{{{r.n}}}$"
                bloco.append(f"{m if not bloco else ''} & {alvo if j == 0 else ''} & {tipo} & {fmt(r.sem)} & {fmt(r.com)}{nota} \\\\")
        p2[kind] += bloco + ["\\midrule"]
for kind, nome in [("Podado", "podado"), ("Aleatório", "aleatorio")]:
    (AQUI / f"tab_parte2_{nome}.tex").write_text("\n".join(p2[kind][:-1]) + "\n" if p2[kind] else "", encoding="utf-8")
json.dump(numeros, open(AQUI / "numeros.json", "w"), indent=1, default=float)
print(res.round(0).to_string())
print("média:", med.round(1).to_dict())
print("parte 2:", {k: len(v) for k, v in p2.items()})
