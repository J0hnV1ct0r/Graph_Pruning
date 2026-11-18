"""
Transferência da ordem de poda (GraphPFN, G2T-LimiX, NodePFN): as camadas removidas no dataset ALVO são escolhidas
pelo BI medido em outro dataset (ORIGEM). Responde se a ordem de BI é do modelo ou do dataset.

- ordens: bi_todos e bi_alvo de <modelo>_fase2/<origem> (média das 3 sementes, camada 0 protegida), k = 1..6;
  com origem = alvo (ordem própria) também as 3 ordens aleatórias de fase3_ordens, para os datasets novos que não
  têm fase 3;
- avaliação no alvo, sem re-treino, sementes 0, 1, 2, com o mesmo pipeline da fase 3 de cada modelo;
- conjuntos já avaliados na fase 3 (ou na ablação da fase 2) do alvo são reaproveitados (fonte = "fase3").

Uso (da raiz do repositório, no ambiente de cada modelo):
    ~/miniconda3/envs/graphpfn/bin/python pipeline/transferencia_poda.py --modelo graphpfn
    ~/venvs/g2tfm/bin/python              pipeline/transferencia_poda.py --modelo g2t
    ~/venvs/nodepfn/bin/python            pipeline/transferencia_poda.py --modelo nodepfn
Saída: Block_Influence_results/transferencia_poda/<modelo>/avaliacoes.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "pipeline"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fase3_ordens as f3  # noqa: E402

R = RAIZ / "Block_Influence_results"
CFG = {
    "graphpfn": dict(nome="GraphPFN", fase2="graphfpn_fase2", fase3="graphfpn_fase3",
                     datasets=["tolokers-2", "wiki-cs", "city-roads-M"]),
    "g2t": dict(nome="G2T-LimiX", fase2="g2t_fase2", fase3="g2t_fase3",
                datasets=["tolokers-2", "wiki-cs", "city-roads-M"]),
    "nodepfn": dict(nome="NodePFN", fase2="nodepfn_fase2", fase3="nodepfn_fase3",
                    datasets=["tolokers-2", "wiki-cs"]),
}


def ordens(fase2, ds, com_aleatorias):
    """Ordens pelo BI (sem precisar da ablação) e, se pedido, as 3 aleatórias de fase3_ordens (mesma semente)."""
    bi = pd.read_csv(R / fase2 / ds / "block_influence.csv").groupby("layer").mean(numeric_only=True).drop(index=0)
    out = {"bi_todos": [int(c) for c in bi.cos_all.sort_values(kind="stable").index],
           "bi_alvo": [int(c) for c in bi.cos_tgt.sort_values(kind="stable").index]}
    if com_aleatorias:
        rng = np.random.default_rng(0)
        for r in range(f3.N_ALEATORIAS):
            out[f"aleatoria_{r}"] = [int(c) for c in rng.permutation(np.arange(1, 12))]
    return out


def avaliador(modelo):
    """Retorna avaliar(alvo, seed, conj) -> (val, test) no pipeline da fase 3 do modelo."""
    if modelo == "graphpfn":
        sys.path.insert(0, str(RAIZ / "GraphFPN" / "projeto"))
        import fase23_graphpfn as g  # faz chdir para GraphFPN/ (checkpoints)
        estado = {}

        def avaliar(alvo, seed, conj):
            if estado.get("alvo") != alvo:
                estado.update(alvo=alvo, t=g.prepare_task(alvo))
            estado.setdefault("base", g.load_base())
            return g.avaliar(estado["base"], estado["t"], seed, remove=conj)
        return avaliar
    if modelo == "g2t":
        os.chdir(RAIZ / "G2T-FM")
        sys.path.insert(0, str(RAIZ / "G2T-FM" / "projeto"))
        import fase2_bi_ablacao as f2
        f2.lib.init()
        f2.lib.configure_torch()

        def avaliar(alvo, seed, conj):
            val, test, _ = f2.rodar(alvo, seed, remover=conj)
            return val, test
        return avaliar
    if modelo == "nodepfn":
        os.chdir(RAIZ / "NodePFN")
        sys.path.insert(0, str(RAIZ / "NodePFN" / "projeto"))
        import avaliar_projeto as ap_
        from fase3_poda import removedor

        def avaliar(alvo, seed, conj):
            hp = json.load(open(ap_.RESULTS / f"{alvo}.json"))["hp"]
            r = ap_.avaliar(alvo, seed, "cuda", hp, preparar_modelo=removedor(set(conj)))
            return r["metrica_artigo_val"], r["metrica_artigo_test"]
        return avaliar
    raise ValueError(modelo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=list(CFG))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--datasets", default=None, help="lista separada por vírgula (padrão: todos do modelo)")
    a = ap.parse_args()
    c = CFG[a.modelo]
    out = R / "transferencia_poda" / a.modelo / "avaliacoes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    feitos = pd.read_csv(out, dtype={"removidas": str}) if out.exists() else pd.DataFrame()
    avaliar = None
    dss = a.datasets.split(",") if a.datasets else c["datasets"]
    for alvo in dss:
        cache = {}
        caminho3 = R / c["fase3"] / alvo / "avaliacoes.csv"
        if caminho3.exists():
            f3df = pd.read_csv(caminho3, dtype={"removidas": str})
            cache = {(r.removidas, int(r.seed)): (r.val, r.test) for r in f3df.itertuples()}
        if a.modelo == "graphpfn":  # avaliações do protocolo do artigo (mesmo pipeline e sementes)
            met = {"tolokers-2": "average_precision", "questions": "average_precision", "artnet-exp": "average_precision",
                   "wiki-cs": "accuracy", "city-roads-M": "r2"}[alvo]
            art = pd.read_csv(R / "graphfpn_poda_cosseno_artigo" / "avaliacoes.csv")
            for r in art[art.dataset == alvo].itertuples():
                conj = [int(x) for x in r.removidas.strip("[]").split(",") if x.strip()]
                if conj:
                    cache.setdefault(("-".join(map(str, sorted(conj))), int(r.seed)),
                                     (getattr(r, f"val_{met}"), getattr(r, f"test_{met}")))
        if (R / c["fase2"] / alvo / "ablacao_individual.csv").exists():
            for conj, d in f3.ablacao_conhecida(c["fase2"], alvo).items():
                for s, vt in d.items():
                    cache.setdefault(("-".join(map(str, conj)), s), vt)
        for origem in [alvo] + [d for d in dss if d != alvo]:
            ords = ordens(c["fase2"], origem, com_aleatorias=origem == alvo)
            for estr in ords:
                for k in f3.KS:
                    conj = tuple(sorted(ords[estr][:k]))
                    chave = "-".join(map(str, conj))
                    for seed in a.seeds:
                        if len(feitos) and ((feitos.alvo == alvo) & (feitos.origem == origem)
                                            & (feitos.estrategia == estr) & (feitos.k == k) & (feitos.seed == seed)).any():
                            continue
                        if (chave, seed) in cache:
                            (val, test), fonte = cache[(chave, seed)], "fase3"
                        else:
                            avaliar = avaliar or avaliador(a.modelo)
                            (val, test), fonte = avaliar(alvo, seed, conj), "novo"
                            cache[(chave, seed)] = (val, test)
                        linha = dict(modelo=c["nome"], alvo=alvo, origem=origem, estrategia=estr, k=k,
                                     removidas=chave, seed=seed, val=val, test=test, fonte=fonte)
                        feitos = pd.concat([feitos, pd.DataFrame([linha])], ignore_index=True)
                        feitos.to_csv(out, index=False)
                        print(linha, flush=True)
