"""
Roda o pipeline inteiro (ou partes dele) com uma configuração só. Edite as variáveis da seção CONFIGURAÇÃO e rode
da raiz do repositório, com qualquer Python 3:

    python3 rodar_pipeline.py

Cada etapa chama os scripts de <Modelo>/projeto/ e de pipeline/ (descritos em docs/pipeline_atual.md) com o Python do
ambiente de cada modelo, um comando por vez, na ordem: preparação dos dados → fase 1 → fase 2 → fase 3 → fase 4 (KD)
→ custo → resumos → transferência da ordem de poda → transferência do KD → relatórios. Os scripts gravam a cada
avaliação e retomam o que já existe, então rodar de novo só completa o que falta.

O que cada fase faz:
  fase 1  modelo inteiro, sem alteração: é a referência (100% do desempenho retido)
  fase 2  importância de cada camada: Block Influence (sem rótulos) e ablação individual (remove uma camada por vez)
  fase 3  poda sem re-treino: remove as k primeiras camadas de cada ordem (BI, ablação, aleatórias), k = 1…6
  fase 4  KD: remove metade das camadas e treina o modelo menor para imitar o inteiro (student podado × aleatório)
Explicação completa em docs/pipeline_atual.md, seção "As fases".

Comece com SIMULAR = True para ver os comandos sem executar nada.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ======================================================================================================================
# CONFIGURAÇÃO
# ======================================================================================================================

# Modelos: "graphpfn", "g2t" (G2T-LimiX), "nodepfn", "ephris"
MODELOS = ["graphpfn", "g2t", "nodepfn", "ephris"]

# Datasets do protocolo comum. Combinações que um modelo não suporta são puladas (NodePFN e Ephris não fazem regressão,
# então não rodam no city-roads-M). O Ephris também aceita artnet-exp, city-reviews, questions, facebook e pubmed.
DATASETS = ["tolokers-2", "wiki-cs", "amazon-ratings", "city-roads-M"]

# Sementes (as fases 1–3 do Ephris usam sempre 0, 1 e 2, definidas em Ephris/projeto/comum.py)
SEMENTES = [0, 1, 2]

# Etapas a rodar (True/False)
ETAPAS = {
    "preparar_dados": True,       # splits do G2T-FM/data/ (usados também pelo NodePFN)
    "fase1": True,                # modelo inteiro
    "fase2": True,                # BI por camada e ablação individual
    "fase3": True,                # poda cumulativa sem re-treino, k = 1…6
    "fase4_kd": True,             # KD, student podado × aleatório (salva os students para a transferência)
    "custo": True,                # parâmetros, latência e pico de GPU (tolokers-2, semente 0)
    "resumos": True,              # tabelas-resumo das fases
    "transferencia_poda": True,   # parte 1 do EXP-013: ordem de BI de outro dataset
    "transferencia_kd": True,     # parte 2 do EXP-013: avalia os students da fase 4 em outros datasets (sem treinar)
    "relatorios": True,           # tabelas do relatório de transferência e compilação dos PDFs
}

# KD (fase 4)
KD_STUDENTS = ["Podado", "Aleatório"]
KD_PASSOS = 200                   # máximo de passos
KD_PACIENCIA = 4                  # avaliações sem melhora na validação até parar (0 = sem parada antecipada)
KD_LR = None                      # None = varredura em {1e-4, 3e-4, 1e-3}; ou um valor fixo, ex.: 1e-4

# Transferência entre datasets (partes 1 e 2). Por padrão, os datasets informativos (o amazon-ratings fica perto do
# acaso em todos os modelos e foi deixado de fora no EXP-013).
TRANSF_DATASETS = ["tolokers-2", "wiki-cs", "city-roads-M"]
# A parte 2 não treina de novo: avalia os students salvos pela fase 4 (pasta <modelo>_fase4_kd) ou, se lá não houver,
# os da rodada do EXP-013 (pasta <modelo>_kd_transfer), feita quando a fase 4 ainda não salvava os students.

# Execução
SIMULAR = True                    # True = só mostra os comandos, não executa nada
SOBRESCREVER = False              # False = pula etapas que não retomam e cujo resultado já existe (ver PRONTO abaixo)
PARAR_NA_FALHA = False            # True = interrompe na primeira falha; False = registra e segue
PASTA_LOGS = "logs_pipeline"      # um log por comando (fora do git)

# Python de cada ambiente
PYTHON = {
    "graphpfn": "~/miniconda3/envs/graphpfn/bin/python",
    "g2t": "~/venvs/g2tfm/bin/python",
    "nodepfn": "~/venvs/nodepfn/bin/python",
    "ephris": "~/miniconda3/envs/ephris/bin/python",
    "resumos": "~/miniconda3/envs/ephris/bin/python",   # precisa de pandas e scipy
}
TECTONIC = "~/miniconda3/envs/latex/bin/tectonic"

# ======================================================================================================================
# Daqui para baixo não precisa mudar.
# ======================================================================================================================

RAIZ = Path(__file__).resolve().parent
R = RAIZ / "Block_Influence_results"

SUPORTE = {  # datasets que cada modelo carrega
    "graphpfn": {"tolokers-2", "wiki-cs", "amazon-ratings", "city-roads-M"},
    "g2t": {"tolokers-2", "wiki-cs", "amazon-ratings", "city-roads-M"},
    "nodepfn": {"tolokers-2", "wiki-cs", "amazon-ratings"},
    "ephris": {"tolokers-2", "wiki-cs", "amazon-ratings", "artnet-exp", "city-reviews", "questions", "facebook",
               "pubmed"},
}
NAO_CABE_KD = {("graphpfn", "wiki-cs")}  # KD não cabe em 16 GB de GPU
CAMADAS_KD = {"graphpfn": 6, "g2t": 6, "nodepfn": 6, "ephris": 5}  # metade das camadas
PASTA_KD = {"graphpfn": "graphfpn_kd_v2", "g2t": "g2t_fase4_kd", "nodepfn": "nodepfn_fase4_kd",
            "ephris": "ephris_fase4_kd"}
PASTA_KD_TRANSF = {"graphpfn": "graphfpn_kd_transfer", "g2t": "g2t_kd_transfer", "nodepfn": "nodepfn_kd_transfer",
                   "ephris": "ephris_kd_transfer"}
SCRIPT_KD = {"graphpfn": "GraphFPN/projeto/kd_v2_graphpfn.py", "g2t": "G2T-FM/projeto/fase4_kd.py",
             "nodepfn": "NodePFN/projeto/fase4_kd.py", "ephris": "Ephris/projeto/fase4_kd.py"}


def py(chave):
    return str(Path(PYTHON[chave]).expanduser())


def datasets_de(modelo, lista=None):
    return [d for d in (DATASETS if lista is None else lista) if d in SUPORTE[modelo]]


def opcoes_memoria_g2t(ds):
    """Ajustes do G2T-LimiX para caber em 16 GB (protocolo do relatório)."""
    if ds == "amazon-ratings":
        return ["--eval-batch", "2048", "--max-ctx", "900", "--train-chunks", "2"]
    if ds == "wiki-cs":
        return ["--eval-batch", "2048"]
    return []


def pasta_students(modelo):
    """Pasta com os student_*.pt do modelo: a da fase 4, se tiver students; senão, a da rodada do EXP-013."""
    for pasta in (PASTA_KD[modelo], PASTA_KD_TRANSF[modelo]):
        if any((R / pasta).glob("student_*.pt")):
            return pasta
    return PASTA_KD[modelo]  # serão criados pela fase 4


def cmd_kd(modelo, ds, seed, kinds, lr, out=None, salvar=False):
    c = [py(modelo), SCRIPT_KD[modelo], "--dataset", ds, "--seed", str(seed), "--prune-k", str(CAMADAS_KD[modelo]),
         "--steps", str(KD_PASSOS), "--patience", str(KD_PACIENCIA), "--students", ",".join(kinds)]
    if lr is not None:
        c += ["--lr", f"{lr:g}"]
    if out:
        c += ["--out", out]
    if salvar:
        c += ["--salvar"]
    if modelo == "g2t":
        c += opcoes_memoria_g2t(ds)
    return c


def montar_comandos():
    """Lista de (nome, comando, pasta de trabalho)."""
    s = [str(x) for x in SEMENTES]
    cmds = []

    def add(nome, c, cwd=RAIZ, pronto=None):
        """pronto: arquivo ou pasta que indica que a etapa já foi feita (para etapas que não retomam sozinhas)."""
        if pronto is not None and not SOBRESCREVER and Path(pronto).exists():
            print(f"  já feito, pulado: {nome} ({Path(pronto).relative_to(RAIZ)})")
            return
        cmds.append((nome, c, cwd))

    if ETAPAS["preparar_dados"] and ({"g2t", "nodepfn"} & set(MODELOS)):
        add("preparar_dados", [py("g2t"), "G2T-FM/projeto/preparar_dados_projeto.py"])

    if ETAPAS["fase1"]:
        for m in MODELOS:
            if m == "graphpfn":
                print("  aviso: a fase 1 do GraphPFN vem do protocolo do artigo "
                      "(GraphFPN/poda_cosseno_artigo_graphpfn.ipynb); não é rodada aqui.")
            elif m == "g2t":
                for d in datasets_de(m):
                    cfg = f"exp/projeto/fase1/limix-icl/{d}/evaluation.toml"
                    add(f"fase1_g2t_{d}", [py(m), "bin/go.py", cfg, "--force"], RAIZ / "G2T-FM",
                        pronto=RAIZ / "G2T-FM" / cfg.replace(".toml", ""))
            elif m == "nodepfn":
                for d in datasets_de(m):
                    # --grade: suavização e TSVD escolhidos pela validação, como no protocolo
                    add(f"fase1_nodepfn_{d}", [py(m), "NodePFN/projeto/avaliar_projeto.py", "--dataset", d, "--grade",
                                               "--seeds", *s], pronto=R / "nodepfn_fase1" / f"{d}.json")
            elif m == "ephris" and datasets_de(m):
                add("fase1_ephris", [py(m), "Ephris/projeto/fase1.py", "--datasets", ",".join(datasets_de(m))])

    for fase in ("fase2", "fase3"):
        if not ETAPAS[fase]:
            continue
        n = fase[-1]
        for m in MODELOS:
            for d in datasets_de(m):
                if m == "graphpfn":
                    add(f"{fase}_graphpfn_{d}", [py(m), "GraphFPN/projeto/fase23_graphpfn.py", "--dataset", d,
                                                 "--fase", n, "--seeds", *s])
                elif m == "g2t" and fase == "fase2":
                    add(f"fase2_g2t_bi_{d}", [py(m), "G2T-FM/projeto/fase2_bi_ablacao.py", "--dataset", d,
                                              "--modo", "bi", "--seeds", *s],
                        pronto=R / "g2t_fase2" / d / "block_influence.csv")
                    add(f"fase2_g2t_ablacao_{d}", [py(m), "G2T-FM/projeto/fase2_bi_ablacao.py", "--dataset", d,
                                                   "--modo", "ablacao", "--seeds", *s])
                elif m == "g2t":
                    add(f"fase3_g2t_{d}", [py(m), "G2T-FM/projeto/fase3_poda.py", "--dataset", d, "--seeds", *s])
                elif m == "nodepfn":
                    if fase == "fase2":
                        add(f"fase2_nodepfn_{d}", [py(m), "NodePFN/projeto/fase2_bi_ablacao.py", "--dataset", d,
                                                   "--seeds", *s], pronto=R / "nodepfn_fase2" / d / "ablacao_individual.csv")
                    else:
                        add(f"fase3_nodepfn_{d}", [py(m), "NodePFN/projeto/fase3_poda.py", "--dataset", d, "--seeds", *s])
                elif m == "ephris" and (fase == "fase2" or not ETAPAS["fase2"]):
                    # o fase23.py do Ephris faz as fases 2 e 3 juntas
                    add(f"fase23_ephris_{d}", [py(m), "Ephris/projeto/fase23.py", "--dataset", d],
                        pronto=R / "ephris_fase3" / d / "poda.csv")

    if ETAPAS["fase4_kd"]:
        for m in MODELOS:
            for d in datasets_de(m):
                if (m, d) in NAO_CABE_KD:
                    print(f"  aviso: KD do {m} no {d} pulado (não cabe em 16 GB).")
                    continue
                for seed in SEMENTES:
                    add(f"kd_{m}_{d}_s{seed}", cmd_kd(m, d, seed, KD_STUDENTS, KD_LR, salvar=True))

    if ETAPAS["custo"]:
        scripts = {"graphpfn": "GraphFPN/projeto/custo_graphpfn.py", "g2t": "G2T-FM/projeto/custo_g2t.py",
                   "nodepfn": "NodePFN/projeto/custo_nodepfn.py", "ephris": "Ephris/projeto/custo_ephris.py"}
        arquivos = {"graphpfn": "graphpfn.csv", "g2t": "g2t.csv", "nodepfn": "nodepfn.csv", "ephris": "ephris.csv"}
        for m in MODELOS:
            add(f"custo_{m}", [py(m), scripts[m]], pronto=R / "custo_pfns" / arquivos[m])

    if ETAPAS["resumos"]:
        for r in ("resumo_fase2", "resumo_fase3", "resumo_fase4", "resumo_kd"):
            add(r, [py("resumos"), f"pipeline/{r}.py"])
        if "ephris" in MODELOS:
            add("resumo_ephris", [py("ephris"), "Ephris/projeto/resumo.py"])

    if ETAPAS["transferencia_poda"]:
        for m in MODELOS:
            dss = datasets_de(m, TRANSF_DATASETS)
            if len(dss) < 2:
                continue
            if m == "ephris":
                add("transf_poda_ephris", [py(m), "Ephris/projeto/transferencia_poda.py", "--datasets", ",".join(dss)])
            else:
                add(f"transf_poda_{m}", [py(m), "pipeline/transferencia_poda.py", "--modelo", m, "--seeds", *s,
                                         "--datasets", ",".join(dss)])

    if ETAPAS["transferencia_kd"]:
        for m in MODELOS:
            dss = datasets_de(m, TRANSF_DATASETS)
            if len(dss) < 2:
                continue
            pasta = pasta_students(m)
            if m == "ephris":
                add("transf_kd_ephris", [py(m), "Ephris/projeto/kd_transferencia.py", "--pasta", pasta,
                                         "--alvos", ",".join(dss)])
            else:
                add(f"transf_kd_{m}", [py(m), "pipeline/kd_transferencia.py", "--modelo", m, "--pasta", pasta,
                                       "--alvos", ",".join(dss)])

    if ETAPAS["relatorios"]:
        add("relatorio_tabelas", [py("resumos"), "Relatorio/transferencia/gerar_tabelas.py"])
        tec = str(Path(TECTONIC).expanduser())
        add("relatorio_principal", [tec, "-X", "compile", "relatorio.tex"], RAIZ / "Relatorio")
        add("relatorio_transferencia", [tec, "-X", "compile", "relatorio_transferencia.tex"], RAIZ / "Relatorio")
    return cmds


def main():
    print(f"Pipeline: modelos {MODELOS}, datasets {DATASETS}, sementes {SEMENTES}")
    print(f"Etapas: {[e for e, ligada in ETAPAS.items() if ligada]}")
    cmds = montar_comandos()
    print(f"{len(cmds)} comandos{' (simulação: nada será executado)' if SIMULAR else ''}\n")
    if SIMULAR:
        for nome, c, cwd in cmds:
            onde = "" if cwd == RAIZ else f"(cd {cwd.relative_to(RAIZ)}) "
            print(f"[{nome}] {onde}{' '.join(c)}")
        return 0
    logs = (RAIZ / PASTA_LOGS / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    falhas = []
    for i, (nome, c, cwd) in enumerate(cmds, 1):
        tic = time.time()
        print(f"[{i}/{len(cmds)}] {nome} ... ", end="", flush=True)
        with open(logs / f"{i:03d}_{nome}.log", "w") as log:
            rc = subprocess.run(c, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{'ok' if rc == 0 else f'FALHOU (código {rc})'} em {(time.time() - tic) / 60:.1f} min", flush=True)
        if rc != 0:
            falhas.append(nome)
            if PARAR_NA_FALHA:
                break
    print(f"\nLogs em {logs.relative_to(RAIZ) if logs.is_relative_to(RAIZ) else logs}.")
    if falhas:
        print(f"Falhas ({len(falhas)}): {', '.join(falhas)}. Rodar de novo completa o que faltou.")
        return 1
    print("Tudo concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
