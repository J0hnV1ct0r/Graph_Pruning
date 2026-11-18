# Pipeline atual

Protocolo comum dos PFNs (GraphPFN, G2T-LimiX, NodePFN e Ephris) e transferência entre datasets. É o que alimenta
`Relatorio/relatorio.pdf` e `Relatorio/relatorio_transferencia.pdf`. Onde está cada pasta e cada ambiente:
[`../DOCUMENTACAO.md`](../DOCUMENTACAO.md). A pipeline anterior está em [`pipeline_antiga.md`](pipeline_antiga.md).

_Atualizado em 18/11/2026._

---

## 1. Protocolo comum dos PFNs

Usado para que GraphPFN, G2T-LimiX, NodePFN e Ephris sejam comparáveis entre si (o relatório principal só traz
análises neste protocolo).

- **Datasets:** tolokers-2, wiki-cs e amazon-ratings nos quatro; city-roads-M no GraphPFN e no G2T-LimiX (NodePFN e
  Ephris não fazem regressão). amazon-ratings fica perto do acaso em todos e não sustenta conclusões.
- **Splits:** os do artigo do GraphPFN: split RL do GraphLand; 10/10/80 com semente 0 nos clássicos (o mesmo
  `split.npz` em todos os modelos).
- **Sementes:** 0, 1, 2 (variam o modelo e o treino, não o split).
- **Métrica:** AP (binária), acurácia (multiclasse), $R^2$ (regressão), reportada como **desempenho retido**:
  `100·(podado − acaso)/(inteiro − acaso)`, pareado por semente.
- **Fases:**
  1. modelo inteiro (reprodução);
  2. BI por camada (cosseno e angular; todos os tokens e token-alvo) e ablação individual;
  3. poda cumulativa sem re-treino, `k = 1…6`, camada 0 protegida, ordens `bi_todos`, `bi_alvo`, `ablacao`
     (escolhida na validação) e 3 aleatórias (`fase3_ordens.py`);
  4. KD com metade das camadas removidas (6 de 12; 5 de 10 no Ephris): student podado × student aleatório,
     perda T²·KL (T = 2) sem rótulos, até 200 passos, parada antecipada com paciência 4, melhor ponto pela validação;
  - custo: parâmetros, latência e pico de GPU com `k = 0, 2, 4, 6` (tolokers-2, semente 0);
  - transferência entre datasets (EXP-013): ver seção 3.
- Inferência em blocos para caber em 16 GB (resultado idêntico ao forward original).

---

## 2. Como rodar cada fase

Comandos reais usados no projeto. Os scripts gravam a cada avaliação e retomam de onde pararam.

### GraphPFN (da raiz, `~/miniconda3/envs/graphpfn/bin/python`)

| Fase | Comando | Saída em `Block_Influence_results/` |
|---|---|---|
| 2 | `GraphFPN/fase23_graphpfn.py --dataset <ds> --fase 2` | `graphfpn_fase2/<ds>/` |
| 3 | `GraphFPN/fase23_graphpfn.py --dataset <ds> --fase 3` | `graphfpn_fase3/<ds>/` |
| 4 (KD) | `GraphFPN/kd_v2_graphpfn.py --dataset <ds> --seed <s>` | `graphfpn_kd_v2/` |
| custo | `GraphFPN/custo_graphpfn.py` | `custo_pfns/graphpfn.csv` |

O modelo inteiro e o BI vêm do protocolo do artigo (`graphfpn_poda_cosseno_artigo/`). A análise mais antiga (splits Planetoid, 1 membro de ensemble) não é comparável com os outros modelos e está
descrita em [`pipeline_antiga.md`](pipeline_antiga.md).

### G2T-LimiX (de dentro de `G2T-FM/`, `~/venvs/g2tfm/bin/python`)

| Fase | Comando | Saída em `../Block_Influence_results/` |
|---|---|---|
| dados | `preparar_dados_projeto.py` | `G2T-FM/data/` (links e `split.npz`) |
| 1 | `bin/go.py exp/projeto/fase1/limix-icl/<ds>/evaluation.toml --force` | pasta do experimento |
| 2 | `fase2_bi_ablacao.py --dataset <ds> --modo bi` e `--modo ablacao` | `g2t_fase2/<ds>/` |
| 3 | `fase3_poda.py --dataset <ds>` | `g2t_fase3/<ds>/` |
| 4 (KD) | `fase4_kd.py --dataset <ds> --seed <s>` (wiki-cs e amazon-ratings com `--eval-batch 2048`; amazon-ratings com `--max-ctx 900 --train-chunks 2`) | `g2t_fase4_kd/` |
| custo | `custo_g2t.py` | `custo_pfns/g2t.csv` |

### NodePFN (de dentro de `NodePFN/`, `~/venvs/nodepfn/bin/python`)

| Fase | Comando | Saída em `../Block_Influence_results/` |
|---|---|---|
| 1 | `avaliar_projeto.py --dataset <ds>` | `nodepfn_fase1/<ds>.json` (com os hiperparâmetros) |
| 2 | `fase2_bi_ablacao.py --dataset <ds>` | `nodepfn_fase2/<ds>/` |
| 3 | `fase3_poda.py --dataset <ds>` | `nodepfn_fase3/<ds>/` |
| 4 (KD) | `fase4_kd.py --dataset <ds> --seed <s>` | `nodepfn_fase4_kd/` |
| custo | `custo_nodepfn.py` | `custo_pfns/nodepfn.csv` |

### Ephris (de dentro de `Ephris/`, `~/miniconda3/envs/ephris/bin/python`)

| Fase | Comando | Saída em `Block_Influence_results/` |
|---|---|---|
| 1 | `projeto/fase1.py` | `ephris_fase1/modelo_inteiro.csv` |
| 2 e 3 | `projeto/fase23.py --dataset <ds>` | `ephris_fase2/<ds>/`, `ephris_fase3/<ds>/` |
| 4 (KD) | `projeto/fase4_kd.py --dataset <ds> --seed <s>` | `ephris_fase4_kd/` |
| custo | `projeto/custo_ephris.py` | `custo_pfns/ephris.csv` |
| resumo | `projeto/resumo.py` | `ephris_fase2/resumo_fase2.csv`, `ephris_fase3/resumo_fase3.csv`, `ephris_fase4_kd/resumo_kd.csv` |

O Ephris roda nos 8 datasets de classificação do protocolo que cabem em 16 GB (tolokers-2, artnet-exp,
city-reviews, amazon-ratings, wiki-cs, questions, facebook, pubmed); na comparação com os outros PFNs entram só
tolokers-2, wiki-cs e amazon-ratings. A atenção de grafo do Ephris é só de inferência; `fase4_kd.py` tem uma versão
diferenciável (diferença máxima de 1,2e-6).

### Resumos (da raiz)

`resumo_fase2.py`, `resumo_fase3.py`, `resumo_fase4.py` e `resumo_kd.py` (KD dos quatro modelos com a mesma
normalização; grava `Block_Influence_results/resumo_kd_todos.csv`).

`rodar_com_vigia.py -- <comando>` roda um comando e o encerra se a RAM ou a GPU ficarem ocupadas por outros processos.

---

## 3. Transferência entre datasets (EXP-013)

Pergunta: medir o BI e fazer o KD no mesmo dataset em que o resultado é avaliado infla os resultados?

- **Parte 1, ordem de poda transferida:** remove no dataset alvo as camadas de menor BI medidas em outro dataset.
  - GraphPFN, G2T-LimiX, NodePFN: `transferencia_poda.py --modelo {graphpfn,g2t,nodepfn}` (da raiz,
    no ambiente de cada modelo) → `transferencia_poda/<modelo>/avaliacoes.csv`
  - Ephris: `projeto/transferencia_poda.py` (de `Ephris/`) → `transferencia_poda/ephris/avaliacoes.csv`
- **Parte 2, student do KD transferido:** refaz o KD com `--salvar` (guarda o melhor student), `--out <modelo>_kd_transfer`
  e o lr do protocolo principal (`--lr`), e avalia cada student, sem treino, nos outros datasets.
  - GraphPFN, G2T-LimiX, NodePFN: `kd_transferencia.py --modelo <m>` (da raiz)
  - Ephris: `projeto/kd_transferencia.py` (de `Ephris/`)
  - Saída: `<modelo>_kd_transfer/{student_*.pt, transferencia.csv}` (`graphfpn_`, `g2t_`, `nodepfn_`, `ephris_`).
    Os `student_*.pt` somam alguns GB e ficam fora do git (`.gitignore`).
- **Em andamento (18/11):** a fila da parte 2 roda na cópia de trabalho `.claude/worktrees/fase1-novos-gfms/`; o KD do
  student aleatório do GraphPFN no city-roads-M foi adiado para o fim dela.

---

## 4. Relatórios

| Arquivo | Conteúdo |
|---|---|
| `Relatorio/relatorio.tex` / `.pdf` | relatório principal: só as análises comparáveis (os quatro PFNs no protocolo comum) |
| `Relatorio/relatorio_transferencia.tex` / `.pdf` | relatório complementar da transferência entre datasets (EXP-013) |
| `Relatorio/transferencia/gerar_tabelas.py` | gera as tabelas do relatório de transferência a partir dos CSVs |

Compilar (de `Relatorio/`):

```bash
~/miniconda3/envs/ephris/bin/python transferencia/gerar_tabelas.py   # só para o relatório de transferência
~/miniconda3/envs/latex/bin/tectonic -X compile relatorio.tex
~/miniconda3/envs/latex/bin/tectonic -X compile relatorio_transferencia.tex
```

---

## 5. Estado e próximos passos

- **Feito:** fases 1–4 e custo nos quatro PFNs; relatório principal; parte 1 da transferência nos quatro modelos.
- **Em andamento (18/11):** parte 2 da transferência (KDs do GraphPFN no city-roads-M e avaliações cruzadas finais).
- **Próximos passos:**
  - datasets mais leves que o city-roads-M para a transferência: minesweeper, amazon-photo e amazon-computer
    (e cora e citeseer), que os quatro modelos carregam e para os quais o NodePFN tem hiperparâmetros oficiais;
  - KD com orçamento igual de tempo de relógio para os students podado e aleatório, rodando sozinho na GPU;
  - quando a fila terminar, copiar os resultados finais para o `main` e remover a cópia de trabalho separada.
