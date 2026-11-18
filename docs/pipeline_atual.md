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
- Inferência em blocos para caber em 16 GB (resultado idêntico ao forward original).

### As fases

Cada fase usa o resultado da anterior. A pergunta geral é: dá para remover camadas inteiras de um PFN para grafos,
escolhidas pelo Block Influence (BI), sem perder desempenho? E, se perder, o KD recupera?

**Fase 1: modelo inteiro (referência).** Roda o modelo pré-treinado, sem nenhuma alteração, em cada dataset e
semente, e confere se o resultado bate com o publicado no artigo. Para que serve: é o 100% do desempenho retido e
garante que o pipeline reproduz o modelo corretamente antes de mexer nele.
Saída: métrica do modelo inteiro por semente (`<modelo>_fase1/` ou `baseline.csv` da fase 2).

**Fase 2: quanto cada camada importa.** Duas medidas, camada por camada:
- **Block Influence (BI):** o quanto a camada muda a representação, $BI = 1 - \cos(h_{entrada}, h_{saída})$, sem usar
  rótulos. Camada com BI baixo quase não muda nada e seria candidata a sair. É medido de duas formas: na média de
  todos os tokens e só no token que o modelo usa para predizer (token-alvo), e com cosseno e com distância angular.
- **Ablação individual:** remove cada camada sozinha e mede o desempenho que sobra. É a importância real da camada,
  mas exige rótulos (usa a validação).
Para que serve: ver se o BI, que é barato e não usa rótulos, aponta as mesmas camadas que a ablação (correlação de
Spearman entre os dois). Saída: `<modelo>_fase2/<ds>/` (`baseline.csv`, `block_influence.csv`,
`ablacao_individual.csv`).

**Fase 3: poda sem re-treino.** Remove as `k` primeiras camadas de uma ordem, para `k = 1…6`, e mede o desempenho
retido, sem treinar nada. A camada 0 nunca sai (removê-la derruba o modelo ao acaso). Ordens comparadas
(`pipeline/fase3_ordens.py`): BI de todos os tokens, BI do token-alvo (menor BI primeiro), ablação (a referência,
escolhida na validação) e 3 ordens aleatórias (o controle). Para que serve: saber quantas camadas dá para tirar sem
perder muito e se o BI escolhe melhor que o acaso. Saída: `<modelo>_fase3/<ds>/` (`plano.csv` com as camadas
removidas em cada caso, `avaliacoes.csv` com o resultado).

**Fase 4: recuperação por destilação (KD).** Remove metade das camadas (6 de 12; 5 de 10 no Ephris), o que
derruba o desempenho, e treina esse modelo menor (*student*) para imitar o modelo inteiro (*professor*). A perda é a
divergência entre as saídas dos dois (T²·KL, temperatura 2), sem usar rótulos; até 200 passos, com parada antecipada
quando a validação para de melhorar (paciência 4) e o melhor ponto escolhido pela validação. Dois students:
- **podado:** começa do modelo com as camadas de menor BI removidas (pesos pré-treinados);
- **aleatório:** mesma arquitetura, pesos aleatórios (no Ephris, só a pilha de camadas podadas).
Para que serve: saber se o modelo podado recupera o desempenho com um treino curto e se começar do podado é melhor
que começar do zero. Saída: `<modelo>_fase4_kd/` (`kd_log_*.csv` com a curva, `kd_resumo_*.json` com o resultado).

**Custo.** Mede parâmetros, latência e pico de memória da GPU removendo 0, 2, 4 e 6 camadas (tolokers-2,
semente 0). Para que serve: saber quanto a poda economiza. Saída: `custo_pfns/<modelo>.csv`.

**Transferência entre datasets (EXP-013).** Repete a fase 3 com a ordem de BI de outro dataset e avalia o student da
fase 4 em datasets diferentes do que foi usado no KD (seção 3). Para que serve: saber se a poda e o KD valem para o
modelo em geral ou só para o dataset em que foram feitos.

---

## 2. Como rodar cada fase

**Tudo de uma vez:** `python3 rodar_pipeline.py` (raiz). Ele monta e executa, na ordem, os comandos desta seção para os
modelos, datasets, sementes e etapas escolhidos nas variáveis do topo do arquivo (sem argumentos de linha de comando),
pulando combinações que não se aplicam e etapas já feitas. Com `SIMULAR = True` só mostra os comandos; logs por
comando em `logs_pipeline/`.

**Um comando por vez:** tudo roda **da raiz do repositório**, no ambiente de cada modelo, com `python <Modelo>/projeto/<script>.py` ou
`python pipeline/<script>.py`. Os scripts gravam a cada avaliação e retomam de onde pararam. A organização das
pastas e as convenções de nomes estão em [`padroes.md`](padroes.md).

### GraphPFN (`~/miniconda3/envs/graphpfn/bin/python`)

| Fase | Comando | Saída em `Block_Influence_results/` |
|---|---|---|
| 2 | `GraphFPN/projeto/fase23_graphpfn.py --dataset <ds> --fase 2` | `graphfpn_fase2/<ds>/` |
| 3 | `GraphFPN/projeto/fase23_graphpfn.py --dataset <ds> --fase 3` | `graphfpn_fase3/<ds>/` |
| 4 (KD) | `GraphFPN/projeto/kd_v2_graphpfn.py --dataset <ds> --seed <s>` | `graphfpn_kd_v2/` |
| custo | `GraphFPN/projeto/custo_graphpfn.py` | `custo_pfns/graphpfn.csv` |

O modelo inteiro e o BI vêm do protocolo do artigo (`graphfpn_poda_cosseno_artigo/`). A análise mais antiga (splits
Planetoid, 1 membro de ensemble) não é comparável com os outros modelos e está descrita em
[`pipeline_antiga.md`](pipeline_antiga.md).

### G2T-LimiX (`~/venvs/g2tfm/bin/python`)

| Fase | Comando | Saída em `Block_Influence_results/` |
|---|---|---|
| dados | `G2T-FM/projeto/preparar_dados_projeto.py` | `G2T-FM/data/` (links para o cache e `split.npz`; fora do git) |
| 1 | `bin/go.py exp/projeto/fase1/limix-icl/<ds>/evaluation.toml --force` (este, de dentro de `G2T-FM/`) | pasta do experimento |
| 2 | `G2T-FM/projeto/fase2_bi_ablacao.py --dataset <ds> --modo bi` e `--modo ablacao` | `g2t_fase2/<ds>/` |
| 3 | `G2T-FM/projeto/fase3_poda.py --dataset <ds>` | `g2t_fase3/<ds>/` |
| 4 (KD) | `G2T-FM/projeto/fase4_kd.py --dataset <ds> --seed <s>` (wiki-cs e amazon-ratings com `--eval-batch 2048`; amazon-ratings com `--max-ctx 900 --train-chunks 2`) | `g2t_fase4_kd/` |
| custo | `G2T-FM/projeto/custo_g2t.py` | `custo_pfns/g2t.csv` |

As configurações das fases ficam em `G2T-FM/exp/projeto/fase1/limix-icl/<ds>/evaluation.toml` (as oficiais com 3
sementes).

### NodePFN (`~/venvs/nodepfn/bin/python`)

| Fase | Comando | Saída em `Block_Influence_results/` |
|---|---|---|
| 1 | `NodePFN/projeto/avaliar_projeto.py --dataset <ds>` | `nodepfn_fase1/<ds>.json` (com os hiperparâmetros) |
| 2 | `NodePFN/projeto/fase2_bi_ablacao.py --dataset <ds>` | `nodepfn_fase2/<ds>/` |
| 3 | `NodePFN/projeto/fase3_poda.py --dataset <ds>` | `nodepfn_fase3/<ds>/` |
| 4 (KD) | `NodePFN/projeto/fase4_kd.py --dataset <ds> --seed <s>` | `nodepfn_fase4_kd/` |
| custo | `NodePFN/projeto/custo_nodepfn.py` | `custo_pfns/nodepfn.csv` |

O NodePFN usa os splits de `G2T-FM/data/` (rodar antes `G2T-FM/projeto/preparar_dados_projeto.py`).

### Ephris (`~/miniconda3/envs/ephris/bin/python`)

| Fase | Comando | Saída em `Block_Influence_results/` |
|---|---|---|
| 1 | `Ephris/projeto/fase1.py` | `ephris_fase1/modelo_inteiro.csv` |
| 2 e 3 | `Ephris/projeto/fase23.py --dataset <ds>` | `ephris_fase2/<ds>/`, `ephris_fase3/<ds>/` |
| 4 (KD) | `Ephris/projeto/fase4_kd.py --dataset <ds> --seed <s>` | `ephris_fase4_kd/` |
| custo | `Ephris/projeto/custo_ephris.py` | `custo_pfns/ephris.csv` |
| resumo | `Ephris/projeto/resumo.py` | `ephris_fase2/resumo_fase2.csv`, `ephris_fase3/resumo_fase3.csv`, `ephris_fase4_kd/resumo_kd.csv` |

O Ephris roda nos 8 datasets de classificação do protocolo que cabem em 16 GB (tolokers-2, artnet-exp,
city-reviews, amazon-ratings, wiki-cs, questions, facebook, pubmed); na comparação com os outros PFNs entram só
tolokers-2, wiki-cs e amazon-ratings. A atenção de grafo do Ephris é só de inferência; `fase4_kd.py` tem uma versão
diferenciável (diferença máxima de 1,2e-6). Os dados preparados ficam em `Ephris/projeto/cache/` (fora do git).

### Scripts comuns (`pipeline/`)

| Script | O que faz |
|---|---|
| `pipeline/fase3_ordens.py` | ordens de poda da fase 3 (BI, ablação, aleatórias) e valores de acaso por dataset; usado pelos scripts de todos os modelos |
| `pipeline/resumo_fase2.py`, `resumo_fase3.py`, `resumo_fase4.py` | resumos das fases 2–4 do G2T-LimiX e do NodePFN |
| `pipeline/resumo_kd.py` | KD dos modelos com a mesma normalização; grava `Block_Influence_results/resumo_kd_todos.csv` |
| `pipeline/rodar_com_vigia.py -- <comando>` | roda um comando e o encerra se a RAM ou a GPU ficarem ocupadas por outros processos |

---

## 3. Transferência entre datasets (EXP-013)

Pergunta: medir o BI e fazer o KD no mesmo dataset em que o resultado é avaliado infla os resultados?

- **Parte 1, ordem de poda transferida:** remove no dataset alvo as camadas de menor BI medidas em outro dataset.
  - GraphPFN, G2T-LimiX, NodePFN: `pipeline/transferencia_poda.py --modelo {graphpfn,g2t,nodepfn}` (no ambiente
    de cada modelo) → `transferencia_poda/<modelo>/avaliacoes.csv`
  - Ephris: `Ephris/projeto/transferencia_poda.py` → `transferencia_poda/ephris/avaliacoes.csv`
- **Parte 2, student do KD transferido:** avalia cada student da fase 4, sem novo treino, nos outros datasets. A fase 4
  salva os students com `--salvar` (`rodar_pipeline.py` já faz isso), então o KD não é treinado duas vezes. Na rodada
  do EXP-013 (18/11), a fase 4 tinha sido feita antes de existir o `--salvar`; por isso o KD foi refeito uma vez, com
  o mesmo protocolo e o mesmo lr (`--out <modelo>_kd_transfer --lr <lr do protocolo>`), e reproduziu os resultados
  originais. Os students dessa rodada ficaram na cópia de trabalho separada (os `.pt` não vão para o git); para
  reavaliar, copie-os para `Block_Influence_results/<modelo>_kd_transfer/`.
  - GraphPFN, G2T-LimiX, NodePFN: `pipeline/kd_transferencia.py --modelo <m> --pasta <pasta com os students>`
  - Ephris: `Ephris/projeto/kd_transferencia.py --pasta <pasta com os students>`
  - Saída: `transferencia.csv` na mesma pasta dos students (`<modelo>_fase4_kd/` nas rodadas novas;
    `<modelo>_kd_transfer/` na rodada do EXP-013). Os `student_*.pt` somam alguns GB e ficam fora do git.
- **Em andamento (18/11):** a fila da parte 2 roda na cópia de trabalho `.claude/worktrees/fase1-novos-gfms/`, que
  ainda usa a organização antiga das pastas; o KD do student aleatório do GraphPFN no city-roads-M foi adiado para o
  fim dela. Como trazer os resultados para o `main`: [`padroes.md`](padroes.md), seção 5.

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
  - quando a fila terminar, trazer os resultados finais para o `main` ([`padroes.md`](padroes.md), seção 5) e
    remover a cópia de trabalho separada.
