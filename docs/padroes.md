# Padrões do repositório

Organização das pastas, convenções de nomes e como acrescentar código, datasets ou resultados mantendo o mesmo
padrão. Vale para tudo o que entrar no `main`, inclusive o que vier da cópia de trabalho separada
(`.claude/worktrees/fase1-novos-gfms/`, organização antiga; seção 5).

_Atualizado em 18/11/2026._

---

## 1. Estrutura

```
Graph_Pruning/
├── rodar_pipeline.py          roda o pipeline inteiro; configuração nas variáveis do topo
├── pipeline/                  scripts comuns a vários modelos
│   ├── fase3_ordens.py        ordens de poda (BI, ablação, aleatórias) e acaso por dataset
│   ├── transferencia_poda.py  transferência da ordem de poda (parte 1 do EXP-013)
│   ├── kd_transferencia.py    transferência do student do KD (parte 2 do EXP-013)
│   ├── resumo_*.py            resumos que juntam vários modelos
│   └── rodar_com_vigia.py
├── <Modelo>/                  código oficial do modelo, sem alterações de lógica
│   └── projeto/               scripts do projeto para esse modelo
├── Block_Influence_results/   resultados (CSV/JSON), uma pasta por modelo e etapa
├── Relatorio/                 relatórios LaTeX e o gerador de tabelas
├── docs/                      esta documentação
└── DOCUMENTACAO.md            índice
```

Modelos com `projeto/`: `AnyGraph/`, `GraphFPN/` (GraphPFN), `G2T-FM/`, `NodePFN/`, `Ephris/`. Os notebooks antigos
do GraphPFN continuam em `GraphFPN/` (pipeline antiga).

**Regra:** o código oficial de cada modelo fica onde veio; tudo o que é do projeto vai para `<Modelo>/projeto/`; o que
serve a mais de um modelo vai para `pipeline/`. Na raiz, só o `rodar_pipeline.py` (ponto de entrada único).

## 2. Scripts

### Nomes

| Etapa | Nome do arquivo em `<Modelo>/projeto/` |
|---|---|
| preparação de dados | `preparar_dados_projeto.py` |
| fase 1 (modelo inteiro) | `fase1.py` (no NodePFN, `avaliar_projeto.py`, que também é o pipeline de avaliação importado pelos outros) |
| fase 2 (BI e ablação individual) | `fase2_bi_ablacao.py` (fases 2 e 3 juntas: `fase23.py` / `fase23_<modelo>.py`) |
| fase 3 (poda cumulativa) | `fase3_poda.py` |
| fase 4 (KD) | `fase4_kd.py` (GraphPFN: `kd_v2_graphpfn.py`) |
| custo | `custo_<modelo>.py` |
| transferência (só se o modelo não couber em `pipeline/`) | `transferencia_poda.py`, `kd_transferencia.py` |
| resumo de um modelo | `resumo.py` ou `resumo_<modelo>.py` |

### Cabeçalho de caminhos

Todo script de `<Modelo>/projeto/` calcula os caminhos a partir do próprio arquivo, nunca do diretório atual, para
rodar de qualquer lugar:

```python
AQUI = Path(__file__).resolve().parents[1]   # <Modelo>/ (código oficial, checkpoints, configs)
RAIZ = AQUI.parent                            # raiz do repositório
sys.path.insert(0, str(RAIZ / "pipeline"))    # fase3_ordens e outros scripts comuns
R = RAIZ / "Block_Influence_results"
```

Se o código oficial do modelo espera um diretório de trabalho (configs com caminhos relativos, checkpoints procurados
em `./checkpoints`), o script faz `os.chdir(AQUI)` logo no início, como em `G2T-FM/projeto/fase2_bi_ablacao.py` e
`GraphFPN/projeto/fase23_graphpfn.py`. Scripts de `pipeline/` usam `RAIZ = Path(__file__).resolve().parents[1]` e
importam os de um modelo inserindo `RAIZ / "<Modelo>" / "projeto"` no `sys.path`.

### Execução

- Sempre da raiz: `python <Modelo>/projeto/<script>.py ...` ou `python pipeline/<script>.py ...`, no ambiente do
  modelo (tabela em [`../DOCUMENTACAO.md`](../DOCUMENTACAO.md)).
- Docstring no topo de cada script: o que faz, protocolo, linha `Uso (da raiz do repositório, <ambiente>): ...` e
  `Saída: Block_Influence_results/...`.
- Gravar resultados a cada avaliação e retomar do que já existe (os scripts atuais fazem isso).
- Nada de caminhos absolutos da máquina nem de pastas temporárias fixas (`tempfile.mkdtemp()` sem `dir=`).

## 3. Resultados (`Block_Influence_results/`)

- Uma pasta por modelo e etapa: `<modelo>_<etapa>/`, com `<modelo>` em minúsculas (`graphfpn`, `g2t`, `nodepfn`,
  `ephris`, `anygraph`) e `<etapa>` = `fase1`, `fase2`, `fase3`, `fase4_kd`, `kd_transfer`. Dentro, uma subpasta por
  dataset quando há vários arquivos por dataset (`g2t_fase2/<ds>/baseline.csv`, `block_influence.csv`,
  `ablacao_individual.csv`; `<modelo>_fase3/<ds>/plano.csv`, `avaliacoes.csv`).
- Pastas que juntam modelos: `custo_pfns/<modelo>.csv`, `transferencia_poda/<modelo>/avaliacoes.csv`,
  `resumo_kd_todos.csv`.
- KD: `kd_log_<ds>_k<k>_s<semente>_<Podado|Aleatório>_<passos>p<paciência>[_lr<lr>][_ctx<n>].csv` (curva) e
  `kd_resumo_<mesmo sufixo>.json` (resumo). `200p4` = protocolo principal; `200p0`/`1000p0` = sem parada antecipada.
- Students e estados de treino (`student_*.pt`, `estado_*.pt`) ficam fora do git (`.gitignore`).
- Pastas da pipeline antiga mantêm os nomes de antes (`graphfpn_analise_unificada/`, `layer_pruning_evaluation/`, …).

## 4. Como acrescentar

**Um dataset novo** (ex.: os próximos passos minesweeper, amazon-photo, amazon-computer):
1. carregador no script de avaliação de cada modelo (`G2T-FM/exp/projeto/fase1/limix-icl/<ds>/evaluation.toml`;
   `HP` e `carregar()` em `NodePFN/projeto/avaliar_projeto.py`; `DATASETS` e `load_classic()` em
   `GraphFPN/projeto/fase23_graphpfn.py` e `ALL_DATASETS` em `kd_v2_graphpfn.py`; `Ephris/projeto/comum.py`);
2. split igual em todos os modelos (`random_split` 10/10/80 com semente 0, ou o split oficial do GraphLand);
3. acaso do dataset em `ACASO` de `pipeline/fase3_ordens.py`;
4. rodar as fases na ordem 1 → 2 → 3 → 4 (pelo `rodar_pipeline.py`, acrescentando o dataset em `SUPORTE` e em
   `DATASETS`) e, se for para a transferência, acrescentar o dataset em `CFG` de
   `pipeline/transferencia_poda.py` e `pipeline/kd_transferencia.py` e em `MODELOS` de
   `Relatorio/transferencia/gerar_tabelas.py`.

**Um modelo novo:** código oficial em `<Modelo>/` (sem o `.git` do repositório de origem), scripts em
`<Modelo>/projeto/` com os nomes e o cabeçalho da seção 2, resultados em `<modelo>_<etapa>/`, uma linha na tabela de
modelos de `DOCUMENTACAO.md` e as entradas do modelo em `rodar_pipeline.py` (`PYTHON`, `SUPORTE`, `CAMADAS_KD`,
`PASTA_KD`, `PASTA_KD_TRANSF`, `SCRIPT_KD` e os comandos de cada etapa em `montar_comandos()`).

## 5. Trazer o restante da cópia de trabalho separada

A fila da parte 2 do EXP-013 ainda roda em `.claude/worktrees/fase1-novos-gfms/`, com a organização antiga (scripts
na raiz e direto nas pastas dos modelos). O que falta trazer de lá são **só resultados**; os scripts de lá já estão
no `main`, reorganizados. Os nomes das pastas de resultados não mudaram, então basta copiar:

```bash
# da raiz do main, depois que o arquivo de estado da fila mostrar "FILA CONCLUÍDA" (e "FILA3 CONCLUÍDA")
W=.claude/worktrees/fase1-novos-gfms
rsync -a --update --exclude='*.pt' "$W/Block_Influence_results/" Block_Influence_results/
~/miniconda3/envs/ephris/bin/python Relatorio/transferencia/gerar_tabelas.py
(cd Relatorio && ~/miniconda3/envs/latex/bin/tectonic -X compile relatorio_transferencia.tex)
git add Block_Influence_results Relatorio && git commit
```

- `--exclude='*.pt'`: os students não vão para o `main` (vários GB). Se precisar deles no `main` para rodar a
  avaliação cruzada de novo, copie à parte, sem `git add`.
- `--update`: não sobrescreve um arquivo que estiver mais novo no `main`.
- Se algum script for alterado na cópia separada, a mudança deve ser refeita no arquivo correspondente do `main`:

| Cópia separada (organização antiga) | `main` |
|---|---|
| `fase3_ordens.py`, `transferencia_poda.py`, `kd_transferencia.py`, `resumo_*.py`, `rodar_com_vigia.py` | `pipeline/` |
| `GraphFPN/{fase23_graphpfn,kd_v2_graphpfn,custo_graphpfn}.py` | `GraphFPN/projeto/` |
| `G2T-FM/{preparar_dados_projeto,fase2_bi_ablacao,fase3_poda,fase4_kd,custo_g2t}.py` | `G2T-FM/projeto/` |
| `NodePFN/{avaliar_projeto,fase2_bi_ablacao,fase3_poda,fase4_kd,custo_nodepfn}.py` | `NodePFN/projeto/` |
| `Notebooks_pruning/{anygraph_fases,anygraph_kd,evaluate_layer_pruning_repetido}.py`, `resumo_anygraph.py`, `download_assets.py` | `AnyGraph/projeto/` |
| `fila3_kd_aleatorio_city.sh`, `ADIAR_KD_ALEATORIO_CITY` | não trazer (temporários da fila) |

Depois disso, a cópia separada pode ser removida (`git worktree remove .claude/worktrees/fase1-novos-gfms` e
`git branch -D worktree-fase1-novos-gfms`).
