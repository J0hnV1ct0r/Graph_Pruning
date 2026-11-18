# Graph_Pruning

Poda de camadas inteiras em Graph Foundation Models (GFMs), com as camadas escolhidas pelo **Block Influence** do
ShortGPT, e recuperação por destilação de conhecimento (KD). Modelos: GraphPFN, G2T-FM (G2T-LimiX), NodePFN e Ephris
(protocolo comum) e AnyGraph (pipeline antiga).

**Documentação** (modelos, protocolo, o que cada fase faz, onde estão os resultados e os relatórios):
[`DOCUMENTACAO.md`](DOCUMENTACAO.md).

---

## ▶️ Rodar o pipeline (`rodar_pipeline.py`)

O `rodar_pipeline.py`, na raiz, roda o pipeline inteiro ou partes dele: preparação dos dados → fase 1 (modelo
inteiro) → fase 2 (Block Influence, ablação e transferência da ordem de poda) → fase 3 (poda sem re-treino) → fase 4
(KD e transferência do KD) → custo → resumos → relatórios. A configuração fica em variáveis no topo do arquivo, sem
argumentos de linha de comando.

### 1. Pré-requisitos

**GPU:** NVIDIA com CUDA ≥ 12.8 (os experimentos rodaram numa RTX 5060 Ti de 16 GB).

**Ambientes Python.** Cada modelo tem o seu; o runner chama o Python certo de cada um (caminhos em `PYTHON`, no topo do
`rodar_pipeline.py`; ajuste se os seus estiverem em outro lugar).

| Ambiente | Usado por | Versões |
|---|---|---|
| conda `graph_pruning` | base dos venvs `g2tfm` e `nodepfn` (seção [Ambiente base](#-ambiente-base-graph_pruning)) | Python 3.12, torch 2.11.0+cu128 |
| `~/venvs/g2tfm` | G2T-LimiX | venv sobre o `graph_pruning`, dgl 2.4.0 |
| `~/venvs/nodepfn` | NodePFN | venv sobre o `graph_pruning` |
| conda `graphpfn` | GraphPFN | Python 3.11, torch 2.11.0+cu128, dgl 2.4.0, torch_geometric 2.8.0, `graphpfn` instalado de `GraphFPN/` |
| conda `ephris` | Ephris, resumos e tabelas dos relatórios | Python 3.11, torch 2.11.0+cu130, torch_geometric 2.8.0, `ephris` instalado de `Ephris/` |
| conda `latex` | compilar os relatórios | tectonic 0.17 |

Como criar (as versões acima são as instaladas; os comandos foram reconstruídos a partir delas e das notas de
portabilidade, `G2T-FM/NOTAS_PORTABILIDADE.md` e `NodePFN/NOTAS_PORTABILIDADE.md`):

```bash
# G2T-LimiX e NodePFN: venvs sobre o conda graph_pruning
~/miniconda3/envs/graph_pruning/bin/python -m venv --system-site-packages ~/venvs/g2tfm
~/venvs/g2tfm/bin/pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
~/venvs/g2tfm/bin/pip install loguru tomli_w rtdl-num-embeddings rtdl-revisiting-models kditransform tensorboard

~/miniconda3/envs/graph_pruning/bin/python -m venv --system-site-packages ~/venvs/nodepfn
~/venvs/nodepfn/bin/pip install ConfigSpace==1.2.0 gpytorch==1.15.2 gdown tensorboardX

# GraphPFN (o pacote pede torch<2.5, que não roda na RTX 5060 Ti; por isso --no-deps)
conda create -n graphpfn python=3.11 -y
~/miniconda3/envs/graphpfn/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
~/miniconda3/envs/graphpfn/bin/pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
~/miniconda3/envs/graphpfn/bin/pip install torch_geometric==2.8.0 pandas scikit-learn scipy psutil pyyaml
~/miniconda3/envs/graphpfn/bin/pip install -e GraphFPN --no-deps

# Ephris
conda create -n ephris python=3.11 -y
~/miniconda3/envs/ephris/bin/pip install -r Ephris/environment.lock
~/miniconda3/envs/ephris/bin/pip install -e Ephris pandas scikit-learn scipy pyyaml

# LaTeX
conda create -n latex -c conda-forge tectonic -y
```

**Pesos dos modelos:**

| Modelo | Onde | Como obter |
|---|---|---|
| GraphPFN | `GraphFPN/checkpoints/LimiX-16M.ckpt` e `graphpfn-adapters-1_3.pt` | já estão no repositório |
| G2T-LimiX | `G2T-FM/checkpoints/LimiX-16M.ckpt` (fora do git) | `cp GraphFPN/checkpoints/LimiX-16M.ckpt G2T-FM/checkpoints/` (o mesmo arquivo); `pearl_random.pt` já está no repositório |
| NodePFN | `NodePFN/models_ckpts/nodepfn/checkpoint_epoch_30.ckpt` (fora do git) | Git LFS do repositório oficial `jeongwhanchoi/NodePFN`; SHA-256 `6c2da16cedd2646e81b3e6e36ff71f269fcb5d63d04537a25c603bf407d7a961` |
| Ephris | cache do Hugging Face | baixado automaticamente na primeira execução (`nums-ai/ephris`) |

**Dados:**

- **wiki-cs, amazon-ratings, minesweeper, amazon-photo, amazon-computers, cora e citeseer** (PyG): baixados
  automaticamente na primeira execução (em `~/.cache/pyg_data/`; o G2T-LimiX baixa a sua cópia em `G2T-FM/data/`).
- **tolokers-2 e city-roads-M** (benchmark GraphLand, arXiv:2409.14500): precisam estar em
  `~/.cache/graphland/<dataset>/<dataset>/raw/<dataset>/`, com `edgelist.csv`, `features.csv`, `info.yaml`,
  `split_masks_RL.csv` e `targets.csv`.
- **Splits 10/10/80 e links em `G2T-FM/data/`** (usados pelo G2T-LimiX e pelo NodePFN): criados pela etapa
  `preparar_dados` do próprio runner.

### 2. Configurar

Abra o `rodar_pipeline.py` e edite as variáveis da seção `CONFIGURAÇÃO`:

| Variável | O que define | Padrão |
|---|---|---|
| `MODELOS` | `"graphpfn"`, `"g2t"`, `"nodepfn"`, `"ephris"` | os quatro |
| `DATASETS_EXECUTADOS`, `DATASETS_PLANEJADOS`, `DATASETS` | datasets já executados (tolokers-2, wiki-cs, amazon-ratings, city-roads-M), planejados (minesweeper, amazon-photo, amazon-computers, cora, citeseer) e os que vão rodar (combinações que um modelo não suporta são puladas) | `DATASETS` = executados + planejados |
| `SEMENTES` | sementes | 0, 1, 2 |
| `ETAPAS` | liga/desliga cada etapa (`preparar_dados`, `fase1`…`fase4_kd`, `custo`, `resumos`, `relatorios`) | todas |
| `KD_STUDENTS`, `KD_PASSOS`, `KD_PACIENCIA`, `KD_LR` | KD da fase 4 | podado e aleatório, 200, 4, varredura |
| `TRANSFERIR_PODA`, `TRANSFERIR_KD`, `TRANSF_DATASETS` | transferência entre datasets (fases 2 e 4) | ligadas; tolokers-2, wiki-cs, city-roads-M e os planejados |
| `SIMULAR` | `True` só mostra os comandos | `True` |
| `SOBRESCREVER` | `False` pula etapas que não retomam e já estão feitas | `False` |
| `PARAR_NA_FALHA` | interromper na primeira falha | `False` |
| `PYTHON`, `TECTONIC` | caminho do Python de cada ambiente e do tectonic | os da tabela acima |

### 3. Executar

Da raiz do repositório, com qualquer Python 3:

```bash
python3 rodar_pipeline.py          # com SIMULAR = True: lista os comandos, não executa nada
```

Confira a lista e, para executar de verdade, mude `SIMULAR = False` e rode de novo. Cada comando roda um por vez,
com o Python do seu ambiente; o log de cada um fica em `logs_pipeline/<data_hora>/`. Os scripts gravam o resultado a
cada avaliação, então, se algo falhar ou for interrompido, rodar de novo completa só o que faltou. Os resultados saem
em `Block_Influence_results/` e os relatórios em `Relatorio/`.

Para deixar rodando sem depender do terminal:

```bash
mkdir -p logs_pipeline && nohup python3 rodar_pipeline.py > logs_pipeline/execucao.out 2>&1 &
```

**Exemplos de configuração** (linhas para colocar logo depois da definição de `ETAPAS`):

```python
# só o NodePFN no wiki-cs, fases 1 a 3
MODELOS = ["nodepfn"]; DATASETS = ["wiki-cs"]
ETAPAS = {k: k in ("preparar_dados", "fase1", "fase2", "fase3") for k in ETAPAS}

# só recompilar os relatórios a partir dos resultados existentes
ETAPAS = {k: k == "relatorios" for k in ETAPAS}
```

Cada script também pode ser rodado sozinho; os comandos estão em [`docs/pipeline_atual.md`](docs/pipeline_atual.md).

---

## 🚀 Ambiente base (`graph_pruning`)

### 1. Ativação do Ambiente Conda existente
O ambiente conda `graph_pruning` já está configurado com Python 3.12, PyTorch 2.11+cu128 e aceleração CUDA para GPU RTX 5060 Ti.

```bash
conda activate graph_pruning
```

### 2. Recriação do Ambiente (se necessário)

#### Opção A: via `environment.yml` (Conda)
```bash
conda env create -f environment.yml
conda activate graph_pruning
```

#### Opção B: via `requirements.txt` (Pip)
```bash
conda create -n graph_pruning python=3.12 -y
conda activate graph_pruning
pip install -r requirements.txt
python -m ipykernel install --user --name graph_pruning --display-name "Python (graph_pruning)"
```

---
