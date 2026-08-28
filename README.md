# Graph_Pruning & AnyGraph

Repositório para investigação e experimentos de poda (pruning) em Graph Foundation Models (GFMs) baseados na arquitetura **AnyGraph**.

---

## 🚀 Configuração do Ambiente

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

