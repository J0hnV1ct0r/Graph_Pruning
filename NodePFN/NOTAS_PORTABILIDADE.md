# NodePFN neste projeto

Código copiado de https://github.com/jeongwhanchoi/NodePFN (branch `main`, baixado em 21/10/2026), licença em `LICENSE`.
Checkpoint `models_ckpts/nodepfn/checkpoint_epoch_30.ckpt` (116.129.388 bytes) baixado do Git LFS do repositório oficial;
SHA-256 `6c2da16cedd2646e81b3e6e36ff71f269fcb5d63d04537a25c603bf407d7a961`, igual ao do ponteiro LFS publicado.

Modelo carregado: `TransformerModel`, 29.008.404 parâmetros, 12 camadas `TransformerEncoderLayer` (emsize 256, 4 cabeças).

## Ambiente

`~/venvs/nodepfn`: venv com `--system-site-packages` sobre o conda `graph_pruning` (Python 3.12, torch 2.11.0+cu128).
O `requirements.txt` original pede torch 2.2 + cu121, que não roda na RTX 5060 Ti (Blackwell exige CUDA >= 12.8).
Pacotes instalados só no venv: ConfigSpace 1.2.0, gpytorch 1.15.2 (+ linear_operator), gdown, tensorboardX.

```bash
cd NodePFN/nodepfn
~/venvs/nodepfn/bin/python node_classification.py ...   # sempre com e=30 (época do checkpoint)
```

## Alterações no código original (portabilidade)

- `nodepfn/layer.py`: `Optional`, `Tensor`, `Module`, `MultiheadAttention`, `Linear`, `Dropout` e `LayerNorm`
  passam a ser importados de `typing`, `torch` e `torch.nn`, porque `torch.nn.modules.transformer` não os reexporta mais.
- `nodepfn/scripts/model_builder.py`: `weights_only=False` nos 3 `torch.load` (no torch >= 2.6 o padrão é `True`, e o
  checkpoint guarda a config junto com os pesos). Só é seguro porque o checkpoint vem do repositório oficial com hash conferido.

## Cuidado

`NodePFNClassifier` usa `e=0` por padrão; sem `checkpoint_epoch_0.ckpt` ele tenta baixar um arquivo de um link que
devolve uma página HTML (aconteceu no primeiro teste; o arquivo foi apagado). Passar sempre `e=30`.
