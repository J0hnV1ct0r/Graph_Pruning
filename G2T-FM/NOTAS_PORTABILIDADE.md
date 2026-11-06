# G2T-FM neste projeto

Código copiado de https://github.com/yandex-research/G2T-FM (branch `main`, baixado em 21/10/2026), licenças em
`LICENSE`, `LICENSES/` e `NOTICE`. Nenhum arquivo do código original foi alterado.

Backbone usado: LimiX-16M (`checkpoints/LimiX-16M.ckpt`, cópia do mesmo arquivo de `GraphFPN/checkpoints/`), para ficar
comparável ao GraphPFN. Modelo carregado com `load_tfm("limix", ...)`: `LimiXWrapper` → `FeaturesTransformer`,
16.526.413 parâmetros, 12 camadas `EncoderBaseLayer` (a mesma classe de bloco que o GraphPFN envolve com os adaptadores
de grafo). `checkpoints/pearl_random.pt` vem do próprio repositório.

## Ambiente

`~/venvs/g2tfm`: venv com `--system-site-packages` sobre o conda `graph_pruning` (Python 3.12, torch 2.11.0+cu128).
O código usa sintaxe do Python 3.12 (`def check[T](...)` em `lib/util.py`), então o conda `graphpfn` (3.11) não serve.
O `pyproject.toml` fixa torch 2.4.0 e flash-attn 2.8.3; aqui a atenção usa o caminho em torch puro do LimiX.
Pacotes instalados só no venv: dgl 2.4.0+cu121 (o `graph_pruning` tem o 2.1), loguru, tomli_w, rtdl-num-embeddings,
rtdl-revisiting-models, kditransform, tensorboard.

```bash
cd G2T-FM
~/venvs/g2tfm/bin/python bin/go.py exp/g2t/main/limix/<regime>/10/<dataset>/<config>.toml --force
```
