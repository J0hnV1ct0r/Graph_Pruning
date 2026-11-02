"""
Monta G2T-FM/data/ para os experimentos do projeto, usando os mesmos dados e splits do GraphPFN.

- GraphLand (tolokers-2, city-roads-M): links para o cache usado pelo GraphPFN; split oficial RL.
- Clássicos (amazon-ratings, wiki-cs): links para o cache do PyG usado pelo GraphPFN e um split.npz gerado
  com a mesma função random_split de GraphFPN/poda_cosseno_artigo_graphpfn.ipynb (10/10/80, semente 0).
  O artigo do G2T-FM usa um split estratificado que não foi publicado; usar o do GraphPFN deixa os modelos
  comparáveis entre si.

Uso (de dentro de G2T-FM/): python preparar_dados_projeto.py
"""
from pathlib import Path

import numpy as np

GRAPHLAND = Path.home() / ".cache" / "graphland"
PYG = Path.home() / ".cache" / "pyg_data"
DATA = Path(__file__).resolve().parent / "data"


def random_split(n, seed=0):
    """Igual a random_split de poda_cosseno_artigo_graphpfn.ipynb."""
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.1 * n), int(0.2 * n)
    masks = {k: np.zeros(n, bool) for k in ["train", "val", "test"]}
    masks["train"][perm[:a]] = True
    masks["val"][perm[a:b]] = True
    masks["test"][perm[b:]] = True
    return masks


def link(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src)


for name in ["tolokers-2", "city-roads-M"]:
    link(DATA / name, GRAPHLAND / name / name / "raw" / name)

# amazon-ratings: HeterophilousGraphDataset(root=data/amazon-ratings) procura data/amazon-ratings/amazon_ratings/
link(DATA / "amazon-ratings" / "amazon_ratings", PYG / "amazon_ratings")
# wiki-cs: WikiCS(root=data/wiki-cs) procura data/wiki-cs/raw e data/wiki-cs/processed
link(DATA / "wiki-cs" / "raw", PYG / "wiki-cs" / "raw")
link(DATA / "wiki-cs" / "processed", PYG / "wiki-cs" / "processed")

for name, n in [("amazon-ratings", 24492), ("wiki-cs", 11701)]:
    masks = random_split(n)
    np.savez(DATA / name / "split.npz", **masks)
    print(name, {k: int(v.sum()) for k, v in masks.items()})
