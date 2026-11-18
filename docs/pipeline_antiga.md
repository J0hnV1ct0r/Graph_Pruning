# Pipeline antiga

Experimentos anteriores ao protocolo comum dos PFNs. Alimentavam a versão completa anterior do relatório,
`Relatorio/relatorio_completo_backup.tex`, e foram retirados do relatório atual porque não são comparáveis entre
modelos: usam datasets, splits ou configurações próprios de cada modelo. A pipeline atual está em
[`pipeline_atual.md`](pipeline_atual.md); pastas e ambientes, em [`../DOCUMENTACAO.md`](../DOCUMENTACAO.md).

_Atualizado em 18/11/2026._

---

## 1. AnyGraph: Block Influence e poda por Tiers (`main`)

Zero-shot, checkpoints `pretrain_link1` e `pretrain_link2`, datasets próprios do AnyGraph (arxiv, cora, home,
pubmed, tech). As mesmas camadas são removidas em todos os experts.

| Etapa | Arquivo | Saída em `Block_Influence_results/` |
|---|---|---|
| BI cosseno e angular por expert e camada | `Notebooks_block_influence/block_influence.ipynb` | `cosseno/`, `angular/`, `comparacao_angular_x_cosseno/` |
| Poda Tier 1 (camadas 5 e 6) e Tier 2 (4–7), 1 repetição | `Notebooks_pruning/evaluate_layer_pruning.ipynb` | `layer_pruning_evaluation/` |
| Mesma poda, 10 repetições pareadas e recarga corrigida | `Notebooks_pruning/evaluate_layer_pruning_repetido.py --checkpoint <ckpt> --dataset <ds>` (da raiz, conda `graph_pruning`) | `layer_pruning_evaluation_repetido/` |
| Poda de camadas pelo BI | `AnyGraph/Pruning_Methods/layer_pruner.py` (`BlockInfluenceLayerPruner`) | — |

Dados e checkpoints: `python download_assets.py --all` (Hugging Face). Os rankings de cosseno e angular são
idênticos; as camadas 4–7 são as de menor BI em todos os experts.

## 2. GraphPFN: análise em splits Planetoid (`main`, `GraphFPN/`, conda `graphpfn`)

Análise original do GraphPFN, com cora e pubmed no split Planetoid (PCA de 64 dimensões), tolokers-2,
city-roads-M e artnet-views, e 1 membro de ensemble. Não é comparável com os outros modelos.

| Etapa | Arquivo | Saída em `Block_Influence_results/` |
|---|---|---|
| Fase 0 (invalidada): limiares fixos de BI em amostras de 500 nós | `generate_eval_notebooks.py` (gera os notebooks de avaliação) | resultados não estão no repositório |
| Análise unificada: BI por camada e sub-bloco, ablação, critérios V1–V4 | `analise_camadas_e_poda_graphpfn.ipynb` | `graphfpn_analise_unificada/` |
| Poda das `k` camadas de menor BI, livre | `poda_cosseno_k_menores_graphpfn.ipynb` | `graphfpn_poda_cosseno_k/` |
| Mesma poda com a camada 0 protegida | `poda_cosseno_k_menores_sem_camada0_graphpfn.ipynb` | `graphfpn_poda_cosseno_k_sem_camada0/` |
| Poda no protocolo do artigo (10 datasets, ensemble de 10, 3 sementes) | `poda_cosseno_artigo_graphpfn.ipynb` | `graphfpn_poda_cosseno_artigo/` |
| KD v1, student podado × aleatório, 200 passos fixos | `kd_podado_vs_aleatorio_graphpfn.ipynb` | `graphfpn_kd_podado_vs_aleatorio/` |

**O que continua sendo usado na pipeline atual:** o modelo inteiro, o BI e as avaliações de
`graphfpn_poda_cosseno_artigo/` (mesmo pipeline e sementes do protocolo comum) são reaproveitados por
`GraphFPN/fase23_graphpfn.py`; e o KD atual do GraphPFN é o `GraphFPN/kd_v2_graphpfn.py`, versão 2 do notebook de KD
(melhor ponto pela validação e parada antecipada).

## 3. Relatório

`Relatorio/relatorio_completo_backup.tex`: versão completa anterior do relatório, com as seções do AnyGraph e a análise do GraphPFN em splits Planetoid, além dos PFNs no protocolo comum. As figuras
dessa versão estão em `Relatorio/figuras/anygraph/` e `Relatorio/figuras/graphpfn/`.
