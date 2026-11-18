# Documentação do repositório Graph_Pruning

Poda de camadas inteiras em Graph Foundation Models (GFMs), com as camadas escolhidas pelo **Block Influence** do
ShortGPT, $BI = 1 - \cos(h_{in}, h_{out})$, e recuperação por **destilação de conhecimento (KD)**, nos modelos
AnyGraph, GraphPFN, G2T-FM, NodePFN e Ephris.

_Atualizado em 18/11/2026._

---

## 1. Organização

Tudo está no branch `main`, na raiz do repositório:

- uma pasta por modelo (`AnyGraph/`, `GraphFPN/`, `G2T-FM/`, `NodePFN/`, `Ephris/`), com o código do modelo e os
  scripts do projeto (no Ephris, em `Ephris/projeto/`);
- scripts comuns na raiz (`fase3_ordens.py`, `transferencia_poda.py`, `kd_transferencia.py`, `resumo_*.py`,
  `rodar_com_vigia.py`);
- resultados em `Block_Influence_results/`, uma pasta por modelo e fase;
- relatórios em `Relatorio/`.

Ficam fora do git: checkpoints dos modelos, datasets, o cache de dados do Ephris e os students salvos do KD
(`student_*.pt`, vários GB).

**Resultados parciais (18/11/2026):** a parte 2 da transferência entre datasets ainda roda numa cópia de trabalho
separada (`.claude/worktrees/fase1-novos-gfms/`, branch `worktree-fase1-novos-gfms`); os resultados dela são copiados
para `Block_Influence_results/` do `main` quando terminam.

---

## 2. Modelos

| Modelo | Pasta | Ambiente | Camadas podadas |
|---|---|---|---|
| AnyGraph (`pretrain_link1/2`) | `AnyGraph/` | conda `graph_pruning` | 8 camadas MLP de cada expert |
| GraphPFN | `GraphFPN/` | conda `graphpfn` | 12 camadas do LimiX com adaptadores de grafo |
| G2T-FM (G2T-LimiX) | `G2T-FM/` | `~/venvs/g2tfm` | 12 camadas do LimiX-16M |
| NodePFN | `NodePFN/` | `~/venvs/nodepfn` | 12 camadas Transformer |
| Ephris | `Ephris/` (`Ephris/projeto/` = scripts do projeto) | conda `ephris` | 10 camadas MP_ICL |

Publicações: GraphPFN (ICML 2026), G2T-FM (workshop da NeurIPS 2025), NodePFN (ICLR 2026), Ephris (preprint,
arXiv:2609.37057). Os quatro PFNs foram pré-treinados só com dados sintéticos. Detalhes na Tabela 1 de
`Relatorio/relatorio.pdf`.

Ambientes e mudanças de portabilidade: `G2T-FM/NOTAS_PORTABILIDADE.md` e `NodePFN/NOTAS_PORTABILIDADE.md`. Os
pins originais de PyTorch não rodam na RTX 5060 Ti.

---

## 3. Documentos

| Documento | Conteúdo |
|---|---|
| [`docs/pipeline_atual.md`](docs/pipeline_atual.md) | protocolo comum dos PFNs (fases 1–4 e custo), transferência entre datasets, como rodar cada fase, relatórios atuais, estado e próximos passos |
| [`docs/pipeline_antiga.md`](docs/pipeline_antiga.md) | pipeline anterior: Block Influence e poda por Tiers no AnyGraph, análise do GraphPFN em splits Planetoid e a versão completa anterior do relatório |
