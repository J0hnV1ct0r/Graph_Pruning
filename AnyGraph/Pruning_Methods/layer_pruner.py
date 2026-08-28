import copy
import os
import torch as t
import torch.nn.functional as F
import pandas as pd
import numpy as np


class BlockInfluenceLayerPruner:
    """
    Pruner de Camadas baseado em Block Influence (BI) para o modelo AnyGraph.
    
    Implementa a metodologia do ShortGPT adaptada para arquiteturas Mixture-of-Experts (MoE),
    permitindo remover as camadas com menor impacto funcional tanto por expert (especializado)
    quanto globalmente (uniforme).
    """

    def __init__(self, model, df_bi=None, bi_csv_path=None):
        self.model = copy.deepcopy(model)
        self.original_model = copy.deepcopy(model)
        self.df_bi = None
        self.removed_layers_per_expert = {}  # {expert_id: [removed_layer_indices]}
        
        if df_bi is not None:
            self.set_bi_scores(df_bi)
        elif bi_csv_path is not None and os.path.exists(bi_csv_path):
            self.load_bi_from_csv(bi_csv_path)

    @staticmethod
    def block_influence(
        input_hidden_state: t.Tensor,
        output_hidden_state: t.Tensor,
        angular: bool = False,
        eps: float = 1e-8,
        reduce_mean: bool = True
    ):
        """
        Calcula o Block Influence (BI) entre entrada e saida de uma camada.
        """
        d = input_hidden_state.shape[-1]
        h_in = input_hidden_state.reshape(-1, d)
        h_out = output_hidden_state.reshape(-1, d)

        sim = F.cosine_similarity(h_in, h_out, dim=-1, eps=eps)
        sim = sim.clamp(-1.0, 1.0)

        if angular:
            bi = t.arccos(sim) / np.pi
        else:
            bi = 1.0 - sim

        if reduce_mean:
            return bi.mean().item()
        return bi

    def compute_bi(self, input_embeds: t.Tensor, angular: bool = False):
        """
        Calcula o Block Influence para cada camada e expert propagando embeddings de amostra.
        Retorna DataFrame com colunas: ['Expert', 'Layer', 'Block Influence']
        """
        self.model.eval()
        rows = []

        with t.no_grad():
            for exp_idx, expert in enumerate(self.model.experts):
                if hasattr(expert, "trainable_nn") and hasattr(expert.trainable_nn, "dense_layers"):
                    mlp = expert.trainable_nn
                    h = input_embeds.detach()
                    num_layers = len(mlp.dense_layers)

                    for layer_idx in range(num_layers):
                        h_in = h
                        dense_out = mlp.dense_layers[layer_idx](h_in)
                        if hasattr(mlp, "dropout") and mlp.training:
                            dense_out = mlp.dropout(dense_out)
                        h_out = mlp.layer_norms[layer_idx](dense_out + h_in)

                        bi = self.block_influence(h_in, h_out, angular=angular, reduce_mean=True)
                        rows.append({
                            "Expert": exp_idx,
                            "Layer": layer_idx,
                            "Block Influence": bi
                        })
                        h = h_out

        self.df_bi = pd.DataFrame(rows)
        return self.df_bi

    def set_bi_scores(self, df_bi: pd.DataFrame):
        """
        Define a tabela de Block Influence diretamente a partir de um DataFrame.
        """
        self.df_bi = df_bi.copy()

    def load_bi_from_csv(self, csv_path: str):
        """
        Carrega scores de Block Influence a partir de um arquivo CSV.
        Suporta formato long (Expert, Layer, Block Influence) ou wide (pivot por Expert).
        """
        df = pd.read_csv(csv_path)
        if "Expert" in df.columns and "Layer" in df.columns and "Block Influence" in df.columns:
            self.df_bi = df
        elif "Expert" in df.columns:
            # Formato wide (pivot)
            df_long = df.melt(id_vars=["Expert"], var_name="Layer", value_name="Block Influence")
            df_long["Layer"] = df_long["Layer"].astype(int)
            self.df_bi = df_long
        else:
            raise ValueError(f"Formato de CSV nao reconhecido para BI: {csv_path}")
        return self.df_bi

    def get_layers_to_remove(self, n_layers_to_prune: int = 1, strategy: str = "per_expert"):
        """
        Identifica os indices das camadas a serem removidas por expert.
        
        Args:
            n_layers_to_prune (int): Quantidade de camadas a podar em cada expert.
            strategy (str): 'per_expert' (remove as n camadas com menor BI de CADA expert)
                            ou 'global' (remove as n camadas com menor MEDIA de BI geral).
        
        Returns:
            dict: {expert_id: [indices_das_camadas_a_remover]}
        """
        if self.df_bi is None:
            raise ValueError("Scores de Block Influence nao foram calculados ou carregados. Execute compute_bi() ou load_bi_from_csv().")

        num_experts = len(self.model.experts)
        layers_to_remove = {}

        if strategy == "per_expert":
            for exp_idx in range(num_experts):
                exp_df = self.df_bi[self.df_bi["Expert"] == exp_idx].sort_values("Block Influence")
                to_remove = exp_df["Layer"].iloc[:n_layers_to_prune].tolist()
                layers_to_remove[exp_idx] = sorted(to_remove)

        elif strategy == "global":
            global_ranking = self.df_bi.groupby("Layer")["Block Influence"].mean().sort_values()
            to_remove_global = global_ranking.index[:n_layers_to_prune].tolist()
            for exp_idx in range(num_experts):
                layers_to_remove[exp_idx] = sorted(to_remove_global)

        else:
            raise ValueError(f"Estrategia '{strategy}' desconhecida. Use 'per_expert' ou 'global'.")

        return layers_to_remove

    def prune(self, n_layers_to_prune: int = 1, strategy: str = "per_expert"):
        """
        Executa a poda de camadas com base no menor Block Influence.
        
        Args:
            n_layers_to_prune (int): Quantidade de camadas a podar por expert.
            strategy (str): 'per_expert' ou 'global'.
            
        Returns:
            nn.Module: O modelo AnyGraph podado.
        """
        if n_layers_to_prune == 0:
            self.model = copy.deepcopy(self.original_model)
            self.removed_layers_per_expert = {i: [] for i in range(len(self.model.experts))}
            return self.model

        layers_to_remove = self.get_layers_to_remove(n_layers_to_prune, strategy=strategy)
        return self.prune_specific_layers(layers_to_remove)

    def prune_specific_layers(self, layers_to_remove_dict: dict):
        """
        Remove camadas especificas de cada expert de acordo com o dicionario fornecido.
        
        Args:
            layers_to_remove_dict (dict): {expert_id: [indices_a_remover]}
        """
        self.model = copy.deepcopy(self.original_model)
        self.removed_layers_per_expert = layers_to_remove_dict

        for exp_idx, expert in enumerate(self.model.experts):
            if exp_idx not in layers_to_remove_dict:
                continue

            to_remove = set(layers_to_remove_dict[exp_idx])
            if not to_remove:
                continue

            if hasattr(expert, "trainable_nn") and hasattr(expert.trainable_nn, "dense_layers"):
                mlp = expert.trainable_nn
                new_dense = [layer for i, layer in enumerate(mlp.dense_layers) if i not in to_remove]
                new_norms = [norm for i, norm in enumerate(mlp.layer_norms) if i not in to_remove]
                
                mlp.dense_layers = t.nn.Sequential(*new_dense)
                mlp.layer_norms = t.nn.Sequential(*new_norms)

            elif hasattr(expert, "trainable_nn") and hasattr(expert.trainable_nn, "gt_layers"):
                gt = expert.trainable_nn
                new_gt = [layer for i, layer in enumerate(gt.gt_layers) if i not in to_remove]
                gt.gt_layers = t.nn.Sequential(*new_gt)

        return self.model

    def get_model(self):
        """Retorna o modelo atualmente podado."""
        return self.model

    def sparsity_report(self):
        """
        Relatorio estruturado da reducao de parametros apos a poda de camadas.
        """
        total_orig = sum(p.numel() for p in self.original_model.parameters())
        total_curr = sum(p.numel() for p in self.model.parameters())
        removed_params = total_orig - total_curr

        orig_expert_params = sum(
            sum(p.numel() for p in exp.trainable_nn.parameters())
            for exp in self.original_model.experts if hasattr(exp, "trainable_nn")
        )
        curr_expert_params = sum(
            sum(p.numel() for p in exp.trainable_nn.parameters())
            for exp in self.model.experts if hasattr(exp, "trainable_nn")
        )
        removed_expert_params = orig_expert_params - curr_expert_params

        return {
            "Total Params Original": total_orig,
            "Total Params Pruned": total_curr,
            "Params Removed": removed_params,
            "Total Sparsity (%)": 100.0 * removed_params / total_orig if total_orig > 0 else 0.0,
            "Expert Params Original": orig_expert_params,
            "Expert Params Pruned": curr_expert_params,
            "Expert Sparsity (%)": 100.0 * removed_expert_params / orig_expert_params if orig_expert_params > 0 else 0.0,
        }

    def layer_report(self):
        """
        Retorna um DataFrame detalhando o estado de cada camada e expert.
        """
        rows = []
        for exp_idx, expert in enumerate(self.model.experts):
            if hasattr(expert, "trainable_nn") and hasattr(expert.trainable_nn, "dense_layers"):
                mlp = expert.trainable_nn
                orig_layers = len(self.original_model.experts[exp_idx].trainable_nn.dense_layers)
                curr_layers = len(mlp.dense_layers)
                removed = self.removed_layers_per_expert.get(exp_idx, [])
                
                rows.append({
                    "Expert": exp_idx,
                    "Original Layers": orig_layers,
                    "Remaining Layers": curr_layers,
                    "Removed Layers": str(removed),
                    "Layers Removed Count": len(removed)
                })
        return pd.DataFrame(rows)

    def summary(self):
        """Imprime um resumo textual claro da poda realizada."""
        report = self.sparsity_report()
        print("=" * 60)
        print(" RESUMO DA PODA DE CAMADAS (BLOCK INFLUENCE)")
        print("=" * 60)
        print(f"Parametros Totais:   {report['Total Params Original']:,} -> {report['Total Params Pruned']:,} (-{report['Total Sparsity (%)']:.2f}%)")
        print(f"Parametros Experts:  {report['Expert Params Original']:,} -> {report['Expert Params Pruned']:,} (-{report['Expert Sparsity (%)']:.2f}%)")
        print("-" * 60)
        print("Camadas removidas por expert:")
        for exp_idx, removed in sorted(self.removed_layers_per_expert.items()):
            print(f"  Expert {exp_idx:02d}: Camadas removidas {removed}")
        print("=" * 60)
