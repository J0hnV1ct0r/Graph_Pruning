import copy
import os
import torch as t
import torch.nn.functional as F
import pandas as pd
import numpy as np


class BlockInfluenceLayerPruner:
    """
    Pruner de camadas baseado em Block Influence (BI) para o modelo OpenGraph.

    Permite:
    - calcular o BI de cada GTLayer;
    - identificar camadas com menor BI;
    - remover camadas específicas;
    - gerar relatório de parâmetros e camadas.
    """

    def __init__(self, model, df_bi=None, bi_csv_path=None):
        self.model = copy.deepcopy(model)
        self.original_model = copy.deepcopy(model)
        self.df_bi = None
        self.removed_layers = []

        if df_bi is not None:
            self.set_bi_scores(df_bi)

        elif bi_csv_path is not None and os.path.exists(bi_csv_path):
            self.load_bi_from_csv(bi_csv_path)

    # ============================================================
    # 1. BLOCK INFLUENCE
    # ============================================================

    @staticmethod
    def block_influence(
        input_hidden_state: t.Tensor,
        output_hidden_state: t.Tensor,
        angular: bool = False,
        eps: float = 1e-8,
        reduce_mean: bool = True
    ):
        """
        Calcula o Block Influence entre a entrada e a saída de uma camada.
        """

        d = input_hidden_state.shape[-1]

        h_in = input_hidden_state.reshape(-1, d)
        h_out = output_hidden_state.reshape(-1, d)

        sim = F.cosine_similarity(
            h_in,
            h_out,
            dim=-1,
            eps=eps
        )

        sim = sim.clamp(-1.0, 1.0)

        if angular:
            bi = t.arccos(sim) / np.pi
        else:
            bi = 1.0 - sim

        if reduce_mean:
            return bi.mean().item()

        return bi

    # ============================================================
    # 2. COMPUTAR BI
    # ============================================================

    def compute_bi(
        self,
        input_embeds: t.Tensor,
        angular: bool = False
    ):
        """
        Calcula o Block Influence de cada GTLayer do OpenGraph.

        Retorna DataFrame com:
        - Layer
        - Block Influence
        """

        self.model.eval()

        rows = []

        with t.no_grad():

            h = input_embeds.detach()

            for layer_idx, layer in enumerate(
                self.model.graphTransformer.gt_layers
            ):

                h_in = h

                h_out = layer(h_in) / 10

                bi = self.block_influence(
                    h_in,
                    h_out,
                    angular=angular,
                    reduce_mean=True
                )

                rows.append({
                    "Layer": layer_idx,
                    "Block Influence": bi
                })

                h = h_out

        self.df_bi = pd.DataFrame(rows)

        return self.df_bi

    # ============================================================
    # 3. DEFINIR BI MANUALMENTE
    # ============================================================

    def set_bi_scores(self, df_bi: pd.DataFrame):
        """
        Define os scores de Block Influence diretamente.
        """

        self.df_bi = df_bi.copy()

    # ============================================================
    # 4. CARREGAR BI DE CSV
    # ============================================================

    def load_bi_from_csv(self, csv_path: str):
        """
        Carrega scores de Block Influence de um CSV.
        """

        df = pd.read_csv(csv_path)

        if "Layer" in df.columns and "Block Influence" in df.columns:

            self.df_bi = df

        else:
            raise ValueError(
                f"Formato de CSV nao reconhecido para BI: {csv_path}"
            )

        return self.df_bi

    # ============================================================
    # 5. ESCOLHER CAMADAS PARA REMOVER
    # ============================================================

    def get_layers_to_remove(
        self,
        n_layers_to_prune: int = 1
    ):
        """
        Retorna os índices das camadas com menor Block Influence.
        """

        if self.df_bi is None:
            raise ValueError(
                "Scores de Block Influence nao foram calculados "
                "ou carregados."
            )

        ranking = (
            self.df_bi
            .sort_values("Block Influence")
        )

        layers_to_remove = (
            ranking["Layer"]
            .iloc[:n_layers_to_prune]
            .tolist()
        )

        return sorted(layers_to_remove)

    # ============================================================
    # 6. PRUNING AUTOMÁTICO
    # ============================================================

    def prune(
        self,
        n_layers_to_prune: int = 1
    ):
        """
        Remove as n camadas com menor Block Influence.
        """

        if n_layers_to_prune == 0:

            self.model = copy.deepcopy(
                self.original_model
            )

            self.removed_layers = []

            return self.model

        layers_to_remove = self.get_layers_to_remove(
            n_layers_to_prune
        )

        return self.prune_specific_layers(
            layers_to_remove
        )

    # ============================================================
    # 7. PRUNING DE CAMADAS ESPECÍFICAS
    # ============================================================

    def prune_specific_layers(
        self,
        layers_to_remove
    ):
        """
        Remove GTLayers específicas do OpenGraph.

        Args:
            layers_to_remove:
                Lista contendo os índices das camadas
                que devem ser removidas.
        """

        self.model = copy.deepcopy(
            self.original_model
        )

        self.removed_layers = sorted(
            layers_to_remove
        )

        gt = self.model.graphTransformer

        to_remove = set(
            layers_to_remove
        )

        new_gt = [
            layer
            for i, layer in enumerate(gt.gt_layers)
            if i not in to_remove
        ]

        gt.gt_layers = t.nn.Sequential(
            *new_gt
        )

        return self.model

    # ============================================================
    # 8. OBTER MODELO
    # ============================================================

    def get_model(self):
        """
        Retorna o modelo atualmente podado.
        """

        return self.model

    # ============================================================
    # 9. RELATÓRIO DE PARÂMETROS
    # ============================================================

    def sparsity_report(self):
        """
        Relatório da redução de parâmetros.
        """

        total_orig = sum(
            p.numel()
            for p in self.original_model.parameters()
        )

        total_curr = sum(
            p.numel()
            for p in self.model.parameters()
        )

        removed_params = (
            total_orig - total_curr
        )

        return {
            "Total Params Original": total_orig,
            "Total Params Pruned": total_curr,
            "Params Removed": removed_params,
            "Total Sparsity (%)":
                100.0 * removed_params / total_orig
                if total_orig > 0 else 0.0
        }

    # ============================================================
    # 10. RELATÓRIO DAS CAMADAS
    # ============================================================

    def layer_report(self):
        """
        Retorna um DataFrame com o estado das GTLayers.
        """

        original_layers = len(
            self.original_model
            .graphTransformer
            .gt_layers
        )

        current_layers = len(
            self.model
            .graphTransformer
            .gt_layers
        )

        return pd.DataFrame([{
            "Original Layers": original_layers,
            "Remaining Layers": current_layers,
            "Removed Layers": str(self.removed_layers),
            "Layers Removed Count": len(
                self.removed_layers
            )
        }])

    # ============================================================
    # 11. RESUMO
    # ============================================================

    def summary(self):

        report = self.sparsity_report()

        print("=" * 60)
        print(" RESUMO DA PODA DE CAMADAS (BLOCK INFLUENCE)")
        print("=" * 60)

        print(
            f"Parametros Totais: "
            f"{report['Total Params Original']:,} "
            f"-> "
            f"{report['Total Params Pruned']:,} "
            f"(-{report['Total Sparsity (%)']:.2f}%)"
        )

        print("-" * 60)

        print(
            f"Camadas originais: "
            f"{len(self.original_model.graphTransformer.gt_layers)}"
        )

        print(
            f"Camadas restantes: "
            f"{len(self.model.graphTransformer.gt_layers)}"
        )

        print(
            f"Camadas removidas: "
            f"{self.removed_layers}"
        )

        print("=" * 60)