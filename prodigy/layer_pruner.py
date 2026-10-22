"""Block Influence pruning utilities for PRODIGY's S2/UX/M2 architecture.

The class supports:
- computing OpenGraph-style cosine BI for each internal S2 and M2 layer;
- loading BI results from a DataFrame/CSV;
- Tier 1: remove the lowest-BI S2 layer;
- Tier 2: independently remove the lowest-BI layer from S2 and M2;
- structural layer removal and parameter-count reports.

Important: S2 layer 0 changes the feature width before message passing. To make
its BI cosine comparison dimensionally valid, its input is projected with that
layer's ``lin_x``. M2 BI is calculated on the first ``start_right`` rows, as in
the notebook's previous experiment. These choices are recorded in the output.
"""

from __future__ import annotations

import copy
import gc
import os
import re
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class PRODIGYBlockInfluencePruner:
    """Calculate layer-wise BI and apply structural pruning to S2 and M2.

    Parameters
    ----------
    model:
        A constructed PRODIGY ``SingleLayerGeneralGNN`` with S2 at
        ``layer_list[0]`` and M2 at ``layer_list[2]``.
    df_bi:
        Optional table containing precomputed BI scores. Expected to include
        ``Bloco`` or ``Block``, ``Camada`` or ``Layer``, and a BI column such
        as ``BI média`` or ``Block Influence``.
    bi_csv_path:
        Optional CSV path with the same table format.
    """

    S2_INDEX = 0
    M2_INDEX = 2

    def __init__(
        self,
        model: nn.Module,
        df_bi: Optional[pd.DataFrame] = None,
        bi_csv_path: Optional[str] = None,
    ) -> None:
        self.original_model = copy.deepcopy(model)
        self.model = copy.deepcopy(model)
        self.df_bi: Optional[pd.DataFrame] = None
        self.removed_layers: dict[str, list[int]] = {"S2": [], "M2": []}

        self._validate_architecture(self.original_model)

        if df_bi is not None:
            self.set_bi_scores(df_bi)
        elif bi_csv_path is not None:
            if not os.path.exists(bi_csv_path):
                raise FileNotFoundError(f"Arquivo de BI não encontrado: {bi_csv_path}")
            self.load_bi_from_csv(bi_csv_path)

    @staticmethod
    def block_influence(
        input_hidden_state: torch.Tensor,
        output_hidden_state: torch.Tensor,
        angular: bool = False,
        eps: float = 1e-8,
        reduce_mean: bool = True,
    ):
        """OpenGraph-style BI: mean(1 - cosine) or normalized angular BI."""
        if input_hidden_state.shape != output_hidden_state.shape:
            raise ValueError(
                "BI exige representações com o mesmo shape; recebido "
                f"{tuple(input_hidden_state.shape)} e "
                f"{tuple(output_hidden_state.shape)}."
            )
        if input_hidden_state.ndim < 2:
            raise ValueError("As representações devem ter pelo menos 2 dimensões.")

        d = input_hidden_state.shape[-1]
        h_in = input_hidden_state.reshape(-1, d).float()
        h_out = output_hidden_state.reshape(-1, d).float()
        sim = F.cosine_similarity(h_in, h_out, dim=-1, eps=eps).clamp(-1.0, 1.0)

        if angular:
            bi = torch.arccos(sim) / np.pi
        else:
            bi = 1.0 - sim

        if reduce_mean:
            return bi.mean().item()
        return bi

    @staticmethod
    def _validate_architecture(model: nn.Module) -> None:
        if not hasattr(model, "layer_list") or len(model.layer_list) < 3:
            raise TypeError(
                "Modelo esperado: SingleLayerGeneralGNN com S2 em layer_list[0] "
                "e M2 em layer_list[2]."
            )
        s2 = model.layer_list[PRODIGYBlockInfluencePruner.S2_INDEX]
        m2 = model.layer_list[PRODIGYBlockInfluencePruner.M2_INDEX]
        if not hasattr(s2, "module_list"):
            raise TypeError("O bloco S2 não contém module_list.")
        if not hasattr(m2, "gnn_layers") or not hasattr(m2, "num_gnn_layers"):
            raise TypeError("O bloco M2 precisa de gnn_layers e num_gnn_layers.")
        if len(s2.module_list) < 1 or len(m2.gnn_layers) < 1:
            raise ValueError("S2 e M2 devem conter pelo menos uma camada interna.")

    @staticmethod
    def _batch_to_device(batch: Sequence[Any], device: torch.device) -> list[Any]:
        moved = []
        for item in batch:
            to_method = getattr(item, "to", None)
            moved.append(to_method(device) if callable(to_method) else item)
        return moved

    @staticmethod
    def _read_layer_input(args, kwargs):
        x_in = kwargs.get("x")
        if x_in is None and args:
            x_in = args[0]
        return x_in

    def compute_bi(
        self,
        val_loader,
        device: Optional[torch.device | str] = None,
        angular: bool = False,
        max_batches: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute BI per S2 GraphSAGE layer and M2 MetaGNNLayer.

        The model's original forward methods and weights are not replaced.
        Temporary hooks are always removed, including when inference fails.
        Scores are averaged per batch and then across batches, matching the
        averaging convention used by the previous notebook experiment.
        """
        self._validate_architecture(self.original_model)
        model = self.original_model
        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)

        s2 = model.layer_list[self.S2_INDEX]
        m2 = model.layer_list[self.M2_INDEX]
        captured: dict[str, dict[int, Optional[dict[str, Any]]]] = {"S2": {}, "M2": {}}
        layers_by_block = {"S2": list(s2.module_list), "M2": list(m2.gnn_layers)}
        batch_scores = {
            (block, i): []
            for block, layers in layers_by_block.items()
            for i in range(len(layers))
        }
        valid_nodes = {key: 0 for key in batch_scores}
        skipped_batches = {key: 0 for key in batch_scores}
        handles = []

        def make_hook(block_name: str, layer_idx: int):
            def hook(module, args, kwargs, output):
                x_in = self._read_layer_input(args, kwargs)
                if not torch.is_tensor(x_in) or not torch.is_tensor(output):
                    captured[block_name][layer_idx] = None
                    return

                # First S2 layer receives a representation whose width can
                # differ from the convolution output. Project it with lin_x
                # before applying the cosine-based BI formula.
                with torch.no_grad():
                    bi_input = module.lin_x(x_in) if block_name == "S2" and layer_idx == 0 else x_in

                start_right = kwargs.get("start_right")
                if torch.is_tensor(start_right):
                    start_right = int(start_right.item())
                elif start_right is not None:
                    start_right = int(start_right)

                captured[block_name][layer_idx] = {
                    "input": bi_input.detach().cpu().clone(),
                    "output": output.detach().cpu().clone(),
                    "start_right": start_right,
                }
            return hook

        for i, layer in enumerate(s2.module_list):
            handles.append(layer.register_forward_hook(make_hook("S2", i), with_kwargs=True))
        for i, layer in enumerate(m2.gnn_layers):
            handles.append(layer.register_forward_hook(make_hook("M2", i), with_kwargs=True))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    captured["S2"].clear()
                    captured["M2"].clear()
                    batch = self._batch_to_device(batch, device)
                    model(*batch)

                    for block_name, layers in layers_by_block.items():
                        for i in range(len(layers)):
                            key = (block_name, i)
                            record = captured[block_name].get(i)
                            if record is None:
                                skipped_batches[key] += 1
                                continue

                            h_in = record["input"]
                            h_out = record["output"]

                            # Match activation placement in the module forward.
                            if block_name == "S2":
                                h_out = F.relu(h_out)
                            elif i < len(layers) - 1:
                                h_out = m2.gnn_non_linear(h_out)

                            # Keep the supernode prefix for M2, as in the
                            # existing PRODIGY notebook BI calculation.
                            if block_name == "M2":
                                n_supernodes = record["start_right"]
                                if n_supernodes is None or n_supernodes <= 0:
                                    skipped_batches[key] += 1
                                    continue
                                h_in = h_in[:n_supernodes]
                                h_out = h_out[:n_supernodes]

                            if h_in.shape != h_out.shape or h_in.shape[0] == 0:
                                skipped_batches[key] += 1
                                continue

                            h_in = h_in.reshape(-1, h_in.shape[-1]).float()
                            h_out = h_out.reshape(-1, h_out.shape[-1]).float()
                            sim = F.cosine_similarity(h_in, h_out, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
                            values = torch.arccos(sim) / np.pi if angular else 1.0 - sim
                            finite = torch.isfinite(values)
                            if not finite.any():
                                skipped_batches[key] += 1
                                continue

                            batch_scores[key].append(values[finite].mean().item())
                            valid_nodes[key] += int(finite.sum().item())
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows = []
        for block_name, layers in layers_by_block.items():
            for i, layer in enumerate(layers):
                key = (block_name, i)
                values = np.asarray(batch_scores[key], dtype=float)
                if values.size == 0:
                    continue
                rows.append({
                    "Bloco": block_name,
                    "Layer Index": i,
                    "Camada": f"{block_name} - camada {i + 1}",
                    "Tipo": type(layer).__name__,
                    "BI média": float(values.mean()),
                    "Desvio padrão": float(values.std(ddof=0)),
                    "Lotes válidos": int(values.size),
                    "Nós válidos": int(valid_nodes[key]),
                    "Lotes ignorados": int(skipped_batches[key]),
                    "Métrica": "BI angular" if angular else "BI (1 - cosseno)",
                    "Observação": (
                        "Entrada projetada por lin_x antes da comparação"
                        if block_name == "S2" and i == 0
                        else "M2 calculado apenas sobre o prefixo de supernós"
                        if block_name == "M2"
                        else "Entrada/saída da camada com ativação do bloco"
                    ),
                })

        self.df_bi = pd.DataFrame(rows)
        if not self.df_bi.empty:
            self.df_bi = self.df_bi.sort_values(["Bloco", "Layer Index"]).reset_index(drop=True)
        return self.df_bi

    @staticmethod
    def _normalize_bi_dataframe(df_bi: pd.DataFrame) -> pd.DataFrame:
        df = df_bi.copy()
        block_col = next((c for c in ["Bloco", "Block"] if c in df.columns), None)
        if block_col is None:
            raise ValueError("A tabela de BI precisa conter a coluna 'Bloco' ou 'Block'.")
        if block_col != "Bloco":
            df = df.rename(columns={block_col: "Bloco"})

        bi_col = next((c for c in ["BI média", "BI média entre lotes", "Block Influence", "BI"] if c in df.columns), None)
        if bi_col is None:
            raise ValueError("A tabela de BI não contém uma coluna de score reconhecida.")
        if bi_col != "BI média":
            df = df.rename(columns={bi_col: "BI média"})

        if "Layer Index" not in df.columns:
            layer_col = next((c for c in ["Camada", "Layer"] if c in df.columns), None)
            if layer_col is None:
                raise ValueError("A tabela de BI precisa conter 'Camada', 'Layer' ou 'Layer Index'.")

            def parse_index(value):
                if isinstance(value, (int, np.integer)):
                    return int(value)
                if isinstance(value, (float, np.floating)) and float(value).is_integer():
                    return int(value)
                match = re.search(r"camada\s*(\d+)", str(value), flags=re.IGNORECASE)
                if match:
                    return int(match.group(1)) - 1
                match = re.fullmatch(r"\s*\d+\s*", str(value))
                if match:
                    # A coluna 'Layer' costuma ser índice zero-based no OpenGraph.
                    return int(match.group(0))
                raise ValueError(f"Não foi possível interpretar o índice da camada: {value!r}")

            df["Layer Index"] = df[layer_col].map(parse_index)

        df["Bloco"] = df["Bloco"].astype(str).str.upper().str.strip()
        df["BI média"] = pd.to_numeric(df["BI média"], errors="raise")
        df["Layer Index"] = pd.to_numeric(df["Layer Index"], errors="raise").astype(int)
        df = df[df["Bloco"].isin(["S2", "M2"])].copy()
        if df.empty:
            raise ValueError("Nenhuma linha para os blocos S2 ou M2 foi encontrada.")
        if not np.isfinite(df["BI média"].to_numpy(dtype=float)).all():
            raise ValueError("A tabela de BI contém valores não finitos.")
        return df

    def set_bi_scores(self, df_bi: pd.DataFrame) -> pd.DataFrame:
        """Set BI scores from a DataFrame produced by compute_bi or the notebook."""
        self.df_bi = self._normalize_bi_dataframe(df_bi)
        return self.df_bi

    def load_bi_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load precomputed BI scores from CSV."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo de BI não encontrado: {csv_path}")
        return self.set_bi_scores(pd.read_csv(csv_path))

    def _lowest_bi_layer(self, block_name: str) -> int:
        if self.df_bi is None:
            raise ValueError("Calcule ou carregue os scores de BI antes do pruning.")
        block_name = block_name.upper()
        subset = self.df_bi[self.df_bi["Bloco"] == block_name]
        if subset.empty:
            raise ValueError(f"Não há scores de BI para o bloco {block_name}.")
        return int(subset.sort_values("BI média", ascending=True).iloc[0]["Layer Index"])

    def get_layers_to_remove(self) -> dict[str, list[int]]:
        """Return lowest-BI layer in each block (zero-based indices)."""
        return {
            "S2": [self._lowest_bi_layer("S2")],
            "M2": [self._lowest_bi_layer("M2")],
        }

    def prune_tier1(self) -> nn.Module:
        """Tier 1: remove the lowest-BI internal layer from S2 only."""
        scores = self.get_layers_to_remove()
        return self.prune_specific_layers(s2_layers_to_remove=scores["S2"], m2_layers_to_remove=[])

    def prune_tier2(self) -> nn.Module:
        """Tier 2: remove the lowest-BI layer from S2 and M2, independently."""
        scores = self.get_layers_to_remove()
        return self.prune_specific_layers(s2_layers_to_remove=scores["S2"], m2_layers_to_remove=scores["M2"])

    def prune_specific_layers(
        self,
        s2_layers_to_remove: Optional[Sequence[int]] = None,
        m2_layers_to_remove: Optional[Sequence[int]] = None,
    ) -> nn.Module:
        """Build a fresh pruned copy. Indices are zero-based within each block."""
        s2_remove = sorted(set(s2_layers_to_remove or []))
        m2_remove = sorted(set(m2_layers_to_remove or []))
        self.model = copy.deepcopy(self.original_model)
        self._validate_architecture(self.model)

        s2 = self.model.layer_list[self.S2_INDEX]
        m2 = self.model.layer_list[self.M2_INDEX]

        # The inspected PRODIGY configuration has no reset_after_layer entries.
        reset_after = getattr(s2, "reset_after_layer", None)
        if reset_after not in (None, [], (), set()):
            raise NotImplementedError(
                "Este pruner ainda não remapeia reset_after_layer. "
                "A configuração inspecionada deve ter reset_after_layer=None."
            )
        if getattr(m2, "gnn_layers_back", None) is not None:
            raise NotImplementedError(
                "Este pruner foi preparado para M2 sem gnn_layers_back, como na configuração inspecionada."
            )

        old_s2_count = len(s2.module_list)
        old_m2_count = len(m2.gnn_layers)
        for idx in s2_remove:
            if idx < 0 or idx >= old_s2_count:
                raise IndexError(f"Índice S2 inválido: {idx}; camadas disponíveis: {old_s2_count}.")
        # In the inspected PRODIGY setup, graph.x reaches S2 with 770
        # features and SAGE layer 0 projects them to 256. The original
        # second SAGE layer expects 256 features, so removing layer 0 and
        # retaining layer 1 would create a dimension mismatch. The current
        # BI ranking removes S2 layer 1 (zero-based index 1), which is safe.
        if 0 in s2_remove and old_s2_count > 1 and 1 not in s2_remove:
            raise ValueError(
                "Não é possível remover S2 camada 1 mantendo a camada 2 nesta "
                "configuração: a camada 2 espera embeddings de 256 dimensões, "
                "mas a entrada do S2 tem 770. Para esse experimento, remova a "
                "última camada de S2, ou implemente uma projeção de entrada compatível."
            )
        for idx in m2_remove:
            if idx < 0 or idx >= old_m2_count:
                raise IndexError(f"Índice M2 inválido: {idx}; camadas disponíveis: {old_m2_count}.")
        if len(s2_remove) >= old_s2_count:
            raise ValueError("O pruning deve preservar pelo menos uma camada S2.")
        if len(m2_remove) >= old_m2_count:
            raise ValueError("O pruning deve preservar pelo menos uma camada M2.")

        s2.module_list = nn.ModuleList([
            layer for i, layer in enumerate(s2.module_list) if i not in set(s2_remove)
        ])
        m2.gnn_layers = nn.ModuleList([
            layer for i, layer in enumerate(m2.gnn_layers) if i not in set(m2_remove)
        ])
        m2.num_gnn_layers = len(m2.gnn_layers)

        self.removed_layers = {"S2": s2_remove, "M2": m2_remove}
        return self.model

    def get_model(self) -> nn.Module:
        """Return the currently selected model (original or pruned)."""
        return self.model

    def sparsity_report(self) -> dict[str, int | float]:
        """Report the actual parameter-count reduction after structural pruning."""
        total_original = sum(p.numel() for p in self.original_model.parameters())
        total_current = sum(p.numel() for p in self.model.parameters())
        removed = total_original - total_current
        return {
            "Total Params Original": total_original,
            "Total Params Pruned": total_current,
            "Params Removed": removed,
            "Parameter Reduction (%)": 100.0 * removed / total_original if total_original else 0.0,
        }

    def layer_report(self) -> pd.DataFrame:
        s2_orig = len(self.original_model.layer_list[self.S2_INDEX].module_list)
        m2_orig = len(self.original_model.layer_list[self.M2_INDEX].gnn_layers)
        s2_now = len(self.model.layer_list[self.S2_INDEX].module_list)
        m2_now = len(self.model.layer_list[self.M2_INDEX].gnn_layers)
        return pd.DataFrame([
            {
                "Bloco": "S2",
                "Camadas originais": s2_orig,
                "Camadas restantes": s2_now,
                "Índices removidos (zero-based)": self.removed_layers.get("S2", []),
            },
            {
                "Bloco": "M2",
                "Camadas originais": m2_orig,
                "Camadas restantes": m2_now,
                "Índices removidos (zero-based)": self.removed_layers.get("M2", []),
            },
        ])

    def summary(self) -> None:
        report = self.sparsity_report()
        print("=" * 64)
        print("RESUMO DO PRUNING POR BLOCK INFLUENCE — PRODIGY")
        print("=" * 64)
        print(
            f"Parâmetros: {report['Total Params Original']:,} -> "
            f"{report['Total Params Pruned']:,} "
            f"(redução de {report['Parameter Reduction (%)']:.2f}%)"
        )
        print("Camadas removidas:", self.removed_layers)
        print("\nCamadas por bloco:")
        print(self.layer_report().to_string(index=False))
        print("=" * 64)
