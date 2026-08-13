import copy
import torch as t
import pandas as pd


class GlobalMagnitudePruner:

    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.threshold = None
        self.sparsity = None

    def prune(self, sparsity):

        self.sparsity = sparsity

        # Pesos que serão podados
        params = {
            name: param
            for name, param in self.model.named_parameters()
            if "trainable_nn" in name and "linear.weight" in name
        }

        # Todos os pesos
        all_weights = t.cat([
            param.detach().abs().flatten()
            for param in params.values()
        ])

        # Threshold global
        self.threshold = t.quantile(
            all_weights,
            sparsity
        ).item()

        # Aplica diretamente nos parâmetros do modelo
        with t.no_grad():

            for param in params.values():

                mask = param.abs() > self.threshold

                param.copy_(
                    param * mask
                )

        print(f"Global Pruning: {sparsity * 100:.1f}%")
        print(f"Threshold: {self.threshold:.8f}")

        return self.model

    def sparsity_report(self):

        total = 0
        zeros = 0

        for name, param in self.model.named_parameters():

            if "trainable_nn" in name and "linear.weight" in name:

                total += param.numel()
                zeros += (param == 0).sum().item()

        return {
            "Threshold": self.threshold,
            "Requested Sparsity (%)": self.sparsity * 100,
            "Total": total,
            "Zero": zeros,
            "Remaining": total - zeros,
            "Sparsity (%)": 100 * zeros / total
        }

    def expert_report(self):

        rows = []

        for expert in range(len(self.model.experts)):

            total = 0
            zeros = 0

            for layer in range(8):

                name = (
                    f"experts.{expert}."
                    f"trainable_nn.dense_layers.{layer}."
                    f"linear.weight"
                )

                param = dict(
                    self.model.named_parameters()
                )[name]

                total += param.numel()
                zeros += (param == 0).sum().item()

            rows.append({
                "Expert": expert,
                "Weights": total,
                "Zeros": zeros,
                "Remaining": total - zeros,
                "Sparsity (%)": 100 * zeros / total
            })

        return pd.DataFrame(rows)

    def layer_report(self):

        rows = []

        for expert in range(len(self.model.experts)):

            for layer in range(8):

                name = (
                    f"experts.{expert}."
                    f"trainable_nn.dense_layers.{layer}."
                    f"linear.weight"
                )

                param = dict(
                    self.model.named_parameters()
                )[name]

                total = param.numel()
                zeros = (param == 0).sum().item()

                rows.append({
                    "Expert": expert,
                    "Layer": layer,
                    "Weights": total,
                    "Zeros": zeros,
                    "Remaining": total - zeros,
                    "Sparsity (%)": 100 * zeros / total
                })

        return pd.DataFrame(rows)

    def remaining_statistics(self):

        weights = []

        for name, param in self.model.named_parameters():

            if "trainable_nn" in name and "linear.weight" in name:

                w = param.detach().cpu()
                w = w[w != 0]

                weights.append(w)

        weights = t.cat(weights)

        return {
            "Remaining": len(weights),
            "Mean": weights.mean().item(),
            "Std": weights.std().item(),
            "Median": weights.median().item(),
            "Min": weights.min().item(),
            "Max": weights.max().item()
        }