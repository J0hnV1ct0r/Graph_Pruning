import torch as t
import pandas as pd


class GlobalMagnitudePruner:

    def __init__(self, model):

        self.model = model

        self.threshold = None
        self.sparsity = None

        self.weight_dict = {}
        self.mask_dict = {}

        self.collect_weights()

    def collect_weights(self):

            self.weight_dict = {}

            for name, param in self.model.named_parameters():

                if (
                    "trainable_nn" in name
                    and "linear.weight" in name
                ):

                    self.weight_dict[name] = param

    def compute_threshold(self, sparsity):

            self.sparsity = sparsity

            all_weights = t.cat([
                p.detach().abs().flatten()
                for p in self.weight_dict.values()
            ])

            self.threshold = t.quantile(
                all_weights,
                sparsity
            ).item()

            return self.threshold   
     
    def create_masks(self):

        self.mask_dict = {}

        for name, weight in self.weight_dict.items():

            mask = (
                weight.detach().abs()
                > self.threshold
            )

            self.mask_dict[name] = mask       

    @t.no_grad()
    def apply_masks(self):

        for name, weight in self.weight_dict.items():

            weight.mul_(

                self.mask_dict[name]

            )

    def prune(self, sparsity):

        self.compute_threshold(sparsity)

        self.create_masks()

        self.apply_masks()

        print(f"Threshold = {self.threshold:.6f}")

    def sparsity_report(self):

        total = 0
        zeros = 0

        for p in self.weight_dict.values():

            total += p.numel()

            zeros += (p == 0).sum().item()

        return {

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

                w = self.weight_dict[
                    f"experts.{expert}.trainable_nn.dense_layers.{layer}.linear.weight"
                ]

                total += w.numel()

                zeros += (w == 0).sum().item()

            rows.append({

                "Expert": expert,

                "Weights": total,

                "Zeros": zeros,

                "Sparsity (%)": 100 * zeros / total

            })

        return pd.DataFrame(rows)