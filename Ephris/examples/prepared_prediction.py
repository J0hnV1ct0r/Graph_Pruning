"""Reuse one prepared graph for repeated prediction without reloading weights."""

import torch

from ephris import EphrisClassifier

x = torch.tensor([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
edge_index = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
context_indices = torch.tensor([0, 1])
query_indices = torch.tensor([2, 3])
context_labels = torch.tensor([0, 1])

model = EphrisClassifier(random_state=42, device="auto")
model.prepare(
    x,
    edge_index,
    context_indices=context_indices,
    context_labels=context_labels,
)
for _ in range(2):
    # Each iteration reruns inference and reuses the prepared graph and weights.
    log_probabilities = model.predict_prepared_log_proba()
    query_log_probabilities = log_probabilities[query_indices]
    query_labels = model.classes_[query_log_probabilities.argmax(dim=1)]
    print("Predicted labels:", query_labels)
    print("Query probabilities:", query_log_probabilities.exp())

# Call prepare() again when graph inputs or context labels change.
# Reuse the returned tensors directly when another prediction is unnecessary.
