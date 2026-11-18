"""Predict labels for query nodes using a pretrained Ephris checkpoint."""

import torch

from ephris import EphrisClassifier

x = torch.tensor([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
edge_index = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
context_indices = torch.tensor([0, 1])
query_indices = torch.tensor([2, 3])
context_labels = torch.tensor([0, 1])

model = EphrisClassifier(random_state=42, device="auto")
labels = model.predict(
    x,
    edge_index,
    context_indices=context_indices,
    context_labels=context_labels,
)
# Predictions cover every node; select the query nodes for the application.
print("Predicted labels:", labels[query_indices])

# Use predict_proba() instead when probabilities are needed. If both outputs
# are needed, derive labels from that result using classes_ (see the README).
