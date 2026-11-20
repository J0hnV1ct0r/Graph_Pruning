# Inference details

## Graph inputs and outputs

`EphrisClassifier` performs transductive node classification. All nodes' features
and edges are available during inference; only context labels are visible.
No `fit()` call is required: pass the graph and context to a prediction method,
or call `prepare()` explicitly to reuse preparation.

| Argument | Meaning |
| --- | --- |
| `x` | Numeric features, shape `[N, F]`, as a NumPy array or PyTorch tensor |
| `edge_index` | Integer node IDs in `[0, N)`, shape `[2, E]` |
| `context_indices` | Unique integer node indices `[K]` in `[0, N)` |
| `context_labels` | Integer labels `[K]`, paired with `context_indices` in the supplied order |

Pass context arguments by name. Indices need not be sorted; each label belongs
to the index at the same position. Full-graph labels are not accepted unless
every node is explicitly included. For an existing Boolean mask, use
`mask.nonzero(as_tuple=True)[0]` (PyTorch) or `np.flatnonzero(mask)` (NumPy).

At least two distinct classes must be present in the context. The native head
supports up to ten classes; larger tasks use deterministic error-correcting output
codes (ECOC). Edges are unweighted and treated as undirected: Ephris adds reverse
edges and missing self-loops internally. Edge attributes are not supported. The
predictor does not train or update the pretrained weights.

All prediction methods return results for **all nodes**, including context nodes.
Select query rows with `[query_indices]` or a Boolean query mask.

| Method | CPU output | Use |
| --- | --- | --- |
| `predict()` | Original integer labels `[N]` | Node classification |
| `predict_proba()` | FP32 probabilities `[N, C]` | Confidence scores and probability-based metrics |
| `predict_log_proba()` | FP32 natural-log probabilities `[N, C]` | Negative log-likelihood and calculations in log space |

Each log-probability is `log(p(class | node))`; exponentiating it recovers the
probability. This lets callers use log values directly for loss calculations
and combine probabilities in log space.
`C` is the number of distinct context labels.
`classes_` is a sorted CPU integer tensor defining
the probability-column order. Labels need not be consecutive: `[2, 5, 9]` are
encoded internally as `[0, 1, 2]`. Only observed classes can be predicted.
Probability column `j` corresponds to `classes_[j]`, which may differ from label
`j`. When a downstream evaluator expects a fixed class order, align the columns
using `classes_` and fill absent class columns with zero there. For example,
`classes_ = [0, 2]` means Ephris columns `[p0, p2]` become `[p0, 0, p2]` in an
evaluator ordered as `[0, 1, 2]`.
All three methods accept the same graph and context inputs. When both labels and
probabilities are needed, call `predict_proba()` once and obtain labels with
`model.classes_[probabilities.argmax(dim=1)]`. Calling `predict()` afterward
would repeat preparation and inference.
A successful preparation replaces `classes_`; a failed preparation clears it.

See [classification.py](../examples/classification.py) for a standalone example.

## Checkpoints

The [nums-ai/ephris](https://huggingface.co/nums-ai/ephris) model repository
hosts the weights. Downloads do not require a Hugging Face account or access token.

The first prediction downloads `config.json` and the checkpoint at a fixed revision.
The configuration specifies the checkpoint file, format, model settings and SHA256;
the downloaded checkpoint is verified before loading. Subsequent calls reuse the
Hugging Face cache. Construction only stores options; downloading and loading
happen on the first prediction or explicit `prepare()`. After successful
preparation, `checkpoint_` records the resolved file path. Weights are not included
in Git, wheels or source archives.

To use a previously downloaded file without accessing the Hub:

```python
from ephris import EphrisClassifier

model = EphrisClassifier(checkpoint="/path/to/ephris.pt", device="cpu")
```

The [checkpoint manifest](https://huggingface.co/nums-ai/ephris/blob/94aa9caf6cf7fb7c44109e15e0de8a6a4b6b14a0/manifest.json) records the checkpoint and individual tensor hashes
at the same pinned revision as the weights. The [model license](https://huggingface.co/nums-ai/ephris/blob/94aa9caf6cf7fb7c44109e15e0de8a6a4b6b14a0/LICENSE) is separate
from the inference code's Apache-2.0 license.

`random_state=42` is a constructor option controlling the ECOC codebook; use a
nonnegative integer. Changed options apply to the next prediction with graph
inputs or an explicit `prepare()`, not an already prepared task.

## Platform support and precision

`device="auto"` selects CUDA when available, otherwise CPU. Explicit `"cpu"` and
CUDA device strings such as `"cuda:0"` are supported. Platform support covers
standard CPython 3.11–3.14 on Linux; free-threaded builds and Python 3.15 are
not supported. CI checks CPU installation and tests across supported Python
versions. GPU measurements use Python 3.11, CUDA 13.0 and NVIDIA H200.
Supported CUDA GPUs use BF16 autocast; CPU and unsupported GPUs use FP32.

Feature and class permutations use a fixed local generator. The constructor’s `random_state` controls
ECOC, not a prediction ensemble. GPU reductions and adaptive chunk sizes can change
floating-point results. Fixed seeds do not guarantee identical probabilities
across runs, devices, memory budgets or library versions. Normalization and
probability aggregation retain FP32 stabilization; CUDA may use TF32 for FP32
matrix multiplication.

## Repeated prediction

`prepare()` loads weights and prepares the graph's features and context once.
`predict_prepared_log_proba()` then predicts on that same graph without repeating
preparation. It does not accept new query nodes. Call `prepare()` again whenever
features, edges, context indices, labels or the class count change.

```python
model.prepare(
    x, edge_index, context_indices=context_indices,
    context_labels=context_labels,
)
log_probabilities = model.predict_prepared_log_proba()
probabilities = log_probabilities.exp()
labels = model.classes_[log_probabilities.argmax(dim=1)]
print(labels[query_indices])
print(probabilities[query_indices])
```

This reuses prepared state, not cached output probabilities. A failed preparation
invalidates the previous task. See [prepared_prediction.py](../examples/prepared_prediction.py).
If the inputs and outputs are unchanged, reuse the returned tensors directly.

## Notebook comparison

From a source checkout:

```bash
python -m pip install '.[notebook]'
python -m jupyterlab examples/inference_demo.ipynb
```

Use the same environment's kernel. The demo compares Ephris with two-layer
GCN/GAT baselines trained for 200 epochs on the same context labels. It runs
Cornell, Reed98, Cora and the full CityNetwork Paris road graph. Binary tasks
are scored with ROC-AUC and multiclass tasks with accuracy, using one seeded
50% context / 50% query split. Query labels enter scoring only.

The saved outputs contain the executed performance table and chart. The notebook
exports full-precision scores and PNG/PDF charts to `outputs/notebook/`.
A CUDA GPU is recommended for the full comparison. The baseline training code
stays in the notebook. [Saved demo results](benchmarks/README.md).

See the README's [installation instructions](../README.md#installation) for
environment setup.
