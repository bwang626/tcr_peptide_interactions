# Graph Embeddings

Predicts TCR-peptide binding by representing each pair as a graph and training a relational graph attention network (R-GAT) end-to-end. The penultimate layer is exported as a fixed-size embedding for downstream models.

## Files

```
embeddings/graph/
├── model.py     # RGATLayer, TCRPeptideGNN, GraphEmbedder
├── train.py     # training script (run this)
└── README.md
```

## Graph construction

Each (TCR, peptide) pair is one graph:

- **Nodes** (max 30 + 15 = 45): one per amino-acid residue. Features are `aa_embedding + position_embedding + chain_type_embedding` (TCR vs. peptide).
- **Edges, three types:**
  - 0: TCR backbone — sequential `(i, i+1)` within the CDR3β chain
  - 1: peptide backbone — sequential within the epitope
  - 2: bipartite — every TCR residue ↔ every peptide residue (potential contact sites)

Padding positions are masked out of attention so variable-length sequences share a fixed adjacency.

## Architecture

```
AA + position + chain-type embeddings   (B, 45, 32)
                ↓
RGATLayer × 3  (per-edge-type projection, 4-head attention)
                ↓
masked mean-pool over TCR nodes  →  Linear → tcr_emb  (B, 64)
masked mean-pool over peptide    →  Linear → pep_emb  (B, 64)
                ↓
concat                                   (B, 128) ← pair embedding
                ↓
MLP (128 → 64 → 1) → binding logit
```

The attention layer uses a separate weight matrix per edge type (R-GAT style), so backbone messages and contact messages are learned independently.

## Training objective

Binary classification on positives from `processed/combined_trb_clean.csv` vs.
shuffled-peptide negatives. For each TCR in a batch, a peptide is sampled
uniformly from the dataset (rejecting any peptide actually paired with that
TCR). Loss is `BCEWithLogitsLoss`; early stopping on validation loss.

## Setup

```bash
conda activate tcr
pip install torch numpy pandas scikit-learn
```

No PyG / DGL dependency — the GNN is implemented directly on dense batched adjacency matrices since the per-pair graph is small (≤45 nodes).

## Generating data first

Same prerequisite as the autoencoder — you need `processed/combined_trb_clean.csv`:

```bash
python fetch_iedb.py
python fetch_vdjdb.py
python tsv_to_csv.py
python build_dataset.py
```

## Training

```bash
# Quick CPU test
python embeddings/graph/train.py --max_samples 5000 --epochs 10

# Full run
python embeddings/graph/train.py --epochs 30
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--data` | `processed/combined_trb_clean.csv` | Input data path |
| `--out` | `outputs/embeddings/graph` | Output directory |
| `--latent_dim` | `64` | Per-side embedding size (combined = 128) |
| `--hidden_dim` | `64` | GNN hidden dim |
| `--num_layers` | `3` | Number of R-GAT layers |
| `--num_heads` | `4` | Attention heads per layer |
| `--dropout` | `0.1` | Dropout |
| `--epochs` | `30` | Max training epochs |
| `--batch_size` | `128` | Mini-batch size |
| `--lr` | `0.001` | Learning rate |
| `--patience` | `5` | Early-stopping patience |
| `--val_frac` | `0.1` | Validation fraction |
| `--max_samples` | None | Cap rows for quick CPU runs |

## Outputs

```
outputs/embeddings/graph/
├── checkpoints/
│   └── graph_embedder.pt
├── tcr_embeddings.npy        shape (N, 64)
├── peptide_embeddings.npy    shape (N, 64)
├── combined_embeddings.npy   shape (N, 128)
├── embedding_index.csv
└── metrics.txt               validation AUROC vs shuffled negatives
```

## Using the embeddings / model

Same load pattern as the autoencoder:

```python
import numpy as np, pandas as pd

X = np.load("outputs/embeddings/graph/combined_embeddings.npy")    # (N, 128)
index = pd.read_csv("outputs/embeddings/graph/embedding_index.csv", index_col=0)
```

Score new pairs directly with the trained model:

```python
import pandas as pd
from embeddings.graph.model import GraphEmbedder

emb = GraphEmbedder.load("outputs/embeddings/graph/checkpoints/graph_embedder.pt")

new_pairs = pd.DataFrame({
    "cdr3":    ["CASSLAPGATNEKLFF", "CASSPGTASYNEKLFF"],
    "peptide": ["GILGFVFTL",        "NLVPMVATV"],
})
probs = emb.predict_proba(new_pairs)        # (2,) — binding probability
Z     = emb.transform(new_pairs)            # (2, 128) — pair embedding
```

## Why a graph?

Sequence-only encoders (one-hot, autoencoder) treat the TCR and peptide independently and rely on a downstream model to learn the cross-interaction. The graph formulation puts TCR-peptide *contacts* directly into the architecture: the bipartite edges let the GNN attend to the most relevant residue pairs, which is the actual biology of binding-site recognition. Compare downstream task performance against `one_hot/` and `autoencoder/` to measure the lift.
