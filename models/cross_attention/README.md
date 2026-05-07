# Cross-Attention TCR-Peptide Binding Predictor

Implements the pairwise cross-attention approach: every TCR residue is compared against every peptide residue through a learned attention mechanism, capturing position-specific interaction signals rather than treating the sequences as bags of residues.

## Motivation

Sequence encoders (one-hot, autoencoder, ESM) embed TCR and peptide independently, then concatenate. This misses the key structural reality: TCR binding is driven by specific residue-residue contacts between the CDR3 loop and the peptide. Cross-attention makes these contacts explicit — the attention weight matrix at each layer is exactly the (L_tcr × L_pep) pairwise interaction score.

## Input: raw sequences, not pre-computed embeddings

This model takes **raw amino acid sequences** as input — it is not a downstream classifier on top of `embeddings/` outputs. It learns its own AA embeddings from scratch and requires per-residue representations `(B, L, D)` to compute the pairwise matrix.

The pre-computed embeddings in `embeddings/` (one-hot, autoencoder, graph, ESM) are all pooled into a single fixed vector per sequence — the positional structure needed for cross-attention is gone. They feed into separate MLP/RF classifiers, not this model.

The one future exception: ESM produces per-residue hidden states before pooling. If `generate_embeddings.py` is modified to save those, they could replace the learned AA embedding layer here. That would require changes to the ESM pipeline (Jeff's code).

## Architecture

```
CDR3 sequence          Peptide sequence
     ↓                       ↓
AA embed + sinusoidal PE     AA embed + sinusoidal PE
     ↓                       ↓
┌────────────────────────────────────────┐
│  CrossAttentionBlock × n_layers        │
│                                        │
│  TCR  → self-attn(TCR)                 │
│       → cross-attn(Q=TCR, K/V=Pep)    │  ← (L_tcr × L_pep) interaction matrix
│       → feed-forward                   │
│                                        │
│  Pep  → self-attn(Pep)                 │
│       → cross-attn(Q=Pep, K/V=TCR)    │  ← (L_pep × L_tcr) interaction matrix
│       → feed-forward                   │
└────────────────────────────────────────┘
     ↓                       ↓
masked mean-pool         masked mean-pool
     ↓                       ↓
     └──────────┬────────────┘
           concat (+ optional metadata)
                ↓
           MLP → logit
```

**Key design choices:**
- Self-attention before cross-attention: each sequence builds internal context before attending to the other
- Shared weights across TCR and peptide paths: symmetric inductive bias, fewer parameters
- Sinusoidal positional encoding: position in CDR3/peptide is meaningful (CDR3 tip residues dominate binding)
- Masked mean-pooling: padding positions are excluded from the pooled representation

## Optional metadata

V/J gene and MHC class can be appended after pooling via `--use_metadata`. Two encoders are supported — the same ones from `embeddings/feature_augment/`:

| `--metadata_type` | Encoder | Dim | Notes |
|---|---|---|---|
| `one_hot` (default) | `OneHotFeatureAugmenter` | 164 | Sparse, interpretable, no training needed |
| `cat_ae` | `CatAEFeatureAugmenter` | 65 | Dense learned latents; similar genes → similar vectors |

The model only sees a `(B, meta_dim)` float tensor — it is agnostic to which encoder produced it. `meta_dim` is set automatically from whichever augmenter is used.

| Config | Classifier input dim |
|---|---|
| No metadata | 2 × d_model = 128 (defaults) |
| `--use_metadata --metadata_type one_hot` | 128 + 164 = 292 |
| `--use_metadata --metadata_type cat_ae` | 128 + 65 = 193 |

## Usage

```python
from models.cross_attention.model import CrossAttentionTCRPep, collate_sequences

model = CrossAttentionTCRPep(d_model=64, n_heads=4, n_layers=2)

# programmatic inference
probs = model.predict_proba(
    tcr_seqs=["CASSIVGGNEQFF", "CASSMRSTGELFF"],
    pep_seqs=["GILGFVFTL",     "NLVPMVATV"],
)

# load from checkpoint
import torch
ckpt = torch.load("outputs/models/cross_attention/checkpoints/checkpoint.pt")
model = CrossAttentionTCRPep(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
```

## Training

Data splitting and negative generation are handled upstream by a separate module. `train.py` expects pre-split, pre-labeled CSVs with columns `cdr3`, `peptide`, `label` (1 = binding, 0 = non-binding), and optionally `v_gene`, `j_gene`, `mhc_class` when using metadata.

```bash
# from repo root
python -m models.cross_attention.train \
    --train data/train.csv --val data/val.csv

python -m models.cross_attention.train \
    --train data/train.csv --val data/val.csv \
    --use_metadata                              # one-hot V/J + MHC (default)

python -m models.cross_attention.train \
    --train data/train.csv --val data/val.csv \
    --use_metadata --metadata_type cat_ae       # dense AE latents

python -m models.cross_attention.train \
    --train data/train.csv --val data/val.csv \
    --d_model 128 --n_layers 3                  # larger model
```

**Key arguments:**

| Argument | Default | Notes |
|---|---|---|
| `--train` | required | CSV with cdr3, peptide, label (+ optional metadata cols) |
| `--val` | required | Same format as --train |
| `--d_model` | 64 | Residue embedding dim; n_heads must divide it |
| `--n_heads` | 4 | Attention heads per block |
| `--n_layers` | 2 | Number of CrossAttentionBlocks |
| `--epochs` | 30 | Max epochs; early stopping on val AUROC |
| `--patience` | 5 | Early stopping patience |
| `--use_metadata` | off | Append V/J + MHC metadata after pooling |
| `--metadata_type` | `one_hot` | `one_hot` (164-dim) or `cat_ae` (65-dim) |

## Outputs

```
outputs/models/cross_attention/
    checkpoints/checkpoint.pt    model weights + config dict
    metrics.txt                  per-epoch train loss / val loss / val AUROC
```

## Where this fits in the pipeline

```
sequences → cross-attention (end-to-end) → binding probability
```

Unlike the other embedding methods, this model is trained directly on binding labels rather than producing embeddings for a separate classifier. It can optionally consume the same V/J + MHC metadata features used by the other methods.
