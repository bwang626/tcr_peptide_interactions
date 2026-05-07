# ESM Embeddings

Encodes full-length mature TRBβ amino acid sequences using [ESMplusplus (ESMplusplus_large)](https://huggingface.co/Synthyra/ESMplusplus_large), a 36-layer protein language model with 480M parameters. Unlike the other embedding methods which encode only the CDR3, ESM operates on the **complete V–CDR3–J–C sequence** and produces two complementary embeddings per TCR:

- **`full_emb`** — pooled over the entire mature TRBβ chain (V through constant region).
- **`cdr3_emb`** — pooled over the CDR3 residues only, located by `cdr3_start`/`cdr3_end` indices in the sequence.

## Files

```
embeddings/esm/
├── stitchr_utils.py        # wraps stitchr: (V gene, J gene, CDR3) → full-length AA sequence
├── prepare_dataset.py      # builds processed/esm_trb_dataset.csv from combined_trb_clean.csv
├── generate_embeddings.py  # runs ESMplusplus, writes embeddings to outputs/embeddings/esm/
└── README.md               # this file
```

## Prerequisites

### 1 — Install dependencies

```bash
conda activate tcr
pip install torch transformers stitchr tidytcells numpy pandas
```

ESMplusplus weights (~1.8 GB) are downloaded automatically from HuggingFace on first run.

### 2 — Build the cleaned dataset

```bash
python fetch_iedb.py
python fetch_vdjdb.py
python tsv_to_csv.py
python build_dataset.py    # → processed/combined_trb_clean.csv
```

> Also manually download McPAS-TCR from https://friedmanlab.weizmann.ac.il/McPAS-TCR/ and place it at `./McPAS-TCR.csv` before running `build_dataset.py`.

### 3 — Prepare the ESM-specific dataset

`prepare_dataset.py` uses [stitchr](https://github.com/JamieHeather/stitchr) to reconstruct full-length TRBβ amino acid sequences from each row's V gene, J gene, and CDR3, and records the CDR3 start/end positions within that sequence.

```bash
python embeddings/esm/prepare_dataset.py    # → processed/esm_trb_dataset.csv
```

This step takes several minutes for the full dataset (~82 k rows). Progress is printed every 5 000 sequences.

---

## Generating embeddings

All commands are run from the **repo root**.

### Quick test (recommended first run — CPU-safe)

```bash
python embeddings/esm/generate_embeddings.py \
    --data processed/esm_test_dataset.csv \
    --out  outputs/embeddings/esm_test \
    --limit 100
```

`processed/esm_test_dataset.csv` is a pre-built 100-row sample spanning the full sequence-length range, included in the repo for smoke-testing without GPU access.

### Full dataset (GPU recommended)

```bash
python embeddings/esm/generate_embeddings.py
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--data` | `processed/esm_trb_dataset.csv` | Input dataset (output of `prepare_dataset.py`) |
| `--out` | `outputs/embeddings/esm` | Output directory |
| `--model` | `Synthyra/ESMplusplus_large` | HuggingFace model ID |
| `--layer` | _(last layer)_ | 1-indexed transformer layer to extract (1–36); default is the final normed output |
| `--pooling` | `mean` | Pooling over the sequence dimension — `mean` or `max` |
| `--batch_size` | `32` | Sequences per forward pass |
| `--limit` | None | Cap sequences processed, e.g. `--limit 100` for a quick test |

---

## Outputs

All outputs are written to `outputs/embeddings/esm/` and are gitignored.

```
outputs/embeddings/esm/
├── full_embeddings.npy     shape (N, 1280)   full mature TRBβ sequence embedding
├── cdr3_embeddings.npy     shape (N, 1280)   CDR3-region embedding (NaN where span is missing)
└── embedding_index.csv     maps each row back to sequence metadata
```

The integer index in `embedding_index.csv` matches the `.npy` row order exactly. Columns include `full_aa`, `cdr3`, `cdr3_start`, `cdr3_end`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, `mhc_class`, and `source`.

The hidden dimension D = 1280 for `ESMplusplus_large`.

---

## Using the embeddings

```python
import numpy as np
import pandas as pd

# Load embeddings
full_emb = np.load("outputs/embeddings/esm/full_embeddings.npy")  # (N, 1280)
cdr3_emb = np.load("outputs/embeddings/esm/cdr3_embeddings.npy")  # (N, 1280)

# Load row index to map back to sequences and metadata
index = pd.read_csv("outputs/embeddings/esm/embedding_index.csv", index_col=0)
```

Use `full_emb` when your model should reason about the full TCR context. Use `cdr3_emb` when you want to focus on the binding-relevant hypervariable region, consistent with what the autoencoder embeds. You can also concatenate:

```python
X = np.concatenate([full_emb, cdr3_emb], axis=1)   # (N, 2560)
```

Rows with a missing CDR3 span (rare stitching edge cases) have `NaN` in `cdr3_emb`. Filter them before training:

```python
valid = ~np.isnan(cdr3_emb[:, 0])
X = cdr3_emb[valid]
index_valid = index[valid]
```

---

## Batching strategy

Sequences are sorted ascending by length before being chunked into batches. This keeps same-length sequences together, minimising the padding added to the attention matrix and significantly reducing compute on CPU. On GPU the benefit is smaller but still meaningful for long-sequence batches.

---

## Data quality filters

Two filters are applied upstream to prevent stop codons from reaching ESMplusplus (which does not expect them):

1. **`build_dataset.py` — functional gene filter:** `tidytcells.tr.standardize(..., enforce_functional=True)` rejects IMGT genes annotated as ORF or pseudogene. These gene reference sequences can contain in-frame stop codons. This removes ~3 % of rows (≈2 579 / 81 977).

2. **`stitchr_utils.py` — post-translation filter:** after stitchr translates the assembled nucleotide sequence, any sequence containing `*` (stop codon) is discarded and treated as a stitching failure. This catches residual cases that slip past the gene filter.

---

## Architecture context

ESMplusplus_large is a 36-block transformer trained on UniRef90 with a masked-language-modelling objective. Its hidden states encode evolutionary and structural information about each residue in context.

```
Input: tokenised AA sequence  (B, L)
              ↓
Token embedding               (B, L, 1280)
              ↓
Transformer block × 36        (B, L, 1280)   ← hidden_states[1..36]
              ↓
Final LayerNorm               (B, L, 1280)   ← hidden_states[36] [default]
              ↓
Pool over residues            (B, 1280)      ← full_emb  or  cdr3_emb
```

By default, embeddings are extracted from `hidden_states[36]` (the final LayerNorm output, equivalent to `last_hidden_state`). Earlier layers capture more local sequence patterns; later layers capture higher-level structural context.

## Why use ESM over the other embeddings?

| | One-hot | Autoencoder | Graph | ESM |
|---|---|---|---|---|
| Training required | No | Yes | Yes | No (pretrained) |
| Encodes full chain | No (CDR3 only) | No (CDR3 only) | No (CDR3 only) | **Yes** |
| Residue context | None | Local (Conv+GRU) | Contact graph | Global (transformer) |
| Embedding dim | 660 / 330 | 64 | 64 | **1280** |
| Evolutionary info | No | No | No | **Yes** |

ESM embeddings are the most information-rich but also the most expensive to compute. The CDR3-only `cdr3_emb` is the most direct comparison point against the other methods.
