# One-Hot Encoding Baseline

Encodes CDR3β TCR sequences and peptide epitope sequences into fixed-length vectors using one-hot encoding. No training required — this is a pure preprocessing step used as a **benchmark baseline** against learned embeddings (autoencoder, ESM2, graph, etc.).

## Files

```
embeddings/one_hot/
├── model.py           # OneHotEmbedder class + shared AA encoding utilities
├── embed.py           # script to generate and save embeddings
├── test_one_hot.py    # 16 unit tests
└── README.md          # this file
```

> **Note:** `model.py` is also the shared source of truth for AA encoding constants
> (`AA_TO_IDX`, `VOCAB_SIZE`, `encode_sequence`, `encode_sequences`, `decode_one_hot`)
> imported by other embedding modules like `autoencoder/`.

## Embedding dimensions

| Sequence | Max length | Vocab size | Flattened dim |
|---|---|---|---|
| TCR (CDR3β) | 30 | 22 | 660 |
| Peptide | 15 | 22 | 330 |
| Combined | — | — | 990 |

Compare this to the autoencoder's 128-dim combined vector — one-hot is much higher dimensional but requires no training and is fully interpretable.

## Setup

Make sure you are in the repo root and your conda environment is active:

```bash
conda activate tcr
pip install numpy pandas
```

## Generating embeddings

No training required — outputs are written immediately.

### Full dataset

```bash
python embeddings/one_hot/embed.py
```

### Quick test on a subset

```bash
python embeddings/one_hot/embed.py --max_samples 5000
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--data` | `processed/combined_trb_clean.csv` | Input data path |
| `--out` | `outputs/embeddings/one_hot` | Output directory |
| `--tcr_col` | `cdr3` | Column name for CDR3β sequences |
| `--peptide_col` | `peptide` | Column name for peptide epitope sequences |
| `--tcr_max_len` | `30` | Max TCR sequence length |
| `--peptide_max_len` | `15` | Max peptide sequence length |
| `--max_samples` | None | Cap rows for quick test |

## Outputs

```
outputs/embeddings/one_hot/
├── tcr_embeddings.npy        shape (N, 660)
├── peptide_embeddings.npy    shape (N, 330)
├── combined_embeddings.npy   shape (N, 990)
└── embedding_index.csv       maps each row back to (cdr3, peptide)
```

## Using embeddings as model input

Same interface as the autoencoder — just point to a different `.npy` file:

```python
import numpy as np
import pandas as pd

X = np.load("outputs/embeddings/one_hot/combined_embeddings.npy")  # (N, 990)
index = pd.read_csv("outputs/embeddings/one_hot/embedding_index.csv", index_col=0)
```

Load TCR and peptide separately if your model needs them independently:

```python
tcr_z     = np.load("outputs/embeddings/one_hot/tcr_embeddings.npy")     # (N, 660)
peptide_z = np.load("outputs/embeddings/one_hot/peptide_embeddings.npy") # (N, 330)
```

Use the embedder directly in code without saving to disk:

```python
import pandas as pd
from embeddings.one_hot.model import OneHotEmbedder

embedder = OneHotEmbedder()
X = embedder.transform(df)   # df has columns: cdr3, peptide  →  (N, 990)
```

## Running tests

```bash
python -m pytest embeddings/one_hot/test_one_hot.py -v
```

## Why use this as a baseline?

- **No training required**: immediate results, nothing to tune
- **Fully deterministic**: same input always gives same output
- **Interpretable**: each dimension directly corresponds to a specific amino acid at a specific position
- **Sanity check**: if learned embeddings (autoencoder, ESM2, etc.) don't outperform one-hot on downstream tasks, something is wrong with the learned representations