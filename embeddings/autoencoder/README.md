# Autoencoder Embeddings

Encodes CDR3β TCR sequences and peptide epitope sequences into fixed-length dense latent vectors using a sequence-to-latent autoencoder. Supports both a plain autoencoder (AE) and a variational autoencoder (VAE).

## Files

```
embeddings/autoencoder/
├── model.py              # SequenceAutoencoder and AutoencoderEmbedder classes
├── train.py              # training script (run this)
├── test_autoencoder.py   # 20 unit tests
└── README.md             # this file
```

## Setup

Make sure you are in the repo root and your conda environment is active:

```bash
conda activate tcr
pip install torch numpy pandas scikit-learn pytest
```

## Generating the data first

Before training, you need `processed/combined_trb_clean.csv`. From the repo root:

```bash
python fetch_iedb.py       # downloads IEDB raw data → iedb_data/
python fetch_vdjdb.py      # downloads VDJdb release → vdjdb_data/
python tsv_to_csv.py       # converts VDJdb .txt → vdjdb_csv/
python build_dataset.py    # builds cleaned datasets → processed/
```

> **Note:** Also manually download McPAS-TCR from https://friedmanlab.weizmann.ac.il/McPAS-TCR/ and place it at `./McPAS-TCR.csv` in the repo root before running `build_dataset.py`.

---

## Training

All commands are run from the **repo root**.

### Quick CPU test (recommended first run)

```bash
python embeddings/autoencoder/train.py --max_samples 5000 --epochs 20
```

### Plain AE only

```bash
python embeddings/autoencoder/train.py --epochs 100
```

### VAE only

```bash
python embeddings/autoencoder/train.py --vae --epochs 100
```

### Train both and compare (recommended for full runs)

```bash
python embeddings/autoencoder/train.py --compare --epochs 100
```

This trains both models sequentially and prints a side-by-side summary of val losses at the end, also saved to `outputs/embeddings/autoencoder/comparison_summary.txt`.

### All options

| Flag | Default | Description |
|---|---|---|
| `--data` | `processed/combined_trb_clean.csv` | Input data path |
| `--out` | `outputs/embeddings/autoencoder` | Base output directory |
| `--latent_dim` | `64` | Size of the embedding vector |
| `--epochs` | `100` | Max training epochs |
| `--batch_size` | `256` | Mini-batch size |
| `--lr` | `0.001` | Learning rate |
| `--vae` | off | Use variational autoencoder |
| `--kl_weight` | `0.01` | KL divergence weight (VAE only) |
| `--patience` | `10` | Early stopping patience (epochs) |
| `--val_frac` | `0.1` | Fraction of sequences held out for validation |
| `--max_samples` | None | Cap dataset rows for quick CPU runs |
| `--compare` | off | Train both AE and VAE, print comparison |

---

## Outputs

All outputs are written to `outputs/embeddings/autoencoder/` and are gitignored.

```
outputs/embeddings/autoencoder/
├── plain_ae/
│   ├── checkpoints/
│   │   ├── tcr_ae.pt              # trained TCR autoencoder weights
│   │   └── peptide_ae.pt          # trained peptide autoencoder weights
│   ├── tcr_embeddings.npy         # shape (N, latent_dim)
│   ├── peptide_embeddings.npy     # shape (N, latent_dim)
│   ├── combined_embeddings.npy    # shape (N, 2 * latent_dim)
│   └── embedding_index.csv        # maps each row back to (cdr3, peptide)
├── vae/
│   └── ...                        # same structure as plain_ae/
└── comparison_summary.txt         # only written when --compare is used
```

---

## Using embeddings as model input

Load the embeddings and attach labels from the original dataset:

```python
import numpy as np
import pandas as pd

# Load embeddings — one row per TCR-peptide pair
X = np.load("outputs/embeddings/autoencoder/vae/combined_embeddings.npy")  # (N, 128)

# Load the row index to map embeddings back to sequences
index = pd.read_csv("outputs/embeddings/autoencoder/vae/embedding_index.csv", index_col=0)

# Load original data to get labels or other metadata
df = pd.read_csv("processed/combined_trb_clean.csv", low_memory=False)
```

`X` is ready to pass directly into any sklearn or PyTorch model, for example:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

You can also load the TCR and peptide embeddings separately if your model treats them differently (e.g. cross-attention between TCR and peptide):

```python
tcr_z    = np.load("outputs/embeddings/autoencoder/vae/tcr_embeddings.npy")     # (N, 64)
peptide_z = np.load("outputs/embeddings/autoencoder/vae/peptide_embeddings.npy") # (N, 64)
```

### Loading a saved model to re-embed new sequences

```python
from embeddings.autoencoder.model import SequenceAutoencoder

tcr_ae = SequenceAutoencoder.load("outputs/embeddings/autoencoder/vae/checkpoints/tcr_ae.pt")
pep_ae = SequenceAutoencoder.load("outputs/embeddings/autoencoder/vae/checkpoints/peptide_ae.pt")

new_tcr_seqs     = ["CASSLAPGATNEKLFF", "CASSPGTASYNEKLFF"]
new_peptide_seqs = ["GILGFVFTL", "NLVPMVATV"]

tcr_z    = tcr_ae.transform(new_tcr_seqs)       # (2, 64)
peptide_z = pep_ae.transform(new_peptide_seqs)  # (2, 64)
```

---

## Running test cases

```bash
# All 20 tests
python -m pytest embeddings/autoencoder/test_autoencoder.py -v

# VAE tests only
python -m pytest embeddings/autoencoder/test_autoencoder.py -v -k "vae"

# Plain AE tests only
python -m pytest embeddings/autoencoder/test_autoencoder.py -v -k "not vae"
```

---

## Architecture summary

Both models share the same encoder-decoder structure. The only difference is the loss function.

```
Input: one-hot sequence  (N, seq_len, 22)
         ↓
Conv1D × 2               (N, seq_len, 64)
         ↓
Bidirectional GRU        (N, 2 * gru_hidden)
         ↓
Linear → mu, logvar      (N, latent_dim)
         ↓  [VAE: z ~ N(mu, sigma)]  [Plain AE: z = mu]
         z               (N, latent_dim)
         ↓
Linear → GRU × 2         (N, seq_len, gru_hidden)
         ↓
Conv1D × 2 → logits      (N, seq_len, 22)
```

**Plain AE loss:** cross-entropy reconstruction only

**VAE loss:** cross-entropy + `kl_weight × KL(N(mu,sigma) || N(0,1))`

The KL term regularises the latent space to be smoother and more continuous, which can help generalisation in downstream models.