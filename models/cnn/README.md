# CNN TCR-Peptide Binding Predictor

Dual-branch 1D-CNN classifier that consumes pre-computed embeddings from `embeddings/`. Each branch convolves over its own input (TCR or peptide), max-pools, and the pooled features (plus optional V/J + MHC metadata) feed a small MLP head.

## Supported embeddings

The `--embedding` flag selects which pre-computed embedding to read from `outputs/embeddings/<embedding>/`:

| `--embedding`   | Per-residue? | Branch input shape           |
|-----------------|--------------|------------------------------|
| `one_hot`       | yes          | (C=22, L=30) TCR / (22, 15) peptide |
| `autoencoder`   | no (vector)  | (C=1, L=latent_dim) per side |
| `esm`           | no (vector)  | (C=1, L=hidden_dim) per side |

For `one_hot`, the flat `(N, L*V)` array is reshaped to `(N, V, L)` so the CNN convolves over residue positions with vocab as channels. For pooled vector embeddings (autoencoder, ESM) we just unsqueeze to `(N, 1, D)` and convolve along the feature axis.

## Optional V/J + MHC metadata

`--use_feature_augment` concatenates a metadata vector after CNN pooling (same encoders as `embeddings/feature_augment/`):

| `--feature_augment_type` | Encoder                  | Dim (combined dataset) |
|--------------------------|--------------------------|------------------------|
| `one_hot` (default)      | `OneHotFeatureAugmenter` | 1 + |V| + |J|          |
| `cat_ae`                 | `CatAEFeatureAugmenter`  | 1 + 2·latent_dim       |

So the six configurations the CNN supports out of the box are:

- `one_hot`
- `autoencoder`
- `esm`
- `one_hot      + feature_augment` (`--use_feature_augment`)
- `autoencoder  + feature_augment`
- `esm          + feature_augment`

## Expected directory layout

```
data/splits/train.csv          (cdr3, peptide, …, label)
data/splits/val.csv            same schema
outputs/embeddings/<embedding>/train/
    tcr_embeddings.npy         (N, …)
    peptide_embeddings.npy     (N, …)
    embedding_index.csv        cdr3, peptide order check (optional)
outputs/embeddings/<embedding>/val/
    …
```

The training script verifies that embedding rows align with the split CSV row-for-row on `(cdr3, peptide)`.

## Usage

```bash
# from repo root

# one_hot only
python -m models.cnn.train --embedding one_hot

# autoencoder + V/J + MHC (one-hot encoded)
python -m models.cnn.train --embedding autoencoder --use_feature_augment

# ESM + V/J + MHC (cat-AE encoded)
python -m models.cnn.train --embedding esm \
    --use_feature_augment --feature_augment_type cat_ae

# wider conv branches
python -m models.cnn.train --embedding one_hot \
    --conv_channels 128 --branch_out_dim 128
```

**Key arguments:**

| Argument | Default | Notes |
|---|---|---|
| `--embedding` | required | `one_hot` / `autoencoder` / `esm` |
| `--embedding_dir` | `outputs/embeddings/<embedding>` | Override embedding root |
| `--train_csv` | `data/splits/train.csv` | Must contain `cdr3`, `peptide`, `label` |
| `--val_csv` | `data/splits/val.csv` | Same schema |
| `--use_feature_augment` | off | Append V/J + MHC after CNN pooling |
| `--feature_augment_type` | `one_hot` | `one_hot` (sparse) or `cat_ae` (dense) |
| `--conv_channels` | 64 | Filters per conv layer |
| `--kernel_size` | 5 | Conv kernel |
| `--branch_out_dim` | 64 | Pooled vector size per branch |
| `--epochs` | 30 | Max; early stop on val AUROC |
| `--patience` | 5 | Early-stop patience |

## Outputs

```
outputs/models/cnn/<embedding>[_aug-<type>]/
    checkpoints/checkpoint.pt    state_dict + config + args
    metrics.txt                  per-epoch loss / AUROC, best AUC, n_params
```

## Loading a checkpoint

```python
import torch
from models.cnn.model import EmbeddingCNN

ckpt  = torch.load("outputs/models/cnn/one_hot/checkpoints/checkpoint.pt")
model = EmbeddingCNN(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
```
