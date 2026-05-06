# Feature Augmentation

Appends categorical metadata features to sequence embeddings before passing to the downstream model.

Sequence encoders (one-hot, autoencoder, graph, ESM2) only see CDR3 and peptide sequences. This module adds V/J gene and MHC class information, which encodes structural context that isn't captured in the CDR3 alone.

## Features encoded

| Feature | Why |
|---|---|
| `mhc_class` | MHCI vs MHCII determines which T cell type is involved (CD8+ vs CD4+) |
| `v_gene` | Encodes CDR1/CDR2 loops that contact the MHC helices |
| `j_gene` | Influences the CDR3 junction and contributes to binding specificity |

`mhc_a` (HLA allele) is excluded — the dataset is dominated by HLA-A\*02:01, and peptide sequences already implicitly encode HLA anchor preferences.

## Two encoding methods

### `OneHotFeatureAugmenter` — sparse baseline

Encodes v_gene and j_gene as raw one-hot vectors. Fast, no training required.

- Feature dim: **741** (1 + 588 + 152 on full dataset)
- v_gene and j_gene treated as fully independent — no similarity structure

### `CatAEFeatureAugmenter` — learned compression (PepTCR-style)

Trains a small MLP autoencoder on each categorical variable. The encoder bottleneck replaces the one-hot vector with a dense learned representation. Similar V genes (e.g. TRBV19\*01 and TRBV19\*02) compress to similar latent vectors.

- Feature dim: **65** (1 + 32 + 32, with default `latent_dim=32`)
- Follows the categorical autoencoding approach from PepTCR-Net (Le et al. 2025)

## Usage

```python
from embeddings.feature_augment import OneHotFeatureAugmenter, CatAEFeatureAugmenter

# fit on training split only — avoids leaking val/test vocab
aug = OneHotFeatureAugmenter()          # or CatAEFeatureAugmenter(latent_dim=32)
aug.fit(train_df)

# augment any split using the same fitted augmenter
train_aug = aug.augment(train_embeddings, train_df)   # (N, d + feature_dim)
val_aug   = aug.augment(val_embeddings,   val_df)

# save and reload to keep vocabs consistent across runs
aug.save("outputs/feature_augment/one_hot.pkl")       # one_hot: single file
aug.save("outputs/feature_augment/cat_ae/")           # cat_ae:  directory

aug = OneHotFeatureAugmenter.load("outputs/feature_augment/one_hot.pkl")
aug = CatAEFeatureAugmenter.load("outputs/feature_augment/cat_ae/")
```

Swapping methods is a one-line import change — both classes expose the same `fit / augment / save / load` interface.

## CLI

```bash
# one-hot
python -m embeddings.feature_augment.one_hot \
    --embeddings outputs/embeddings/one_hot/combined_embeddings.npy \
    --index      outputs/embeddings/one_hot/embedding_index.csv \
    --data       processed/combined_trb_clean.csv \
    --out        outputs/embeddings/one_hot/combined_augmented.npy \
    --augmenter_out outputs/feature_augment/one_hot.pkl

# categorical AE
python -m embeddings.feature_augment.autoencoder \
    --embeddings outputs/embeddings/one_hot/combined_embeddings.npy \
    --index      outputs/embeddings/one_hot/embedding_index.csv \
    --data       processed/combined_trb_clean.csv \
    --out        outputs/embeddings/one_hot/combined_augmented_cat_ae.npy \
    --augmenter_out outputs/feature_augment/cat_ae/ \
    --latent_dim 32 --epochs 100
```

## Where this fits in the pipeline

```
sequences → [one_hot | autoencoder | graph | ESM2] → (N, d)
                                                          ↓
                              + feature_augment → (N, d + feature_dim)
                                                          ↓
                                                       model
```
