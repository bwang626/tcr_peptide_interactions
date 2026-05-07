# Models

End-to-end binding predictors that take pre-split labeled CSVs (`cdr3`, `peptide`, `label`) as input. Data splitting and negative generation are handled upstream by `data_split/` and `build_negatives.py`.

## Feature augmentation (V/J gene + MHC class)

All models can optionally consume V/J gene and MHC class as metadata. This encoding happens **inside the training script**, not at the `data_split` stage. Both encoders must be fit on training data only to avoid leakage, and keeping it here makes that guarantee explicit. `data_split` simply preserves the raw `v_gene`, `j_gene`, `mhc_class` columns for the model to use.

Two encoders are available (from `embeddings/feature_augment/`):

| Encoder | Flag | Dim | Notes |
|---|---|---|---|
| `OneHotFeatureAugmenter` | `--metadata_type one_hot` | 164 | Sparse, interpretable, no training needed |
| `CatAEFeatureAugmenter` | `--metadata_type cat_ae` | 65 | Dense learned latents; similar genes → similar vectors |

## Cross-attention TCR-peptide binding predictor

Pairwise cross-attention model where every TCR residue attends to every peptide residue at each layer. Takes **raw amino acid sequences** as input — it learns its own residue embeddings and does not consume pre-computed embeddings from `embeddings/`.

```bash
# sequence only
python -m models.cross_attention.train \
    --train data/splits/train.csv --val data/splits/val.csv

# with V/J + MHC metadata (one-hot encoder)
python -m models.cross_attention.train \
    --train data/splits/train.csv --val data/splits/val.csv \
    --use_metadata --metadata_type one_hot

# with V/J + MHC metadata (categorical-AE encoder)
python -m models.cross_attention.train \
    --train data/splits/train.csv --val data/splits/val.csv \
    --use_metadata --metadata_type cat_ae
```

Outputs: `outputs/models/cross_attention/checkpoints/checkpoint.pt`, `metrics.txt`.

See [cross_attention/README.md](cross_attention/README.md) for architecture details, all training arguments, and programmatic inference usage.
