# tcr_peptide_interactions

TCR–peptide binding prediction: data collection, negative generation, splitting, embeddings, and model training.

## Full pipeline (HPC training guide)

Run these steps in order from the repo root. Each step's output is the next step's input.

### Step 1 — fetch raw data

```bash
python fetch_iedb.py            # IEDB bulk exports  →  iedb_data/
python fetch_vdjdb.py           # latest VDJdb release  →  vdjdb_data/
python tsv_to_csv.py            # vdjdb_data/*.txt  →  vdjdb_csv/
```

McPAS-TCR must be downloaded manually from `https://friedmanlab.weizmann.ac.il/McPAS-TCR/` and placed at `./McPAS-TCR.csv`.

### Step 2 — build cleaned dataset (positives only)

```bash
python build_dataset.py
```

Outputs to `processed/`: `vdjdb_trb_clean.csv`, `iedb_trb_clean.csv`, `mcpas_trb_clean.csv`, `combined_trb_clean.csv`.  
Columns: `cdr3`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, `mhc_b`, `mhc_class`, `source`.  
V/J gene names are standardised to IMGT allele format via `tidytcells`.

### Step 3 — generate negatives

```bash
python build_negatives.py
```

Adds synthetic non-binding pairs using Levenshtein-distance shuffling (IMMREP23 methodology).  
Output: `processed/combined_with_negatives_trb.csv` — same schema as above plus a `label` column (1 = binding, 0 = non-binding).

### Step 4 — split into train / val / test

```bash
# recommended: hold out 20% of unique peptides for test (no test peptide leaks into train)
python -m data_split.split

# alternative: hold out 20% of unique CDR3 sequences
python -m data_split.split --strategy tcr_holdout

# custom fractions
python -m data_split.split --test_frac 0.15 --val_frac 0.1
```

Outputs to `data/splits/`: `train.csv`, `val.csv`, `test.csv`, `split_stats.txt`.  
Columns: `cdr3`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, `mhc_class`, `source`, `label`.  
See `data_split/README.md` for full strategy descriptions.

### Step 5 — generate embeddings

All embedding commands run from the repo root. Outputs go to `outputs/embeddings/` (gitignored).

**One-hot** (no training, fast):
```bash
python -m embeddings.one_hot.embed
```

**Autoencoder** (Conv1D + BiGRU, ~10 min on CPU):
```bash
python embeddings/autoencoder/train.py           # plain AE
python embeddings/autoencoder/train.py --vae     # variational AE
```

**Graph / R-GAT** (trains end-to-end on binding labels):
```bash
python embeddings/graph/train.py --epochs 30
```

**ESM** (480M protein language model — GPU strongly recommended):
```bash
python embeddings/esm/prepare_dataset.py         # reconstruct full TRBβ sequences (~5–10 min)
python embeddings/esm/generate_embeddings.py     # mean-pool last layer
```

### Step 6 — feature augmentation (V/J gene + MHC class)

Appends V gene, J gene, and MHC class metadata to one-hot or autoencoder embeddings. Two encoders available:

```bash
# one-hot augmentation of one-hot sequence embeddings
python -m embeddings.feature_augment.one_hot \
    --embeddings outputs/embeddings/one_hot/combined_embeddings.npy \
    --index      outputs/embeddings/one_hot/embedding_index.csv \
    --data       processed/combined_trb_clean.csv \
    --out        outputs/embeddings/one_hot/augmented_combined.npy

# categorical-AE augmentation of autoencoder sequence embeddings
python -m embeddings.feature_augment.autoencoder \
    --embeddings outputs/embeddings/autoencoder/plain_ae/combined_embeddings.npy \
    --index      outputs/embeddings/autoencoder/plain_ae/embedding_index.csv \
    --data       processed/combined_trb_clean.csv \
    --out        outputs/embeddings/autoencoder/plain_ae/augmented_combined.npy
```

Each run writes `augmented_combined.npy` (sequence + metadata features) and `augmented_combined_index.csv` (row provenance).  
See `embeddings/feature_augment/README.md` for encoder dimensions and all options.

### Step 7 — train the cross-attention model

Takes pre-split labeled CSVs directly — no further splitting inside the training script.

```bash
# baseline: sequence-only
python -m models.cross_attention.train \
    --train data/splits/train.csv \
    --val   data/splits/val.csv

# with V/J + MHC metadata (one-hot encoder, +164 dim)
python -m models.cross_attention.train \
    --train data/splits/train.csv \
    --val   data/splits/val.csv \
    --use_metadata

# with V/J + MHC metadata (categorical-AE encoder, +65 dim)
python -m models.cross_attention.train \
    --train data/splits/train.csv \
    --val   data/splits/val.csv \
    --use_metadata --metadata_type cat_ae

# larger model
python -m models.cross_attention.train \
    --train data/splits/train.csv \
    --val   data/splits/val.csv \
    --d_model 128 --n_layers 3
```

Outputs to `outputs/models/cross_attention/`: `checkpoints/checkpoint.pt`, `metrics.txt`.  
The test split (`data/splits/test.csv`) is **not** passed to training — evaluate separately after training.  
See `models/cross_attention/README.md` for architecture details and all arguments.

---

## Data

Bulk data files are **not** committed (the IEDB MHC-ligand export alone is 8+ GB).
Regenerate them locally:

```bash
python fetch_iedb.py            # IEDB bulk exports -> ./iedb_data/
python fetch_vdjdb.py           # latest VDJdb release -> ./vdjdb_data/
python tsv_to_csv.py            # vdjdb_data/*.txt -> vdjdb_csv/*.csv
python build_dataset.py         # cleaned VDJdb, IEDB, McPAS, and combined TRB datasets -> ./processed/
```

For the ESM embedding pipeline, one additional step reconstructs full-length TRBβ amino acid sequences from each row's V gene, J gene, and CDR3:

```bash
python embeddings/esm/prepare_dataset.py   # full-length TRBβ sequences + CDR3 spans -> ./processed/esm_trb_dataset.csv
```

### McPAS-TCR source

- Official site used for download: `https://friedmanlab.weizmann.ac.il/McPAS-TCR/`
- Local file expected by the pipeline: `./McPAS-TCR.csv`

## Suggested workflow

1. Fetch raw data with the scripts above.
2. Explore schemas and prototype cleaning rules in `notebooks/data_cleaning_visualization.ipynb`.
3. Move stable cleaning logic into `build_dataset.py`.
4. Write reproducible outputs to `processed/`.

Detailed writeups:

- `docs/cleaning_overview.md`
- `docs/vdjdb_cleaning_summary.md`
- `docs/iedb_cleaning_summary.md`
- `docs/combined_dataset_summary.md`

## Current cleaning rules

`build_dataset.py` currently builds:

- `processed/vdjdb_trb_clean.csv`
- `processed/iedb_trb_clean.csv`
- `processed/mcpas_trb_clean.csv`
- `processed/combined_trb_clean.csv`

All three outputs use a standardized schema:

- `cdr3`
- `tcr_chain`
- `v_gene`
- `j_gene`
- `peptide`
- `mhc_a`
- `mhc_b`
- `mhc_class`
- `source`

The VDJdb path applies these filters to `vdjdb_csv/vdjdb.csv`:

- keep only `HomoSapiens`
- keep only the requested TCR chain (`TRB` by default)
- require non-empty TCR, peptide, HLA, and V/J fields
- drop TCR or peptide sequences containing wildcard characters `X`, `*`, or `?`
- drop TCRs that map to more than one distinct peptide
- standardise V/J gene names to IMGT allele format via `tidytcells`; reject non-functional (ORF, pseudogene) alleles
- drop duplicate rows after filtering

The IEDB path applies these filters to `iedb_data/tcr_full_v3.csv`:

- keep only beta-chain TCR rows from `Chain 2`
- keep only human TCR rows based on the chain organism IRI
- keep only rows associated with a high-confidence assay (multimer/tetramer, SPR, or x-ray crystallography)
- use curated CDR3/V/J values when available, otherwise calculated values
- require non-empty `cdr3`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, and inferred `mhc_class`
- drop wildcard `cdr3` or peptide sequences containing `X`, `*`, or `?`
- standardise V/J gene names to IMGT allele format via `tidytcells`; reject non-functional (ORF, pseudogene) alleles
- drop duplicate rows after filtering

The McPAS path applies these filters to `McPAS-TCR.csv`:

- keep only `Human`
- use `CDR3.beta.aa`, `TRBV`, `TRBJ`, `Epitope.peptide`, and `MHC`
- infer `mhc_class` from the `MHC` field
- require non-empty `cdr3`, `v_gene`, `j_gene`, `peptide`, `mhc_a`, and inferred `mhc_class`
- drop wildcard `cdr3` or peptide sequences containing `X`, `*`, or `?`
- standardise V/J gene names to IMGT allele format via `tidytcells`; reject non-functional (ORF, pseudogene) alleles
- drop duplicate rows after filtering

The combined output is a row-wise concatenation of the cleaned VDJdb, IEDB, and McPAS datasets, with `source` set to `VDJdb`, `IEDB`, or `McPAS`.

> **Gene quality filter:** all three sources apply `enforce_functional=True` during V/J standardisation. This rejects IMGT genes annotated as ORF or pseudogene, whose reference nucleotide sequences can contain in-frame stop codons. Approximately 3 % of rows are dropped by this filter (~2 579 / 81 977 in the current combined dataset), dominated by TRBV12-1, TRBV21-1, and TRBV23-1.

Sources:
- IEDB: https://www.iedb.org/database_export_v3.php
- VDJdb: https://github.com/antigenomics/vdjdb-db/releases
- McPAS-TCR: https://friedmanlab.weizmann.ac.il/McPAS-TCR/

## Embeddings

Fixed-length vector representations of TCR and peptide sequences, used as input to downstream models. All outputs are written to `outputs/embeddings/` (gitignored).

### One-hot encoding

No training required. Sequences are one-hot encoded and flattened.

```bash
python embeddings/one_hot/embed.py               # full dataset
```

Outputs (in `outputs/embeddings/one_hot/`):

| File | Shape |
|---|---|
| `tcr_embeddings.npy` | (N, 660) |
| `peptide_embeddings.npy` | (N, 330) |
| `combined_embeddings.npy` | (N, 990) |
| `embedding_index.csv` | row → (cdr3, peptide) |

### Autoencoder (plain AE and VAE)

Learned embeddings via a Conv1D + BiGRU sequence autoencoder. Supports plain AE and variational (VAE) modes.

```bash
python embeddings/autoencoder/train.py                            # plain AE
python embeddings/autoencoder/train.py --vae                      # VAE
python embeddings/autoencoder/train.py --compare                  # train both, print comparison
```

Outputs (in `outputs/embeddings/autoencoder/{plain_ae,vae}/`):

| File | Shape |
|---|---|
| `tcr_embeddings.npy` | (N, latent_dim) |
| `peptide_embeddings.npy` | (N, latent_dim) |
| `combined_embeddings.npy` | (N, 2 × latent_dim) |
| `checkpoints/tcr_ae.pt` | trained TCR autoencoder weights |
| `checkpoints/peptide_ae.pt` | trained peptide autoencoder weights |
| `embedding_index.csv` | row → (cdr3, peptide) |

Default `latent_dim=64`, so combined shape is (N, 128). See `embeddings/autoencoder/README.md` for all options.

### Graph (R-GAT)

Learned embeddings from a relational graph attention network trained end-to-end on binding vs. non-binding pairs. Each TCR–peptide pair is one graph; bipartite edges model potential contact sites between TCR and peptide residues.

```bash
python embeddings/graph/train.py --max_samples 5000 --epochs 10   # quick CPU test
python embeddings/graph/train.py --epochs 30                        # full run
```

Outputs (in `outputs/embeddings/graph/`):

| File | Shape |
|---|---|
| `tcr_embeddings.npy` | (N, 64) |
| `peptide_embeddings.npy` | (N, 64) |
| `combined_embeddings.npy` | (N, 128) |
| `checkpoints/graph_embedder.pt` | trained model weights |
| `embedding_index.csv` | row → (cdr3, peptide) |
| `metrics.txt` | validation AUROC |

See `embeddings/graph/README.md` for architecture details and all options.

### ESM (ESMplusplus)

Context-aware embeddings from [ESMplusplus_large](https://huggingface.co/Synthyra/ESMplusplus_large), a 480M-parameter protein language model. Unlike the other methods, ESM encodes the **full mature TRBβ chain** (V through constant region), not just the CDR3. No training is required — the model is pretrained on UniRef90.

Requires an additional dataset preparation step that reconstructs full-length sequences via [stitchr](https://github.com/JamieHeather/stitchr):

```bash
python embeddings/esm/prepare_dataset.py    # → processed/esm_trb_dataset.csv  (~5–10 min)
```

Then generate embeddings (GPU strongly recommended):

```bash
python embeddings/esm/generate_embeddings.py                        # full dataset, mean pooling, last layer
python embeddings/esm/generate_embeddings.py --pooling max          # max pooling
python embeddings/esm/generate_embeddings.py --layer 24             # extract from layer 24 of 36
python embeddings/esm/generate_embeddings.py --limit 100            # quick test (100 sequences)
```

Outputs (in `outputs/embeddings/esm/`):

| File | Shape | Description |
|---|---|---|
| `full_embeddings.npy` | (N, 1280) | Pooled over the full mature TRBβ sequence |
| `cdr3_embeddings.npy` | (N, 1280) | Pooled over CDR3 residues only |
| `embedding_index.csv` | row → sequence metadata | |

A pre-built 100-sequence test set (`processed/esm_test_dataset.csv`) is included for smoke-testing without GPU access. See `embeddings/esm/README.md` for full details.

### Feature augmentation (V/J gene + MHC class)

Appends V gene, J gene, and MHC class metadata to existing one-hot or autoencoder embeddings. Two metadata encoders:

| `--metadata_type` | Encoder | Dim added |
|---|---|---|
| `one_hot` (default) | `OneHotFeatureAugmenter` | 164 |
| `cat_ae` | `CatAEFeatureAugmenter` | 65 |

```bash
python -m embeddings.feature_augment.one_hot \
    --embeddings outputs/embeddings/one_hot/combined_embeddings.npy \
    --index      outputs/embeddings/one_hot/embedding_index.csv \
    --data       processed/combined_trb_clean.csv \
    --out        outputs/embeddings/one_hot/augmented_combined.npy

python -m embeddings.feature_augment.autoencoder \
    --embeddings outputs/embeddings/autoencoder/plain_ae/combined_embeddings.npy \
    --index      outputs/embeddings/autoencoder/plain_ae/embedding_index.csv \
    --data       processed/combined_trb_clean.csv \
    --out        outputs/embeddings/autoencoder/plain_ae/augmented_combined.npy
```

Each command writes `augmented_combined.npy` and a companion `augmented_combined_index.csv`.  
See `embeddings/feature_augment/README.md` for details.

## Data splitting

Splits the labeled dataset (positives + negatives) into train / val / test with no sequence leakage between splits.

```bash
python -m data_split.split                                # peptide holdout (recommended)
python -m data_split.split --strategy tcr_holdout         # TCR holdout
python -m data_split.split --test_frac 0.15 --val_frac 0.1
```

Outputs: `data/splits/train.csv`, `val.csv`, `test.csv`, `split_stats.txt`. See `data_split/README.md`.

## Models

### Cross-attention TCR-peptide binding predictor

Pairwise cross-attention model: every TCR residue attends to every peptide residue at each layer. Takes raw amino acid sequences as input (not pre-computed embeddings). Optionally appends V/J gene + MHC class metadata.

```bash
python -m models.cross_attention.train \
    --train data/splits/train.csv --val data/splits/val.csv

python -m models.cross_attention.train \
    --train data/splits/train.csv --val data/splits/val.csv \
    --use_metadata --metadata_type cat_ae
```

Outputs: `outputs/models/cross_attention/checkpoints/checkpoint.pt`, `metrics.txt`.  
See `models/cross_attention/README.md` for full architecture description and all training arguments.

