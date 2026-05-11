# IMMREP23 parallel pipeline

Adapts the **two best-performing model configs** from our main-pipeline
benchmark to the IMMREP23 TCR specificity prediction challenge
([Kaggle](https://www.kaggle.com/competitions/tcr-specificity-prediction-challenge),
[GitHub mirror](https://github.com/justin-barton/IMMREP23)):

| # | Config | Main-pipeline result |
|---|---|---|
| 1 | **GNN (R-GAT) + cat_ae paired metadata** | AUROC 0.735, AUPRC 0.534 |
| 2 | **Cross-attention + ESM/CDR3 + cat_ae paired metadata** | AUROC 0.685, AUPRC 0.580 |

Runs alongside the main combined-dataset pipeline; nothing in
`data_split/`, `build_negatives.py`, or the existing `models/*/train.py`
scripts is modified.

## What's different from the main pipeline

| Aspect | Main pipeline | IMMREP23 pipeline |
|---|---|---|
| TCR chain | β only (CDR3β) | paired α + β (CDR3α + CDR3β concatenated, max 40 res) |
| Schema | `cdr3, peptide, v_gene, j_gene, mhc_class, ...` | `cdr3a, cdr3b, peptide, va, ja, vb, jb, hla, ...` |
| Sequences | raw | ANARCI-aligned (gaps stripped on load) |
| Train file | positives + negatives | positives only — negatives generated here |
| Test set | held-out peptides (split locally) | provided by organisers (positives + negatives) |
| Eval metric | AUROC, AUPRC, F1 | **Macro AUC0.1 (McClish-standardised, per-peptide mean)** |
| Metadata enc | `OneHotFeatureAugmenter` / `CatAEFeatureAugmenter` (single-chain V/J + MHC class) | `PairedFeatureAugmenter` / `PairedCatAEFeatureAugmenter` (paired Va, Ja, Vb, Jb + HLA) |
| ESM input | full TRBβ via stitchr | per-chain ESM of provided full TCRα and TCRβ; CDR3 spans extracted by substring |

## Quickstart — the two target configs

```bash
# 0. Download official IMMREP23 data (no Kaggle creds needed —
#    the organisers re-publish it openly on GitHub)
python -m immrep23.fetch
# → immrep23_data/{VDJdb_paired_chain.csv, test.csv, solutions.csv, sample_submission.csv}

# ── Config 1: GNN + cat_ae paired metadata ───────────────────────────────────
python -m models.gnn.train_immrep23 \
    --raw_train immrep23_data/VDJdb_paired_chain.csv \
    --use_metadata --metadata_type cat_ae

# ── Config 2: Cross-attention + ESM/CDR3 + cat_ae paired metadata ────────────
# One-time: generate ESM per-residue features for CDR3α + CDR3β (GPU strongly recommended)
python -m immrep23.embed_esm

# Then train
python -m models.cross_attention.train_immrep23 \
    --raw_train immrep23_data/VDJdb_paired_chain.csv \
    --esm_tcr --use_metadata --metadata_type cat_ae
```

Outputs land under:
- `outputs/models/gnn_immrep23/<run>/`
- `outputs/models/cross_attention_immrep23/<run>/`

Each run writes:
- `checkpoints/checkpoint.pt` — model weights + config
- `metrics.txt` / `metrics.json` — training curves + final test metrics
- `per_peptide.csv` — per-peptide AUC0.1 / AUROC / AUPRC on the IMMREP23 test set
- `feature_augment/` — saved augmenter (when `--use_metadata`)
- `results_summary.csv` (one level up) — one row per experiment

## Optional knobs

```bash
# Use one_hot metadata instead of cat_ae
python -m models.gnn.train_immrep23           --raw_train ... --use_metadata --metadata_type one_hot
python -m models.cross_attention.train_immrep23 --raw_train ... --use_metadata --metadata_type one_hot

# Cross-attention without ESM (token-mode TCR side)
python -m models.cross_attention.train_immrep23 --raw_train ... --use_metadata --metadata_type cat_ae

# Pre-build training negatives once and reuse
python -m immrep23.build_negatives \
    --train immrep23_data/VDJdb_paired_chain.csv \
    --out   data/splits_immrep23/train.csv

python -m models.gnn.train_immrep23 --train data/splits_immrep23/train.csv ...
```

## Module layout

```
immrep23/
├── fetch.py            download data from raw.githubusercontent.com
├── dataset.py          column normalisation, gap-stripping, train/val split
├── build_negatives.py  paired-chain Levenshtein-shuffle negatives (1:5)
├── feature_augment.py  PairedFeatureAugmenter + PairedCatAEFeatureAugmenter
├── embed_esm.py        ESMplusplus per-residue embeddings for paired CDR3α+CDR3β
└── evaluate.py         Macro AUC0.1 (McClish-standardised)
```

## Schema mapping

The official columns (capitalised in the source files) are normalised to
lowercase by `dataset.load_train` / `dataset.load_test_with_labels`:

```
Peptide   → peptide        Va,Ja,TCRa     → va, ja, tcra
HLA       → hla            CDR1a, CDR2a   → cdr1a, cdr2a
Target    → label          CDR3a, CDR3a_extended → cdr3a, cdr3a_extended
ID        → id             (β fields analogous)
Label     → label          Usage          → usage
```

The trainers add a synthetic `cdr3` column = `cdr3a + cdr3b` (gap-stripped
already by `dataset.py`), so the existing model code that expects a single
`cdr3` field works unchanged. Max length is 40 residues (α typically ≤18,
β typically ≤22).

## Metric: Macro AUC0.1

Per the lessons-learned paper (Nielsen et al., 2024):

> We calculate this AUC0.1 independently for each peptide in the test set
> and then calculate the arithmetic mean of these peptide-specific AUC0.1
> scores.

`evaluate.macro_auc01(df)` does this — it computes the McClish-standardised
partial AUC up to FPR=0.1 for each peptide group and returns the arithmetic
mean. Pooled AUROC / AUPRC / AUC0.1 are reported alongside for diagnostic
comparison with our main-pipeline numbers.

## ESM strategy

The `embed_esm.py` script embeds every unique full mature **TCRα** and
**TCRβ** sequence (so the per-residue hidden states see the V-J context
that gives ESM its signal), then locates the CDR3α / CDR3β substring
inside its parent chain and extracts those residues' hidden states.

For each row, α-CDR3 residues are concatenated with β-CDR3 residues into
a single (≤40, hidden_dim) array, zero-padded to 40 — matching the
cross-attention model's `tcr_feature_dim` interface exactly. The
trainer's `ESMPairedLookup` keys on `(tcra, tcrb)` so swapped negatives
(which inherit a known TCR's pair) are looked up correctly.

## Why concatenate CDR3α + CDR3β instead of two encoders

Both models already accept a single TCR sequence with a configurable
`max_tcr_len`. Concatenation is a one-line change: extend `max_tcr_len`
to 40 and prepend α onto β. Position encoding gives the model an implicit
"α ends here, β starts here" signal at the chain boundary. A two-encoder
variant is straightforward to add later if benchmarking suggests joint
encoding loses chain-specific signal.
