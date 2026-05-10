# IMMREP23 parallel pipeline

Adapts the cross-attention and GNN models to the IMMREP23 TCR specificity
prediction challenge ([Kaggle](https://www.kaggle.com/competitions/tcr-specificity-prediction-challenge),
[GitHub mirror](https://github.com/justin-barton/IMMREP23)). Runs alongside the
main combined-dataset pipeline; nothing in `data_split/`, `build_negatives.py`,
or `models/*/train.py` is modified.

## What's different from the main pipeline

| Aspect | Main pipeline | IMMREP23 pipeline |
|---|---|---|
| TCR chain | β only (CDR3β) | paired α + β (CDR3α + CDR3β concatenated) |
| Schema | `cdr3, peptide, v_gene, j_gene, mhc_class, ...` | `cdr3a, cdr3b, peptide, va, ja, vb, jb, hla, ...` |
| Sequences | raw | ANARCI-aligned with gaps (stripped on load) |
| Train file | positives + negatives | positives only — negatives generated here |
| Test set | held-out peptides (split locally) | provided by organisers (positives + negatives) |
| Eval metric | AUROC, AUPRC, F1 | **Macro AUC0.1 (McClish-standardised, per-peptide mean)** |
| Models | RF, MLP, CNN, GNN, cross-attention | cross-attention, GNN |

## Quickstart

```bash
# 1. Download official IMMREP23 data (no Kaggle credentials needed —
#    the organisers re-publish it openly on GitHub)
python -m immrep23.fetch
# → immrep23_data/{VDJdb_paired_chain.csv, test.csv, solutions.csv, sample_submission.csv}

# 2a. (Optional) Pre-build training negatives once and reuse across runs
python -m immrep23.build_negatives \
    --train immrep23_data/VDJdb_paired_chain.csv \
    --out   data/splits_immrep23/train.csv

# 2b. (Or) Skip step 2a and let the trainers build negatives in-memory.

# 3. Train cross-attention (sequence only)
python -m models.cross_attention.train_immrep23 \
    --raw_train immrep23_data/VDJdb_paired_chain.csv

# 4. Train cross-attention with paired V/J + HLA metadata
python -m models.cross_attention.train_immrep23 \
    --raw_train immrep23_data/VDJdb_paired_chain.csv --use_metadata

# 5. Train GNN
python -m models.gnn.train_immrep23 \
    --raw_train immrep23_data/VDJdb_paired_chain.csv

python -m models.gnn.train_immrep23 \
    --raw_train immrep23_data/VDJdb_paired_chain.csv --use_metadata
```

Outputs land under `outputs/models/cross_attention_immrep23/<run>/` and
`outputs/models/gnn_immrep23/<run>/`. Each run writes:

- `checkpoints/checkpoint.pt` — model weights + config
- `metrics.txt` / `metrics.json` — training curves + final test metrics
- `per_peptide.csv` — per-peptide AUC0.1 / AUROC / AUPRC on the IMMREP23 test set
- `results_summary.csv` (one level up) — one row per experiment

## Module layout

```
immrep23/
├── fetch.py            download data from raw.githubusercontent.com
├── dataset.py          column normalisation, gap-stripping, train/val split
├── build_negatives.py  paired-chain Levenshtein-shuffle negatives (1:5)
├── feature_augment.py  one-hot encoder for paired V/J + HLA
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
comparison with our main pipeline numbers.

## Why concatenate CDR3α + CDR3β instead of two encoders

Both models already accept a single TCR sequence with a configurable
`max_tcr_len`. Concatenation is a one-line change: extend `max_tcr_len` to
40 and prepend α onto β. The position encoding gives the model an implicit
"α ends here, β starts here" signal at the chain boundary. A two-encoder
variant is straightforward to add later if benchmarking suggests the joint
encoding loses chain-specific signal.
