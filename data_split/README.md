# Data Split

Splits a labeled TCR-peptide dataset into train / val / test sets. Sits between negative generation and model training in the pipeline.

## Pipeline position

```
build_dataset.py          → processed/combined_trb_clean.csv          (positives only)
build_negatives.py        → processed/combined_with_negatives_trb.csv  (positives + negatives, label col)
data_split/split.py       → data/splits/train.csv, val.csv, test.csv
models/cross_attention/train.py  --train data/splits/train.csv --val data/splits/val.csv
```

## Split strategies

### `peptide_holdout` (default, recommended)

Assigns a fraction of **unique peptides** entirely to test. No test peptide appears in train or val. Directly tests whether the model generalizes to unseen antigens — the primary scientific goal per the project proposal.

Val is carved from the non-test rows only, so train/val/test are fully disjoint at the peptide level.

### `tcr_holdout`

Assigns a fraction of **unique CDR3 sequences** entirely to test. Evaluates generalization to unseen T-cell receptors.

### `random`

Stratified random shuffle with no sequence-level holdout. Fast baseline, but test peptides leak into training — overly optimistic for generalization evaluation.

## Usage

```bash
# recommended: peptide holdout, 20% test peptides, 10% val
python -m data_split.split

# tcr holdout
python -m data_split.split --strategy tcr_holdout

# custom fracs
python -m data_split.split --test_frac 0.15 --val_frac 0.1

# from a custom input file
python -m data_split.split --data processed/combined_with_negatives_trb.csv --out data/splits/
```

**Arguments:**

| Argument | Default | Notes |
|---|---|---|
| `--data` | `processed/combined_with_negatives_trb.csv` | Output of `build_negatives.py` |
| `--out` | `data/splits/` | Output directory |
| `--strategy` | `peptide_holdout` | `peptide_holdout`, `tcr_holdout`, or `random` |
| `--test_frac` | 0.2 | Fraction of unique peptides/TCRs (holdout) or rows (random) for test |
| `--val_frac` | 0.1 | Fraction of full dataset for val (drawn from non-test portion) |
| `--seed` | 42 | Random seed |

## Outputs

```
data/splits/
    train.csv
    val.csv
    test.csv
    split_stats.txt    row counts, pos/neg ratio, unique peptides and CDR3s per split
```

All CSVs have columns: `cdr3, v_gene, j_gene, peptide, mhc_a, mhc_class, source, label`

## Notes on test set usage

Test CSVs are written alongside train/val but **are not passed to any training script**. Models only receive `--train` and `--val`. Test evaluation happens in a separate evaluation step after training, keeping the test set fully held out.

For `peptide_holdout`, `test_frac` refers to the fraction of *unique peptides*, not rows. Because some peptides have many more TCR pairs than others, the actual row fraction in test may differ from `test_frac`.
