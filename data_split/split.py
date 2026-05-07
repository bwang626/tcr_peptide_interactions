"""
Split a labeled TCR-peptide dataset into train / val / test sets.

Expects input from build_negatives.py (processed/combined_with_negatives_trb.csv),
which already has a `label` column (1 = binding, 0 = non-binding).

Three strategies (--strategy):

  peptide_holdout  [default, recommended]
    Assign a fraction of unique peptides entirely to test. No test peptide
    appears in train or val. Directly evaluates generalization to unseen
    antigens, which is the primary scientific goal per the project proposal.

  tcr_holdout
    Assign a fraction of unique CDR3 sequences entirely to test. Evaluates
    generalization to unseen T-cell receptors.

  random
    Stratified random split with no sequence-level holdout constraint.
    Fastest but overly optimistic — test peptides leak into training.

In all strategies, val is carved from the non-test portion only.
Positive/negative ratio is preserved in each split via stratification.

Run from repo root:
    python -m data_split.split                                        # peptide holdout, defaults
    python -m data_split.split --strategy tcr_holdout
    python -m data_split.split --strategy random
    python -m data_split.split --test_frac 0.15 --val_frac 0.1

Outputs (--out directory):
    train.csv   val.csv   test.csv
    split_stats.txt       row counts and positive ratios per split
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_COLS = ["cdr3", "v_gene", "j_gene", "peptide", "mhc_a", "mhc_class", "source", "label"]


# ── split strategies ──────────────────────────────────────────────────────────

def _stratified_split(df: pd.DataFrame, test_frac: float, seed: int
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple stratified split preserving positive/negative ratio."""
    train, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_peptide_holdout(
    df: pd.DataFrame, test_frac: float, val_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hold out a fraction of unique peptides for test.

    test_frac refers to the fraction of *unique peptides* assigned to test,
    not the fraction of rows (actual row fraction may differ since some
    peptides have many more pairs than others).
    """
    rng = np.random.default_rng(seed)
    unique_peps = df["peptide"].unique()
    n_test = max(1, round(len(unique_peps) * test_frac))

    test_peps = set(rng.choice(unique_peps, size=n_test, replace=False))
    test_mask = df["peptide"].isin(test_peps)

    df_test    = df[test_mask].reset_index(drop=True)
    df_trainval = df[~test_mask].reset_index(drop=True)

    # Val from the non-test portion, stratified by label
    df_train, df_val = _stratified_split(df_trainval, val_frac, seed)

    logger.info(f"  Peptide holdout: {n_test}/{len(unique_peps)} unique peptides → test")
    return df_train, df_val, df_test


def split_tcr_holdout(
    df: pd.DataFrame, test_frac: float, val_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hold out a fraction of unique CDR3 sequences for test.

    Uses CDR3 alone as the TCR identity key (V/J genes share CDR3s across
    alleles, but CDR3 is the primary binding determinant).
    """
    rng = np.random.default_rng(seed)
    unique_tcrs = df["cdr3"].unique()
    n_test = max(1, round(len(unique_tcrs) * test_frac))

    test_tcrs = set(rng.choice(unique_tcrs, size=n_test, replace=False))
    test_mask = df["cdr3"].isin(test_tcrs)

    df_test     = df[test_mask].reset_index(drop=True)
    df_trainval = df[~test_mask].reset_index(drop=True)

    df_train, df_val = _stratified_split(df_trainval, val_frac, seed)

    logger.info(f"  TCR holdout: {n_test}/{len(unique_tcrs)} unique CDR3s → test")
    return df_train, df_val, df_test


def split_random(
    df: pd.DataFrame, test_frac: float, val_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split — no sequence-level holdout."""
    df_trainval, df_test = _stratified_split(df, test_frac, seed)
    # val_frac refers to fraction of the *full* dataset; rescale for the trainval portion
    val_frac_rescaled = val_frac / (1.0 - test_frac)
    df_train, df_val = _stratified_split(df_trainval, val_frac_rescaled, seed)
    return df_train, df_val, df_test


# ── reporting ─────────────────────────────────────────────────────────────────

def _split_stats(name: str, df: pd.DataFrame) -> str:
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    ratio = n_neg / n_pos if n_pos else float("inf")
    unique_peps = df["peptide"].nunique()
    unique_tcrs = df["cdr3"].nunique()
    return (
        f"{name:6s}  rows={len(df):>7,}  pos={n_pos:>6,}  neg={n_neg:>7,}  "
        f"ratio=1:{ratio:.2f}  peptides={unique_peps:>4,}  CDR3s={unique_tcrs:>6,}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="processed/combined_with_negatives_trb.csv",
                   help="Labeled CSV from build_negatives.py (has 'label' column)")
    p.add_argument("--out",  default="data/splits",
                   help="Output directory for train.csv / val.csv / test.csv")
    p.add_argument("--strategy", choices=["peptide_holdout", "tcr_holdout", "random"],
                   default="peptide_holdout",
                   help="Split strategy (default: peptide_holdout)")
    p.add_argument("--test_frac", type=float, default=0.2,
                   help="Fraction of unique peptides/TCRs (holdout) or rows (random) for test")
    p.add_argument("--val_frac",  type=float, default=0.1,
                   help="Fraction of full dataset reserved for val (from non-test portion)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    logger.info(f"  {len(df):,} rows  |  {(df['label']==1).sum():,} pos  {(df['label']==0).sum():,} neg")

    for col in ("cdr3", "peptide", "label"):
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column '{col}'")

    strategies = {
        "peptide_holdout": split_peptide_holdout,
        "tcr_holdout":     split_tcr_holdout,
        "random":          split_random,
    }
    df_train, df_val, df_test = strategies[args.strategy](
        df, args.test_frac, args.val_frac, args.seed
    )

    # Select output columns (keep only what exists in the data)
    out_cols = [c for c in OUTPUT_COLS if c in df.columns]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train[out_cols].to_csv(out_dir / "train.csv", index=False)
    df_val[out_cols].to_csv(out_dir / "val.csv",   index=False)
    df_test[out_cols].to_csv(out_dir / "test.csv",  index=False)

    # Stats report
    lines = [
        f"strategy:   {args.strategy}",
        f"test_frac:  {args.test_frac}",
        f"val_frac:   {args.val_frac}",
        f"seed:       {args.seed}",
        "",
        _split_stats("train", df_train),
        _split_stats("val",   df_val),
        _split_stats("test",  df_test),
    ]
    stats_text = "\n".join(lines)
    (out_dir / "split_stats.txt").write_text(stats_text + "\n")

    logger.info("\n" + stats_text)
    logger.info(f"\nSplits written to {out_dir}/")


if __name__ == "__main__":
    main()
