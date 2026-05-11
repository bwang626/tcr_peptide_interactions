"""IMMREP23 schema → unified dataframe with normalised column names.

The official files from justin-barton/IMMREP23 use these columns:
    Peptide, HLA,
    Va, Ja, TCRa, CDR1a, CDR2a, CDR3a, CDR3a_extended,
    Vb, Jb, TCRb, CDR1b, CDR2b, CDR3b, CDR3b_extended,
    Target           (training only — always 1 since negatives are an exercise)
    ID               (test only — pair identifier)
    Label, Usage     (solutions.csv only)

This module returns a dataframe with all columns lowercased and a
guaranteed `label` column (1=binder, 0=non-binder). All sequence columns
have ANARCI alignment gaps ("-") stripped — the gaps are a presentation
artefact and our encoders treat them as unknown tokens.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we expose downstream (post-normalisation, lowercase).
SEQUENCE_COLS = (
    "tcra", "cdr1a", "cdr2a", "cdr3a", "cdr3a_extended",
    "tcrb", "cdr1b", "cdr2b", "cdr3b", "cdr3b_extended",
    "peptide",
)
GENE_COLS = ("va", "ja", "vb", "jb")
META_COLS = (*GENE_COLS, "hla")
ALL_COMMON_COLS = (*SEQUENCE_COLS, *META_COLS)


def _strip_gaps(s: object) -> str:
    """Remove ANARCI gap characters ('-' and '.') and uppercase. NaN → ''."""
    if pd.isna(s):
        return ""
    return str(s).replace("-", "").replace(".", "").strip().upper()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names + strip gaps from sequence columns."""
    df = df.rename(columns={c: c.lower() for c in df.columns})
    for col in SEQUENCE_COLS:
        if col in df.columns:
            df[col] = df[col].map(_strip_gaps)
    return df


def load_train(path: str | Path) -> pd.DataFrame:
    """Load the IMMREP23 training file. Always returns label=1 rows (positives only)."""
    df = pd.read_csv(path)
    df = _normalise(df)

    if "target" in df.columns:
        df = df.rename(columns={"target": "label"})
    else:
        df["label"] = 1

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    logger.info("IMMREP23 train: %d rows (%d positives, %d negatives)",
                len(df), n_pos, n_neg)

    if n_neg > 0:
        logger.warning(
            "Train file already contains %d negatives — IMMREP23's official "
            "training file is positives-only. Skipping our negative generator "
            "would normally mean training on positives only; check whether you "
            "really want to use the negatives in this file.", n_neg,
        )

    return df.reset_index(drop=True)


def load_test_with_labels(test_path: str | Path,
                          solutions_path: str | Path) -> pd.DataFrame:
    """Join test.csv (inputs + ID) with solutions.csv (Label + Usage) on ID."""
    test = _normalise(pd.read_csv(test_path))
    sols = pd.read_csv(solutions_path)
    sols.columns = [c.lower() for c in sols.columns]

    if "id" not in test.columns or "id" not in sols.columns:
        raise ValueError(
            f"Both test.csv and solutions.csv must have an `ID` column. "
            f"Test cols: {list(test.columns)[:6]}... Sol cols: {list(sols.columns)[:6]}..."
        )

    keep_sol = ["id", "label"]
    if "usage" in sols.columns:
        keep_sol.append("usage")
    merged = test.merge(sols[keep_sol], on="id", how="inner", validate="one_to_one")
    merged["label"] = merged["label"].astype(int)
    logger.info(
        "IMMREP23 test: %d rows  (%d positives, %d negatives, %d unique peptides)",
        len(merged), int((merged["label"] == 1).sum()),
        int((merged["label"] == 0).sum()), merged["peptide"].nunique(),
    )
    return merged.reset_index(drop=True)


def split_train_val(df: pd.DataFrame,
                    val_frac: float = 0.1,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-peptide stratified split. Each peptide contributes val_frac of its
    rows to val so train and val have the same peptide distribution.

    IMMREP23 has only ~20 epitopes and the test set's peptides overlap with
    train, so a peptide-stratified split (rather than peptide-holdout) is
    correct here — we just want a held-out slice for early stopping that
    matches the train distribution.
    """
    rng_seed = seed
    parts_train, parts_val = [], []
    for pep, grp in df.groupby("peptide", sort=False):
        n_val = max(1, int(round(len(grp) * val_frac)))
        if n_val >= len(grp):
            n_val = max(1, len(grp) - 1)
        v = grp.sample(n=n_val, random_state=rng_seed)
        t = grp.drop(v.index)
        parts_val.append(v)
        parts_train.append(t)
        rng_seed += 1

    df_train = pd.concat(parts_train).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_val   = pd.concat(parts_val).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    logger.info("Split: %d train / %d val rows over %d peptides",
                len(df_train), len(df_val), df["peptide"].nunique())
    return df_train, df_val
