"""
Categorical feature augmentation via raw one-hot encoding.

Encodes mhc_class (1-bit), v_gene, and j_gene as sparse one-hot vectors
and concatenates them onto any sequence embedding array.

Usage:
    from embeddings.feature_augment import OneHotFeatureAugmenter

    aug = OneHotFeatureAugmenter()
    aug.fit(train_df)

    train_aug = aug.augment(train_embeddings, train_df)   # (N, d+741)
    val_aug   = aug.augment(val_embeddings,   val_df)

    aug.save("outputs/feature_augment/one_hot.pkl")
    aug = OneHotFeatureAugmenter.load("outputs/feature_augment/one_hot.pkl")

CLI:
    python -m embeddings.feature_augment.one_hot \\
        --embeddings outputs/embeddings/one_hot/combined_embeddings.npy \\
        --index      outputs/embeddings/one_hot/embedding_index.csv \\
        --data       processed/combined_trb_clean.csv \\
        --out        outputs/embeddings/one_hot/combined_augmented.npy
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ("v_gene", "j_gene")

# Metadata columns written alongside augmented .npy files — matches ESM index schema
# (minus full_aa / cdr3_start / cdr3_end which require stitchr reconstruction).
INDEX_COLS = ["cdr3", "v_gene", "j_gene", "peptide", "mhc_a", "mhc_class", "source"]


class OneHotFeatureAugmenter:
    """
    Augments sequence embeddings with one-hot encoded categorical metadata.

    Features:
      - mhc_class : 1-bit binary (MHCI=0, MHCII=1)
      - v_gene    : one-hot over training vocabulary  (134-dim on IMGT-standardized combined dataset)
      - j_gene    : one-hot over training vocabulary  (29-dim on IMGT-standardized combined dataset)

    Total feature dim: 164 (on the full IMGT-standardized dataset).
    Unknown values at inference time produce an all-zero vector.
    """

    def __init__(self):
        self._vocabs: dict[str, list] = {}
        self._fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "OneHotFeatureAugmenter":
        """Build vocabularies from training data."""
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            vocab = sorted(df[col].dropna().unique().tolist())
            self._vocabs[col] = vocab
            logger.info(f"  {col}: {len(vocab)} categories")
        self._fitted = True
        logger.info(f"OneHotFeatureAugmenter fitted. feature_dim={self.feature_dim}")
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, feature_dim) float32 array of encoded metadata."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        parts = []

        # mhc_class: 1-bit
        mhc = (df["mhc_class"].fillna("MHCI") == "MHCII").astype(np.float32).values
        parts.append(mhc.reshape(-1, 1))

        # v_gene, j_gene: one-hot
        for col in CATEGORICAL_FEATURES:
            v2i = {v: i for i, v in enumerate(self._vocabs[col])}
            n = len(self._vocabs[col])
            indices = df[col].map(v2i)
            oh = np.zeros((len(df), n), dtype=np.float32)
            valid = indices.notna()
            oh[valid.values, indices[valid].astype(int).values] = 1.0
            parts.append(oh)

        return np.concatenate(parts, axis=1)

    # ── Augment ───────────────────────────────────────────────────────────────

    def augment(self, embeddings: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Concatenate (N, d) embeddings with (N, feature_dim) metadata → (N, d+feature_dim)."""
        if len(embeddings) != len(df):
            raise ValueError(f"embeddings has {len(embeddings)} rows but df has {len(df)} rows")
        meta = self.transform(df)
        out = np.concatenate([embeddings, meta], axis=1)
        logger.info(f"Augmented: {embeddings.shape} + {meta.shape} → {out.shape}")
        return out

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return 1 + sum(len(self._vocabs[c]) for c in CATEGORICAL_FEATURES)

    def feature_breakdown(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return {"mhc_class": 1, **{c: len(self._vocabs[c]) for c in CATEGORICAL_FEATURES}}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._vocabs, f)
        logger.info(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "OneHotFeatureAugmenter":
        aug = cls()
        with open(path, "rb") as f:
            aug._vocabs = pickle.load(f)
        aug._fitted = True
        logger.info(f"Loaded from {path} (feature_dim={aug.feature_dim})")
        return aug


# ── Shared CLI helper (reused by autoencoder.py) ──────────────────────────────

def load_aligned_df(embeddings_path: str, index_path: str, data_path: str):
    """Load embeddings and align metadata from the full dataset."""
    embeddings = np.load(embeddings_path)
    index = pd.read_csv(index_path, index_col=0)
    if len(embeddings) != len(index):
        raise ValueError("Embedding row count does not match index row count")

    full = pd.read_csv(data_path, low_memory=False)
    meta_cols = [c for c in ["cdr3", "peptide", "v_gene", "j_gene", "mhc_class"] if c in full.columns]

    # The embedding scripts save the index with index=True, preserving the original
    # DataFrame row labels. Use them for a direct positional lookup so each embedding
    # row gets the exact V/J genes it was generated from — a key-based merge on
    # (cdr3, peptide) alone would assign wrong V/J to the ~19% of pairs that appear
    # with multiple V/J genes in the dataset.
    # Falls back to key-based merge when indices exceed the data file length
    # (e.g. --max_samples runs that reset_index before saving).
    if index.index.max() < len(full):
        df = full.loc[index.index, meta_cols].reset_index(drop=True)
    else:
        df = index.merge(
            full[meta_cols].drop_duplicates(subset=["cdr3", "peptide"]),
            on=["cdr3", "peptide"],
            how="left",
        )

    if len(df) != len(embeddings):
        raise ValueError(f"After lookup got {len(df)} rows, expected {len(embeddings)}")
    return embeddings, df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Augment embeddings with one-hot categorical features.")
    p.add_argument("--embeddings",    required=True)
    p.add_argument("--index",         required=True)
    p.add_argument("--data",          default="processed/combined_trb_clean.csv")
    p.add_argument("--out",           required=True)
    p.add_argument("--augmenter_out", default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    embeddings, df = load_aligned_df(args.embeddings, args.index, args.data)

    aug = OneHotFeatureAugmenter()
    aug.fit(df)
    logger.info("Feature breakdown: " + str(aug.feature_breakdown()))

    augmented = aug.augment(embeddings, df)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, augmented)

    index_path = args.out.replace(".npy", "_index.csv")
    df[[c for c in INDEX_COLS if c in df.columns]].to_csv(index_path, index=True)
    logger.info(f"Saved to {args.out}  index → {index_path}")

    if args.augmenter_out:
        aug.save(args.augmenter_out)


if __name__ == "__main__":
    main()
