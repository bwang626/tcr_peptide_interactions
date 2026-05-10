"""Paired-chain V/J + HLA encoder for IMMREP23.

The repo's main `embeddings/feature_augment/one_hot.py` encodes a single
chain's V/J + an mhc_class column. IMMREP23 has paired α/β V/J genes
(`va`, `ja`, `vb`, `jb`) and a single HLA allele column. This is just a
direct one-hot encoder over those five categorical columns, fit on
training data only.

Usage:
    aug = PairedFeatureAugmenter()
    aug.fit(df_train)
    meta = aug.transform(df_train)   # (N, feature_dim) float32
    aug.feature_dim                  # int
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ("va", "ja", "vb", "jb", "hla")
UNK = "<UNK>"


def _build_vocab(values: pd.Series) -> dict[str, int]:
    vocab = {UNK: 0}
    for v in values.dropna().astype(str).unique():
        if v == "" or v == UNK:
            continue
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def _one_hot(values: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    n = len(values)
    out = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, v in enumerate(values.astype(str).fillna(UNK).tolist()):
        out[i, vocab.get(v, vocab[UNK])] = 1.0
    return out


class PairedFeatureAugmenter:
    """One-hot encoder for paired V/J genes + HLA. Fit on train only."""

    def __init__(self, columns: tuple[str, ...] = CATEGORICAL_COLS):
        self.columns = tuple(columns)
        self.vocabs: dict[str, dict[str, int]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PairedFeatureAugmenter":
        for col in self.columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from training dataframe")
            self.vocabs[col] = _build_vocab(df[col])
        self._fitted = True
        return self

    @property
    def feature_dim(self) -> int:
        return sum(len(v) for v in self.vocabs.values())

    def feature_breakdown(self) -> str:
        return " + ".join(f"{c}:{len(v)}" for c, v in self.vocabs.items())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        return np.concatenate(
            [_one_hot(df[c], self.vocabs[c]) for c in self.columns], axis=1
        )

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, dirpath: str | Path) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        with open(dirpath / "vocabs.pkl", "wb") as f:
            pickle.dump(self.vocabs, f)
        with open(dirpath / "config.json", "w") as f:
            json.dump({"columns": list(self.columns)}, f, indent=2)

    @classmethod
    def load(cls, dirpath: str | Path) -> "PairedFeatureAugmenter":
        dirpath = Path(dirpath)
        with open(dirpath / "config.json") as f:
            cfg = json.load(f)
        with open(dirpath / "vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)
        aug = cls(columns=tuple(cfg["columns"]))
        aug.vocabs = vocabs
        aug._fitted = True
        return aug
