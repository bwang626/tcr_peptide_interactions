"""Paired-chain V/J + HLA encoders for IMMREP23.

The repo's main `embeddings/feature_augment/{one_hot,autoencoder}.py`
encode a single chain's V/J + an mhc_class column. IMMREP23 has paired
α/β V/J genes (`va`, `ja`, `vb`, `jb`) and a single HLA allele column.
This module provides paired versions of both encoders, fit on training
data only:

    PairedFeatureAugmenter          one-hot, dim = sum(|vocab|)
    PairedCatAEFeatureAugmenter     dense AE latents, dim = 5 * latent_dim

Both encoders expose the same .fit / .transform / .feature_dim /
.feature_breakdown / .save / .load API so the training scripts can swap
them with a single string flag.
"""

from __future__ import annotations

from pathlib import Path
import json
import logging
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

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


# ── cat_ae variant ────────────────────────────────────────────────────────────

class _CatAE(nn.Module):
    """Single-variable categorical autoencoder. Mirrors the main pipeline's _CatAE."""

    def __init__(self, vocab_size: int, latent_dim: int):
        super().__init__()
        hidden = max(vocab_size // 2, latent_dim * 4, 128)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, vocab_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def _train_cat_ae(one_hot: np.ndarray, latent_dim: int,
                  epochs: int, batch_size: int, lr: float, patience: int,
                  device: torch.device) -> _CatAE:
    vocab_size = one_hot.shape[1]
    model = _CatAE(vocab_size, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    targets = torch.arange(vocab_size)

    X = torch.tensor(one_hot, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, targets), batch_size=batch_size, shuffle=True)

    best_loss, wait, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch, tgt in loader:
            batch, tgt = batch.to(device), tgt.to(device)
            loss = loss_fn(model(batch), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
            n     += batch.size(0)
        ep_loss = total / max(1, n)
        if ep_loss < best_loss - 1e-5:
            best_loss, wait = ep_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu()


class PairedCatAEFeatureAugmenter:
    """Paired V/J + HLA encoder using one CatAE per categorical column.

    Output dimension: len(columns) * latent_dim (default 5 * 32 = 160).
    """

    def __init__(self, columns: tuple[str, ...] = CATEGORICAL_COLS,
                 latent_dim: int = 32, epochs: int = 100,
                 batch_size: int = 256, lr: float = 1e-3, patience: int = 10):
        self.columns    = tuple(columns)
        self.latent_dim = latent_dim
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self._vocabs: dict[str, list[str]] = {}
        self._models: dict[str, _CatAE] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PairedCatAEFeatureAugmenter":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for col in self.columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from training dataframe")
            vocab = sorted(v for v in df[col].dropna().astype(str).unique() if v != "")
            self._vocabs[col] = vocab
            n = len(vocab)
            if n == 0:
                raise ValueError(f"Column '{col}' has no non-empty values to fit on")
            one_hot = np.eye(n, dtype=np.float32)
            self._models[col] = _train_cat_ae(
                one_hot, self.latent_dim, self.epochs,
                self.batch_size, self.lr, self.patience, device,
            )
            logger.info("CatAE %s fitted (vocab=%d → latent=%d)", col, n, self.latent_dim)
        self._fitted = True
        return self

    @property
    def feature_dim(self) -> int:
        return self.latent_dim * len(self.columns)

    def feature_breakdown(self) -> str:
        return " + ".join(f"{c}:{self.latent_dim}" for c in self.columns)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        parts = []
        for col in self.columns:
            vocab = self._vocabs[col]
            v2i = {v: i for i, v in enumerate(vocab)}
            n = len(vocab)
            indices = df[col].astype(str).map(v2i)
            oh = np.zeros((len(df), n), dtype=np.float32)
            valid = indices.notna()
            oh[valid.values, indices[valid].astype(int).values] = 1.0
            model = self._models[col]
            model.eval()
            with torch.no_grad():
                latents = model.encode(torch.tensor(oh)).numpy()
            parts.append(latents)
        return np.concatenate(parts, axis=1).astype(np.float32)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, dirpath: str | Path) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        with open(dirpath / "vocabs.pkl", "wb") as f:
            pickle.dump(self._vocabs, f)
        cfg = {"columns": list(self.columns), "latent_dim": self.latent_dim,
               "epochs": self.epochs, "batch_size": self.batch_size,
               "lr": self.lr, "patience": self.patience}
        with open(dirpath / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        for col, model in self._models.items():
            torch.save(model.state_dict(), dirpath / f"{col}_ae.pt")

    @classmethod
    def load(cls, dirpath: str | Path) -> "PairedCatAEFeatureAugmenter":
        dirpath = Path(dirpath)
        with open(dirpath / "config.json") as f:
            cfg = json.load(f)
        with open(dirpath / "vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)
        aug = cls(columns=tuple(cfg["columns"]),
                  latent_dim=cfg["latent_dim"], epochs=cfg["epochs"],
                  batch_size=cfg["batch_size"], lr=cfg["lr"], patience=cfg["patience"])
        aug._vocabs = vocabs
        for col in aug.columns:
            n = len(vocabs[col])
            m = _CatAE(n, aug.latent_dim)
            m.load_state_dict(torch.load(dirpath / f"{col}_ae.pt", weights_only=True))
            m.eval()
            aug._models[col] = m
        aug._fitted = True
        return aug


def make_paired_augmenter(metadata_type: str):
    """Factory: 'one_hot' → PairedFeatureAugmenter, 'cat_ae' → PairedCatAEFeatureAugmenter."""
    if metadata_type == "one_hot":
        return PairedFeatureAugmenter()
    if metadata_type == "cat_ae":
        return PairedCatAEFeatureAugmenter()
    raise ValueError(f"Unknown metadata_type {metadata_type!r}; expected 'one_hot' or 'cat_ae'")
