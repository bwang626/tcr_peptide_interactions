"""
Shared utilities for loading split-aligned embedding features.

This module reconstructs per-row features for train/val/test CSV splits by
looking up precomputed TCR and peptide embeddings by sequence, concatenating
them, and optionally appending categorical metadata features.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from embeddings.feature_augment import CatAEFeatureAugmenter, OneHotFeatureAugmenter

logger = logging.getLogger(__name__)

_ARTIFACT_CACHE: dict[tuple[str, str], "EmbeddingArtifact"] = {}
_AUGMENTER_CACHE: dict[tuple[str, str], object] = {}
_SPLIT_DF_CACHE: dict[tuple[str, str], pd.DataFrame] = {}

REQUIRED_SPLIT_COLUMNS = ("cdr3", "peptide", "label")
REQUIRED_META_COLUMNS = ("v_gene", "j_gene", "mhc_class")

_METHOD_LABELS = {
    "one_hot": "onehot",
    "autoencoder/plain_ae": "ae",
    "autoencoder/vae": "vae",
    "esm/cdr3": "esm_cdr3",
    "esm/full": "esm_vdj",
}

_META_LABELS = {
    None: "",
    "one_hot": "_meta",
    "cat_ae": "_meta",
}


@dataclass(frozen=True)
class EmbeddingArtifact:
    method: str
    tcr_lookup: dict[str, np.ndarray]
    peptide_lookup: dict[str, np.ndarray] | None
    tcr_dim: int
    peptide_dim: int | None


def canonicalize_embedding_name(name: str) -> str:
    """Normalize caller-provided embedding names."""
    alias_map = {
        "ae_plain": "autoencoder/plain_ae",
        "plain_ae": "autoencoder/plain_ae",
        "ae_vae": "autoencoder/vae",
        "vae": "autoencoder/vae",
        "esm_cdr3": "esm/cdr3",
        "esm_full": "esm/full",
    }
    normalized = name.strip().replace("\\", "/")
    return alias_map.get(normalized, normalized)


def method_label(name: str) -> str:
    method = canonicalize_embedding_name(name)
    if method not in _METHOD_LABELS:
        raise ValueError(f"Unsupported embedding method: {name}")
    return _METHOD_LABELS[method]


def experiment_dir_name(
    tcr_embedding: str,
    peptide_embedding: str | None = None,
    categorical: str | None = None,
) -> str:
    """Return the standardized experiment directory name."""
    tcr_method = canonicalize_embedding_name(tcr_embedding)
    meta_suffix = _META_LABELS.get(categorical)
    if meta_suffix is None:
        raise ValueError(f"Unsupported categorical mode: {categorical}")
    return f"{method_label(tcr_method)}{meta_suffix}"


def _split_path(split: str, splits_dir: str) -> str:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got {split}")
    return os.path.join(splits_dir, f"{split}.csv")


def _load_split_df(split: str, splits_dir: str) -> pd.DataFrame:
    cache_key = (os.path.abspath(splits_dir), split)
    if cache_key in _SPLIT_DF_CACHE:
        return _SPLIT_DF_CACHE[cache_key]

    path = _split_path(split, splits_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in REQUIRED_SPLIT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Split file {path} is missing columns: {missing}")
    _SPLIT_DF_CACHE[cache_key] = df
    return df


def _first_occurrence_lookup(
    sequences: pd.Series,
    embeddings: np.ndarray,
    kind: str,
    method: str,
) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    for sequence, vector in zip(sequences.astype(str).tolist(), embeddings):
        if sequence not in lookup:
            lookup[sequence] = vector.astype(np.float32, copy=False)
    logger.info(
        "Loaded %s lookup for %s: %d unique sequences",
        kind,
        method,
        len(lookup),
    )
    return lookup


def _embedding_artifact_paths(method: str, embeddings_dir: str) -> tuple[str, str | None, str]:
    method = canonicalize_embedding_name(method)
    if method in {"one_hot", "autoencoder/plain_ae", "autoencoder/vae"}:
        root = os.path.join(embeddings_dir, method)
        return (
            os.path.join(root, "tcr_embeddings.npy"),
            os.path.join(root, "peptide_embeddings.npy"),
            os.path.join(root, "embedding_index.csv"),
        )
    if method == "esm/cdr3":
        root = os.path.join(embeddings_dir, "esm")
        return (
            os.path.join(root, "cdr3_embeddings.npy"),
            None,
            os.path.join(root, "embedding_index.csv"),
        )
    if method == "esm/full":
        root = os.path.join(embeddings_dir, "esm")
        return (
            os.path.join(root, "full_embeddings.npy"),
            None,
            os.path.join(root, "embedding_index.csv"),
        )
    raise ValueError(f"Unsupported embedding method: {method}")


def _load_embedding_artifact(method: str, embeddings_dir: str) -> EmbeddingArtifact:
    method = canonicalize_embedding_name(method)
    cache_key = (os.path.abspath(embeddings_dir), method)
    if cache_key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[cache_key]

    tcr_path, peptide_path, index_path = _embedding_artifact_paths(method, embeddings_dir)
    required_paths = [tcr_path, index_path]
    if peptide_path is not None:
        required_paths.append(peptide_path)

    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing embedding artifact(s) for {method}: " + ", ".join(missing)
        )

    index_df = pd.read_csv(index_path, index_col=0)
    if "cdr3" not in index_df.columns:
        raise ValueError(f"Embedding index {index_path} is missing required column 'cdr3'")

    tcr_embeddings = np.load(tcr_path).astype(np.float32, copy=False)
    if len(index_df) != len(tcr_embeddings):
        raise ValueError(
            f"Length mismatch for {method}: index has {len(index_df)} rows, "
            f"tcr embeddings have {len(tcr_embeddings)}"
        )
    tcr_lookup = _first_occurrence_lookup(index_df["cdr3"], tcr_embeddings, "TCR", method)

    peptide_lookup = None
    peptide_dim = None
    if peptide_path is not None:
        if "peptide" not in index_df.columns:
            raise ValueError(f"Embedding index {index_path} is missing required column 'peptide'")
        peptide_embeddings = np.load(peptide_path).astype(np.float32, copy=False)
        if len(index_df) != len(peptide_embeddings):
            raise ValueError(
                f"Length mismatch for {method}: index has {len(index_df)} rows, "
                f"peptide embeddings have {len(peptide_embeddings)}"
            )
        peptide_lookup = _first_occurrence_lookup(
            index_df["peptide"], peptide_embeddings, "peptide", method
        )
        peptide_dim = int(peptide_embeddings.shape[1])

    artifact = EmbeddingArtifact(
        method=method,
        tcr_lookup=tcr_lookup,
        peptide_lookup=peptide_lookup,
        tcr_dim=int(tcr_embeddings.shape[1]),
        peptide_dim=peptide_dim,
    )
    _ARTIFACT_CACHE[cache_key] = artifact
    return artifact


def _build_augmenter(categorical: str, splits_dir: str):
    if categorical not in {"one_hot", "cat_ae"}:
        raise ValueError(f"categorical must be one of None/one_hot/cat_ae, got {categorical}")

    cache_key = (os.path.abspath(splits_dir), categorical)
    if cache_key in _AUGMENTER_CACHE:
        return _AUGMENTER_CACHE[cache_key]

    train_df = _load_split_df("train", splits_dir)
    missing = [col for col in REQUIRED_META_COLUMNS if col not in train_df.columns]
    if missing:
        raise ValueError(f"Training split is missing metadata columns: {missing}")

    augmenter = CatAEFeatureAugmenter() if categorical == "cat_ae" else OneHotFeatureAugmenter()
    augmenter.fit(train_df)
    _AUGMENTER_CACHE[cache_key] = augmenter
    logger.info(
        "Fitted %s augmenter on training split (feature_dim=%d)",
        categorical,
        augmenter.feature_dim,
    )
    return augmenter


def load_split(
    split: str,
    tcr_embedding: str,
    peptide_embedding: str | None = None,
    categorical: str | None = None,
    splits_dir: str = "data/splits",
    embeddings_dir: str = "outputs/embeddings",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a split's features and labels.

    For ESM TCR embeddings, ``peptide_embedding`` must be provided because the ESM
    artifacts only contain TCR vectors.
    """
    df = _load_split_df(split, splits_dir)

    tcr_method = canonicalize_embedding_name(tcr_embedding)
    peptide_method = canonicalize_embedding_name(peptide_embedding or tcr_embedding)
    if tcr_method.startswith("esm/") and peptide_embedding is None:
        raise ValueError(
            "peptide_embedding is required when using ESM TCR embeddings because "
            "the ESM artifacts do not include peptide vectors"
        )

    tcr_artifact = _load_embedding_artifact(tcr_method, embeddings_dir)
    peptide_artifact = _load_embedding_artifact(peptide_method, embeddings_dir)
    if peptide_artifact.peptide_lookup is None:
        raise ValueError(
            f"Embedding method {peptide_method} does not provide peptide embeddings"
        )

    missing_tcr = sorted(set(df["cdr3"].astype(str)) - set(tcr_artifact.tcr_lookup))
    missing_pep = sorted(set(df["peptide"].astype(str)) - set(peptide_artifact.peptide_lookup))
    if missing_tcr:
        preview = ", ".join(missing_tcr[:5])
        raise KeyError(
            f"{len(missing_tcr)} TCR sequences from split {split} were not found in "
            f"{tcr_method} embeddings. Examples: {preview}"
        )
    if missing_pep:
        preview = ", ".join(missing_pep[:5])
        raise KeyError(
            f"{len(missing_pep)} peptide sequences from split {split} were not found in "
            f"{peptide_method} embeddings. Examples: {preview}"
        )

    tcr_matrix = np.stack(
        [tcr_artifact.tcr_lookup[seq] for seq in df["cdr3"].astype(str)],
        axis=0,
    ).astype(np.float32, copy=False)
    peptide_matrix = np.stack(
        [peptide_artifact.peptide_lookup[seq] for seq in df["peptide"].astype(str)],
        axis=0,
    ).astype(np.float32, copy=False)
    X = np.concatenate([tcr_matrix, peptide_matrix], axis=1)

    if categorical is not None:
        missing = [col for col in REQUIRED_META_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Split {split} is missing metadata columns required for augmentation: {missing}")
        augmenter = _build_augmenter(categorical, splits_dir)
        X = augmenter.augment(X, df).astype(np.float32, copy=False)

    y = df["label"].to_numpy(dtype=np.int64, copy=True)
    logger.info(
        "Loaded split=%s tcr=%s peptide=%s categorical=%s -> X=%s y=%s",
        split,
        tcr_method,
        peptide_method,
        categorical or "none",
        X.shape,
        y.shape,
    )
    return X, y
