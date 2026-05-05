"""
Tests for embeddings/one_hot/model.py
Run from repo root: python -m pytest embeddings/one_hot/test_one_hot.py -v
"""

import numpy as np
import pytest
import pandas as pd

from embeddings.one_hot.model import (
    encode_sequence, encode_sequences,
    OneHotEmbedder,
    VOCAB_SIZE, AA_TO_IDX,
    DEFAULT_MAX_LEN,
)

FAKE_TCR_SEQS = [
    "CASSLAPGATNEKLFF",
    "CASSPGTASYNEKLFF",
    "CASSLGQAYEQYF",
    "CASSQDRGTEAFF",
    "CASSGLAGGYNEQFF",
]

FAKE_PEPTIDE_SEQS = [
    "GILGFVFTL",
    "NLVPMVATV",
    "GLCTLVAML",
    "ELAGIGILTV",
    "SIINFEKL",
]


# ── Encoding tests ────────────────────────────

def test_encode_sequence_shape():
    enc = encode_sequence("CASSLAPG", max_len=15)
    assert enc.shape == (15, VOCAB_SIZE)

def test_encode_sequence_is_one_hot():
    enc = encode_sequence("CASSLAPG", max_len=15)
    assert np.allclose(enc.sum(axis=-1), 1.0), "Each position should sum to 1"

def test_encode_sequence_padding():
    enc = encode_sequence("AAA", max_len=10)
    for pos in range(3, 10):
        assert enc[pos, AA_TO_IDX["-"]] == 1.0

def test_encode_sequence_truncation():
    enc = encode_sequence("A" * 50, max_len=20)
    assert enc.shape == (20, VOCAB_SIZE)

def test_encode_sequences_batch_shape():
    batch = encode_sequences(FAKE_TCR_SEQS, max_len=30)
    assert batch.shape == (len(FAKE_TCR_SEQS), 30, VOCAB_SIZE)

def test_encode_sequences_is_one_hot():
    batch = encode_sequences(FAKE_TCR_SEQS, max_len=30)
    assert np.allclose(batch.sum(axis=-1), 1.0)


# ── OneHotEmbedder tests ──────────────────────

def test_transform_sequences_tcr_shape():
    embedder = OneHotEmbedder()
    z = embedder.transform_sequences(FAKE_TCR_SEQS, seq_type="tcr")
    assert z.shape == (len(FAKE_TCR_SEQS), 30 * VOCAB_SIZE)

def test_transform_sequences_peptide_shape():
    embedder = OneHotEmbedder()
    z = embedder.transform_sequences(FAKE_PEPTIDE_SEQS, seq_type="peptide")
    assert z.shape == (len(FAKE_PEPTIDE_SEQS), 15 * VOCAB_SIZE)

def test_transform_concat_shape():
    embedder = OneHotEmbedder()
    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    X = embedder.transform(df)
    assert X.shape == (5, 30 * VOCAB_SIZE + 15 * VOCAB_SIZE)

def test_transform_no_concat_shape():
    embedder = OneHotEmbedder(concat=False)
    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    result = embedder.transform(df)
    assert result["tcr"].shape    == (5, 30 * VOCAB_SIZE)
    assert result["peptide"].shape == (5, 15 * VOCAB_SIZE)

def test_transform_no_nan():
    embedder = OneHotEmbedder()
    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    X = embedder.transform(df)
    assert not np.any(np.isnan(X))

def test_embedding_dim_attribute():
    embedder = OneHotEmbedder()
    assert embedder.tcr_dim      == 30 * VOCAB_SIZE
    assert embedder.peptide_dim  == 15 * VOCAB_SIZE
    assert embedder.embedding_dim == embedder.tcr_dim + embedder.peptide_dim

def test_custom_max_len():
    embedder = OneHotEmbedder(tcr_max_len=20, peptide_max_len=10)
    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    X = embedder.transform(df)
    assert X.shape == (5, 20 * VOCAB_SIZE + 10 * VOCAB_SIZE)

def test_same_sequence_same_embedding():
    """Same input should always produce identical output (no randomness)."""
    embedder = OneHotEmbedder()
    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    X1 = embedder.transform(df)
    X2 = embedder.transform(df)
    assert np.allclose(X1, X2)

def test_different_sequences_different_embeddings():
    """Different sequences should produce different embeddings."""
    embedder = OneHotEmbedder()
    df = pd.DataFrame({"cdr3": FAKE_TCR_SEQS, "peptide": FAKE_PEPTIDE_SEQS})
    X = embedder.transform(df)
    # No two rows should be identical
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            assert not np.allclose(X[i], X[j]), f"Rows {i} and {j} are identical"

def test_one_hot_vs_autoencoder_dimensions():
    """
    One-hot combined dim (990) should be much larger than autoencoder's (128).
    This is the key tradeoff: one-hot is interpretable but high-dimensional.
    """
    embedder = OneHotEmbedder()
    autoencoder_dim = 64 * 2  # default latent_dim=64 for TCR + peptide
    assert embedder.embedding_dim > autoencoder_dim, (
        f"One-hot dim ({embedder.embedding_dim}) should exceed AE dim ({autoencoder_dim})"
    )