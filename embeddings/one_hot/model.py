"""
One-hot encoding baseline embeddings for TCR and peptide sequences.

Flattens padded one-hot encoded amino acid sequences into fixed-length
dense vectors. No training required — this is a pure preprocessing step
used as a benchmark baseline against learned embeddings (autoencoder, ESM2, etc.)

Encoding:
  - Vocabulary: 20 standard AAs + pad token + unk token = 22 tokens
  - TCR:     padded to max_len=30  → flattened to 30 x 22 = 660-dim vector
  - Peptide: padded to max_len=15  → flattened to 15 x 22 = 330-dim vector
  - Combined: concatenated         →                         990-dim vector

Usage:
    from embeddings.one_hot.model import OneHotEmbedder

    embedder = OneHotEmbedder()
    X = embedder.transform(df)           # df has columns: cdr3, peptide
    # X.shape → (N, 990)

    # Separately:
    tcr_z = embedder.transform_sequences(df["cdr3"].tolist(), seq_type="tcr")
    # tcr_z.shape → (N, 660)
"""

import numpy as np
import pandas as pd
from typing import List, Optional

# ── Constants (mirrors autoencoder/model.py for consistency) ──────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
PAD_TOKEN = "-"
UNK_TOKEN = "X"

AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX[PAD_TOKEN] = 0
AA_TO_IDX[UNK_TOKEN] = len(AMINO_ACIDS) + 1

VOCAB_SIZE = len(AA_TO_IDX)  # 22

DEFAULT_MAX_LEN = {
    "tcr":     30,
    "peptide": 15,
}


# ── Encoding utilities ────────────────────────────────────────────────────────

def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """
    One-hot encode a single sequence, padded/truncated to max_len.
    Returns array of shape (max_len, VOCAB_SIZE).
    """
    indices = [AA_TO_IDX.get(aa, AA_TO_IDX[UNK_TOKEN]) for aa in seq[:max_len]]
    indices += [AA_TO_IDX[PAD_TOKEN]] * (max_len - len(indices))
    one_hot = np.zeros((max_len, VOCAB_SIZE), dtype=np.float32)
    for pos, idx in enumerate(indices):
        one_hot[pos, idx] = 1.0
    return one_hot


def encode_sequences(sequences: List[str], max_len: int) -> np.ndarray:
    """Encode a list of sequences. Returns (N, max_len, VOCAB_SIZE)."""
    return np.stack([encode_sequence(s, max_len) for s in sequences])


def decode_one_hot(one_hot: np.ndarray) -> str:
    """Decode a (max_len, VOCAB_SIZE) array back to an AA string (strips padding)."""
    idx_to_aa = {v: k for k, v in AA_TO_IDX.items()}
    indices = one_hot.argmax(axis=-1)
    return "".join(idx_to_aa.get(i, "X") for i in indices if i != AA_TO_IDX[PAD_TOKEN])


# ── Main embedder ─────────────────────────────────────────────────────────────

class OneHotEmbedder:
    """
    One-hot baseline embedder for TCR-peptide pairs.

    No training required. Sequences are one-hot encoded and flattened
    into fixed-length vectors.

    Args:
        tcr_max_len:     max TCR sequence length (default 30)
        peptide_max_len: max peptide sequence length (default 15)
        concat:          if True (default), concatenate TCR and peptide vectors.
                         if False, return them separately as a dict.

    Embedding dimensions:
        TCR:      tcr_max_len     × 22  (default 660)
        Peptide:  peptide_max_len × 22  (default 330)
        Combined: sum of both           (default 990)
    """

    def __init__(
        self,
        tcr_max_len:     int = DEFAULT_MAX_LEN["tcr"],
        peptide_max_len: int = DEFAULT_MAX_LEN["peptide"],
        concat:          bool = True,
    ):
        self.tcr_max_len     = tcr_max_len
        self.peptide_max_len = peptide_max_len
        self.concat          = concat
        self.tcr_dim         = tcr_max_len * VOCAB_SIZE
        self.peptide_dim     = peptide_max_len * VOCAB_SIZE
        self.embedding_dim   = self.tcr_dim + self.peptide_dim

    def transform_sequences(self, sequences: List[str], seq_type: str) -> np.ndarray:
        """
        Encode and flatten a list of sequences.

        Args:
            sequences: list of amino acid strings
            seq_type:  "tcr" or "peptide"

        Returns:
            numpy array of shape (N, max_len * VOCAB_SIZE)
        """
        max_len = self.tcr_max_len if seq_type == "tcr" else self.peptide_max_len
        encoded = encode_sequences(sequences, max_len)   # (N, max_len, VOCAB_SIZE)
        return encoded.reshape(len(sequences), -1)        # (N, max_len * VOCAB_SIZE)

    def transform(
        self,
        df: pd.DataFrame,
        tcr_col:     str = "cdr3",
        peptide_col: str = "peptide",
    ) -> np.ndarray:
        """
        Embed a DataFrame of TCR-peptide pairs.

        Args:
            df:          pandas DataFrame with TCR and peptide columns
            tcr_col:     column name for CDR3β sequences (default: "cdr3")
            peptide_col: column name for peptide epitope sequences

        Returns:
            if concat=True:  numpy array of shape (N, tcr_dim + peptide_dim)
            if concat=False: dict with keys "tcr" (N, tcr_dim) and "peptide" (N, peptide_dim)
        """
        tcr_z = self.transform_sequences(df[tcr_col].tolist(),     seq_type="tcr")
        pep_z = self.transform_sequences(df[peptide_col].tolist(), seq_type="peptide")

        if self.concat:
            return np.concatenate([tcr_z, pep_z], axis=1)
        return {"tcr": tcr_z, "peptide": pep_z}