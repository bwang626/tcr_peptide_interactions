"""ESM lookup helper that bridges immrep23/embed_esm.py output to eval_unseen.py.

embed_esm.py saves outputs/embeddings/esm_immrep23/test/ with:
    cdr3_per_residue.npy   (N, 40, D) float16  -- CDR3a+CDR3b concatenated
    cdr3_lengths.npy       (N,)       int32
    embedding_index.csv    cdr3a, cdr3b, tcra, tcrb columns

The single-chain cross-attention model (max_tcr_len=30) only uses CDR3b.
This class slices out just the CDR3b portion of each row (second half of
the concatenation), truncates/pads to max_tcr_len, and keys by CDR3b string.
"""

from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ESMImmrepLookup:
    """CDR3b-keyed per-residue ESM feature lookup for the single-chain
    cross-attention model, built from immrep23/embed_esm.py output."""

    def __init__(self, esm_dir: Path, max_len: int = 30):
        esm_dir = Path(esm_dir)
        pr_path  = esm_dir / "cdr3_per_residue.npy"
        len_path = esm_dir / "cdr3_lengths.npy"
        idx_path = esm_dir / "embedding_index.csv"
        for p in (pr_path, len_path, idx_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p}. Run: python -m immrep23.embed_esm --splits test"
                )

        # embed_esm saves (N, 40, D) concatenated CDR3a+CDR3b
        raw_features = np.load(pr_path)    # (N, 40, D)
        raw_lengths  = np.load(len_path)   # (N,) = len(cdr3a) + len(cdr3b)
        idx          = pd.read_csv(idx_path, index_col=0)
        hidden_dim   = raw_features.shape[2]

        self.feature_dim = hidden_dim
        self.max_len     = max_len

        # Extract just the CDR3b portion: CDR3a occupies positions [0, len_a),
        # CDR3b occupies [len_a, len_a+len_b). We recover len_a from cdr3a.
        cdr3a_seqs = idx["cdr3a"].fillna("").astype(str).str.replace("-","").str.upper()
        cdr3b_seqs = idx["cdr3b"].fillna("").astype(str).str.replace("-","").str.upper()

        self.features = np.zeros((len(idx), max_len, hidden_dim), dtype=np.float16)
        self.lengths  = np.zeros(len(idx), dtype=np.int32)

        seen: dict[str, int] = {}
        for row_i, (cdr3a, cdr3b) in enumerate(zip(cdr3a_seqs, cdr3b_seqs)):
            if cdr3b in seen:
                continue
            len_a = min(len(cdr3a), 40)
            len_b = min(len(cdr3b), max_len)
            if len_b == 0:
                continue
            # Slice β residues from the concatenated array
            b_slice = raw_features[row_i, len_a: len_a + len_b]
            self.features[row_i, :len_b] = b_slice
            self.lengths[row_i] = len_b
            seen[cdr3b] = row_i

        self._key_to_row = seen
        logger.info("ESMImmrepLookup: %d unique CDR3b sequences, dim=%d",
                    len(seen), hidden_dim)

    def filter_and_index(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        cdr3b_col = df["cdr3"].astype(str)  # eval_unseen maps CDR3b → cdr3
        keep = cdr3b_col.isin(self._key_to_row).to_numpy()
        n_drop = int((~keep).sum())
        if n_drop:
            logger.warning("Dropping %d/%d rows: CDR3b not in ESM lookup", n_drop, len(df))
        df_out = df.loc[keep].reset_index(drop=True)
        idx    = df_out["cdr3"].map(self._key_to_row).to_numpy(dtype=np.int64)
        return df_out, idx
