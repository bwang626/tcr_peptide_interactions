"""
Generate and save ESMplusplus embeddings for the TCR-peptide dataset.

Reads processed/esm_trb_dataset.csv, passes each full-length mature TRB
amino acid sequence through ESMplusplus, extracts per-residue hidden states
from a chosen transformer layer, and mean- or max-pools them over two spans:

  full_emb  — entire mature sequence (V through C constant region).
  cdr3_emb  — CDR3 residues only (positions cdr3_start:cdr3_end).

Run from the repo root:
    python embeddings/esm/generate_embeddings.py
    python embeddings/esm/generate_embeddings.py --pooling max
    python embeddings/esm/generate_embeddings.py --layer 24
    python embeddings/esm/generate_embeddings.py --limit 100   # quick test

Outputs (outputs/embeddings/esm/):
    full_embeddings.npy      float32  (N, D)  Full-sequence pooled embeddings.
    cdr3_embeddings.npy      float32  (N, D)  CDR3-region pooled embeddings.
                                              Rows with missing CDR3 span are NaN.
    embedding_index.csv               (N,)    Row index + sequence metadata;
                                              integer index matches .npy row order.

Batching strategy
-----------------
Sequences are sorted by length before batching so that sequences of similar
length land in the same batch. This minimises padding in the attention matrix
and maximises GPU/CPU utilisation.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils._config_module as _cm
from transformers import AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL    = "Synthyra/ESMplusplus_large"
DEFAULT_POOLING  = "mean"
DEFAULT_BATCH    = 32
BASE_OUT         = "outputs/embeddings/esm"

INDEX_COLS = ["full_aa", "cdr3", "cdr3_start", "cdr3_end", "v_gene", "j_gene",
              "peptide", "mhc_a", "mhc_class", "source"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_name: str, device: str):
    """Load ESMplusplus, patching a PyTorch ≥2.5 ConfigModule compatibility shim."""
    orig = _cm.ConfigModule.__setattr__

    def _lenient(self, name, value):
        try:
            orig(self, name, value)
        except AttributeError:
            object.__setattr__(self, name, value)

    _cm.ConfigModule.__setattr__ = _lenient
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    _cm.ConfigModule.__setattr__ = orig
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _detect_bos_offset(tokenizer) -> int:
    """Return the number of tokens prepended before the first residue token.

    Probes the tokenizer with a known sequence rather than trusting
    bos_token_id, which may be set even when the tokenizer doesn't prepend it.
    """
    probe = "ACDEFGHIKLM"  # 11 unambiguous residues
    ids = tokenizer([probe], return_tensors="pt")["input_ids"][0]
    return len(ids) - len(probe) - (1 if len(ids) > len(probe) + 1 else 0)


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------

def _resolve_layer(layer_arg: int | None, num_blocks: int) -> int:
    """Map a 1-indexed user layer to the hidden_states tuple index.

    hidden_states layout when output_hidden_states=True:
      [0 .. num_blocks-1] post-block outputs before final LayerNorm.
      [num_blocks]        final LayerNorm output  (== last_hidden_state).

    None  → num_blocks  (default: final normed last layer).
    k     → hidden_states[k]  (1-indexed, so block k output).
    """
    if layer_arg is None:
        return num_blocks
    if not 1 <= layer_arg <= num_blocks:
        raise ValueError(
            f"--layer must be 1..{num_blocks} (got {layer_arg}). "
            "Omit to use the final normed layer."
        )
    return layer_arg


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _pool(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool a (L, D) residue tensor over the sequence dimension → (D,)."""
    if mode == "mean":
        return tensor.mean(dim=0)
    return tensor.max(dim=0).values


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def _make_length_sorted_batches(df: pd.DataFrame, batch_size: int) -> list[list[int]]:
    """Return batches of df row indices, sorted ascending by sequence length.

    Keeping similarly-lengthed sequences together minimises attention-matrix
    padding overhead.
    """
    order = df["full_aa"].str.len().argsort().tolist()
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


# ---------------------------------------------------------------------------
# Per-batch embedding extraction
# ---------------------------------------------------------------------------

def _embed_batch(
    model,
    tokenizer,
    seqs: list[str],
    cdr3_spans: list[tuple[int | None, int | None]],
    layer_idx: int,
    pooling: str,
    device: str,
    bos_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenise and embed one batch.

    Returns
    -------
    full_emb : float32 (B, D)
    cdr3_emb : float32 (B, D)  — NaN rows where CDR3 span is unknown.
    """
    enc = tokenizer(seqs, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    hidden = out.hidden_states[layer_idx]  # (B, L_padded, D)

    full_list: list[np.ndarray] = []
    cdr3_list: list[np.ndarray] = []

    for i, (seq, (cdr3_start, cdr3_end)) in enumerate(zip(seqs, cdr3_spans)):
        tok_start = bos_offset
        tok_end   = bos_offset + len(seq)
        residues  = hidden[i, tok_start:tok_end, :]  # (seq_len, D)

        full_list.append(_pool(residues, pooling).cpu().float().numpy())

        if cdr3_start is not None and cdr3_end is not None:
            cdr3_residues = residues[int(cdr3_start) : int(cdr3_end), :]
            cdr3_list.append(_pool(cdr3_residues, pooling).cpu().float().numpy())
        else:
            cdr3_list.append(np.full(residues.shape[-1], np.nan, dtype=np.float32))

    return np.stack(full_list, axis=0), np.stack(cdr3_list, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",       default="processed/esm_trb_dataset.csv",
                   help="Input dataset CSV (output of prepare_dataset.py).")
    p.add_argument("--out",        default=BASE_OUT,
                   help="Output directory (default: outputs/embeddings/esm).")
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--layer",      type=int, default=None,
                   help="1-indexed transformer layer (default: final normed layer).")
    p.add_argument("--pooling",    choices=["mean", "max"], default=DEFAULT_POOLING,
                   help="Pooling over the sequence dimension (default: mean).")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--limit",      type=int, default=None,
                   help="Cap sequences processed, e.g. --limit 100 for a quick test.")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading {args.model} ...")
    model = _load_model(args.model, device)
    tokenizer  = model.tokenizer
    num_blocks = len(model.transformer.blocks)
    hidden_dim = model.embed.embedding_dim
    layer_idx  = _resolve_layer(args.layer, num_blocks)
    bos_offset = _detect_bos_offset(tokenizer)

    layer_label = (
        f"final normed (index {layer_idx})"
        if layer_idx == num_blocks
        else f"block {layer_idx} pre-norm"
    )
    logger.info(f"  Blocks       : {num_blocks}")
    logger.info(f"  Hidden dim   : {hidden_dim}")
    logger.info(f"  Layer        : {layer_label}")
    logger.info(f"  Pooling      : {args.pooling}")
    logger.info(f"  BOS offset   : {bos_offset}")

    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    df = df.dropna(subset=["full_aa"]).reset_index(drop=True)
    logger.info(f"  {len(df):,} rows with valid sequence")

    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
        logger.info(f"  Limited to {len(df):,} rows (--limit {args.limit})")

    batches = _make_length_sorted_batches(df, args.batch_size)
    logger.info(f"  {len(batches)} batches (batch_size={args.batch_size})")

    full_embs = np.full((len(df), hidden_dim), np.nan, dtype=np.float32)
    cdr3_embs = np.full((len(df), hidden_dim), np.nan, dtype=np.float32)

    for b_idx, row_indices in enumerate(batches):
        seqs = [df.at[i, "full_aa"] for i in row_indices]
        spans = [
            (
                df.at[i, "cdr3_start"] if pd.notna(df.at[i, "cdr3_start"]) else None,
                df.at[i, "cdr3_end"]   if pd.notna(df.at[i, "cdr3_end"])   else None,
            )
            for i in row_indices
        ]

        full_batch, cdr3_batch = _embed_batch(
            model, tokenizer, seqs, spans, layer_idx, args.pooling, device, bos_offset
        )

        for local_i, global_i in enumerate(row_indices):
            full_embs[global_i] = full_batch[local_i]
            cdr3_embs[global_i] = cdr3_batch[local_i]

        if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(batches):
            n_done = min((b_idx + 1) * args.batch_size, len(df))
            logger.info(f"  Batch {b_idx + 1:>4}/{len(batches)}  ({n_done:>6,}/{len(df):,} sequences)")

    # ── Write outputs ─────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    np.save(os.path.join(args.out, "full_embeddings.npy"), full_embs)
    np.save(os.path.join(args.out, "cdr3_embeddings.npy"), cdr3_embs)

    index_cols = [c for c in INDEX_COLS if c in df.columns]
    df[index_cols].to_csv(os.path.join(args.out, "embedding_index.csv"), index=True)

    cdr3_valid = (~np.isnan(cdr3_embs[:, 0])).sum()
    logger.info(f"Outputs written to {args.out}")
    logger.info(f"  full_embeddings.npy : {full_embs.shape}")
    logger.info(f"  cdr3_embeddings.npy : {cdr3_embs.shape}  "
                f"({cdr3_valid:,} valid, {len(df) - cdr3_valid:,} NaN — missing CDR3 span)")
    logger.info(f"  embedding_index.csv : {len(df):,} rows, columns={index_cols}")
    logger.info(f"  Layer   : {layer_label}")
    logger.info(f"  Pooling : {args.pooling}")


if __name__ == "__main__":
    main()
