"""
Generate and save ESMplusplus embeddings for the TCR-peptide dataset.

Reads processed/esm_trb_dataset.csv and produces three embedding files:

  full_emb     — pooled over the entire mature TRB sequence (V–CDR3–J–C).
  cdr3_emb     — pooled over CDR3 residues only (cdr3_start:cdr3_end).
  peptide_emb  — pooled over the epitope sequence.

TCR and peptide layers are set independently. Both default to the final
(layer-normed) layer. Unique peptides are embedded once and mapped to all
rows, so peptide cost scales with ~200 distinct epitopes, not 82k rows.

Run from the repo root:
    python embeddings/esm/generate_embeddings.py
    python embeddings/esm/generate_embeddings.py --pooling max
    python embeddings/esm/generate_embeddings.py --layer 24
    python embeddings/esm/generate_embeddings.py --layer 30 --peptide_layer 24
    python embeddings/esm/generate_embeddings.py --limit 100   # quick test

Outputs (outputs/embeddings/esm/):
    full_embeddings.npy           float32  (N, D)     Full-sequence TCR embeddings.
    cdr3_embeddings.npy           float32  (N, D)     CDR3-region TCR embeddings.
                                                      Rows with missing CDR3 span are NaN.
    peptide_embeddings.npy        float32  (N, D)     Peptide epitope embeddings.
    embedding_index.csv                    (N,)       Row index + sequence metadata.

With --per_residue (required for cross-attention):
    cdr3_residue_embeddings.npz   float32  (N, L, D)  Per-residue CDR3 hidden states,
                                  int32    (N,)        padded to max CDR3 length.
                                                       Keys: "embeddings", "lengths".
    peptide_residue_embeddings.npz float32 (N, L, D)  Per-residue peptide hidden states.
                                   int32   (N,)        Keys: "embeddings", "lengths".

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

def _embed_peptides(
    model,
    tokenizer,
    unique_seqs: list[str],
    layer_idx: int,
    pooling: str,
    device: str,
    bos_offset: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Embed unique peptide sequences and return a seq → pooled-vector mapping."""
    order = sorted(range(len(unique_seqs)), key=lambda i: len(unique_seqs[i]))
    result: dict[str, np.ndarray] = {}

    for start in range(0, len(order), batch_size):
        batch = [unique_seqs[i] for i in order[start : start + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer_idx]
        for local_i, seq in enumerate(batch):
            residues = hidden[local_i, bos_offset : bos_offset + len(seq), :]
            result[seq] = _pool(residues, pooling).cpu().float().numpy()

    return result


def _embed_batch(
    model,
    tokenizer,
    seqs: list[str],
    cdr3_spans: list[tuple[int | None, int | None]],
    layer_idx: int,
    pooling: str,
    device: str,
    bos_offset: int,
    per_residue: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None] | None]:
    """Tokenise and embed one batch.

    Returns
    -------
    full_emb      : float32 (B, D)
    cdr3_emb      : float32 (B, D)       — NaN rows where CDR3 span is unknown.
    cdr3_residues : list of (L_cdr3, D)  — per-residue CDR3 arrays, or None where
                    span is missing; None for the whole list if per_residue=False.
    """
    enc = tokenizer(seqs, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    hidden = out.hidden_states[layer_idx]  # (B, L_padded, D)

    full_list: list[np.ndarray] = []
    cdr3_list: list[np.ndarray] = []
    cdr3_res_list: list[np.ndarray | None] | None = [] if per_residue else None

    for i, (seq, (cdr3_start, cdr3_end)) in enumerate(zip(seqs, cdr3_spans)):
        tok_start = bos_offset
        tok_end   = bos_offset + len(seq)
        residues  = hidden[i, tok_start:tok_end, :]  # (seq_len, D)

        full_list.append(_pool(residues, pooling).cpu().float().numpy())

        if cdr3_start is not None and cdr3_end is not None:
            cdr3_res = residues[int(cdr3_start) : int(cdr3_end), :]
            cdr3_list.append(_pool(cdr3_res, pooling).cpu().float().numpy())
            if cdr3_res_list is not None:
                cdr3_res_list.append(cdr3_res.cpu().float().numpy())
        else:
            cdr3_list.append(np.full(residues.shape[-1], np.nan, dtype=np.float32))
            if cdr3_res_list is not None:
                cdr3_res_list.append(None)

    return np.stack(full_list, axis=0), np.stack(cdr3_list, axis=0), cdr3_res_list


def _save_cdr3_residue_embeddings(
    cdr3_seqs: list[str],
    cdr3_res_arrays: list[np.ndarray | None],
    hidden_dim: int,
    path: str,
) -> None:
    """Deduplicate by CDR3 sequence and save a compact float16 npz.

    Stores only unique CDR3 sequences, halving storage vs. per-row.  Float16
    cuts size by half again vs. float32.  The training loop looks up a CDR3
    string in `sequences` to find its row index into `embeddings`.

    Output npz keys
    ---------------
    embeddings : float16  (n_unique, max_cdr3_len, D)  zero-padded
    lengths    : int16    (n_unique,)                  real residue count
    sequences  : object   (n_unique,)                  CDR3 strings
    """
    # Build unique CDR3 → per-residue array mapping (first occurrence wins)
    seen: dict[str, np.ndarray] = {}
    for seq, arr in zip(cdr3_seqs, cdr3_res_arrays):
        if seq not in seen and arr is not None:
            seen[seq] = arr

    unique_seqs = list(seen.keys())
    arrays      = [seen[s] for s in unique_seqs]
    lengths     = np.array([a.shape[0] for a in arrays], dtype=np.int16)
    max_len     = int(lengths.max()) if len(lengths) else 1

    padded = np.zeros((len(arrays), max_len, hidden_dim), dtype=np.float16)
    for i, arr in enumerate(arrays):
        padded[i, : arr.shape[0]] = arr.astype(np.float16)

    np.savez_compressed(
        path,
        embeddings=padded,
        lengths=lengths,
        sequences=np.array(unique_seqs, dtype=object),
    )
    size_mb = os.path.getsize(path + ".npz" if not path.endswith(".npz") else path) / 1e6
    logger.info(
        f"  cdr3_residue_embeddings.npz : {len(unique_seqs):,} unique CDR3s, "
        f"shape {padded.shape}, {size_mb:.1f} MB"
    )


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
    p.add_argument("--layer",         type=int, default=None,
                   help="1-indexed layer for TCR embeddings (default: final normed layer).")
    p.add_argument("--peptide_layer", type=int, default=None,
                   help="1-indexed layer for peptide embeddings (default: same as --layer).")
    p.add_argument("--pooling",    choices=["mean", "max"], default=DEFAULT_POOLING,
                   help="Pooling over the sequence dimension (default: mean).")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--limit",       type=int, default=None,
                   help="Cap sequences processed, e.g. --limit 100 for a quick test.")
    p.add_argument("--per_residue", action="store_true", default=False,
                   help=(
                       "Also save per-residue CDR3 hidden states for use with the "
                       "cross-attention model. Stored as float16, deduplicated by "
                       "unique CDR3 sequence, in cdr3_residue_embeddings.npz."
                   ))
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
    tcr_layer_idx = _resolve_layer(args.layer, num_blocks)
    pep_layer_idx = _resolve_layer(args.peptide_layer, num_blocks)
    bos_offset    = _detect_bos_offset(tokenizer)

    def _layer_label(idx: int) -> str:
        return f"final normed (index {idx})" if idx == num_blocks else f"block {idx} pre-norm"

    logger.info(f"  Blocks         : {num_blocks}")
    logger.info(f"  Hidden dim     : {hidden_dim}")
    logger.info(f"  TCR layer      : {_layer_label(tcr_layer_idx)}")
    logger.info(f"  Peptide layer  : {_layer_label(pep_layer_idx)}")
    logger.info(f"  Pooling        : {args.pooling}")
    logger.info(f"  BOS offset     : {bos_offset}")

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

    # Per-residue CDR3 collection — kept in insertion order so row i maps to index i
    cdr3_seqs_ordered:     list[str]                = []
    cdr3_res_arrays_ordered: list[np.ndarray | None] = []

    for b_idx, row_indices in enumerate(batches):
        seqs = [df.at[i, "full_aa"] for i in row_indices]
        spans = [
            (
                df.at[i, "cdr3_start"] if pd.notna(df.at[i, "cdr3_start"]) else None,
                df.at[i, "cdr3_end"]   if pd.notna(df.at[i, "cdr3_end"])   else None,
            )
            for i in row_indices
        ]

        full_batch, cdr3_batch, cdr3_res_batch = _embed_batch(
            model, tokenizer, seqs, spans, tcr_layer_idx, args.pooling, device, bos_offset,
            per_residue=args.per_residue,
        )

        for local_i, global_i in enumerate(row_indices):
            full_embs[global_i] = full_batch[local_i]
            cdr3_embs[global_i] = cdr3_batch[local_i]

        if args.per_residue:
            for local_i, global_i in enumerate(row_indices):
                cdr3_seq = df.at[global_i, "cdr3"] if pd.notna(df.at[global_i, "cdr3"]) else ""
                cdr3_seqs_ordered.append(cdr3_seq)
                cdr3_res_arrays_ordered.append(cdr3_res_batch[local_i])  # type: ignore[index]

        if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(batches):
            n_done = min((b_idx + 1) * args.batch_size, len(df))
            logger.info(f"  Batch {b_idx + 1:>4}/{len(batches)}  ({n_done:>6,}/{len(df):,} sequences)")

    # ── Peptide embeddings ────────────────────────────────────────────────────
    unique_peptides = df["peptide"].dropna().unique().tolist()
    logger.info(
        f"Embedding {len(unique_peptides)} unique peptides "
        f"({_layer_label(pep_layer_idx)}) ..."
    )
    pep_map = _embed_peptides(
        model, tokenizer, unique_peptides,
        pep_layer_idx, args.pooling, device, bos_offset, args.batch_size,
    )
    pep_embs = np.full((len(df), hidden_dim), np.nan, dtype=np.float32)
    for i, pep in enumerate(df["peptide"]):
        if pd.notna(pep) and pep in pep_map:
            pep_embs[i] = pep_map[pep]

    # ── Write outputs ─────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)

    np.save(os.path.join(args.out, "full_embeddings.npy"),    full_embs)
    np.save(os.path.join(args.out, "cdr3_embeddings.npy"),    cdr3_embs)
    np.save(os.path.join(args.out, "peptide_embeddings.npy"), pep_embs)

    index_cols = [c for c in INDEX_COLS if c in df.columns]
    df[index_cols].to_csv(os.path.join(args.out, "embedding_index.csv"), index=True)

    if args.per_residue:
        res_path = os.path.join(args.out, "cdr3_residue_embeddings.npz")
        logger.info("Saving per-residue CDR3 embeddings (float16, deduplicated) ...")
        _save_cdr3_residue_embeddings(
            cdr3_seqs_ordered, cdr3_res_arrays_ordered, hidden_dim, res_path
        )

    cdr3_valid = (~np.isnan(cdr3_embs[:, 0])).sum()
    pep_valid  = (~np.isnan(pep_embs[:, 0])).sum()
    logger.info(f"Outputs written to {args.out}")
    logger.info(f"  full_embeddings.npy    : {full_embs.shape}")
    logger.info(f"  cdr3_embeddings.npy    : {cdr3_embs.shape}  "
                f"({cdr3_valid:,} valid, {len(df) - cdr3_valid:,} NaN — missing CDR3 span)")
    logger.info(f"  peptide_embeddings.npy : {pep_embs.shape}  ({pep_valid:,} valid)")
    logger.info(f"  embedding_index.csv    : {len(df):,} rows, columns={index_cols}")
    logger.info(f"  TCR layer    : {_layer_label(tcr_layer_idx)}")
    logger.info(f"  Peptide layer: {_layer_label(pep_layer_idx)}")
    logger.info(f"  Pooling      : {args.pooling}")


if __name__ == "__main__":
    main()
