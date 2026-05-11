"""Generate paired-chain ESMplusplus embeddings for IMMREP23.

For each split (train + test) we:

  1. Embed every unique full mature TCRα and TCRβ sequence with
     ESMplusplus_large (so the per-residue hidden states see the V-J
     context that gives ESM its signal).
  2. Locate the CDR3α / CDR3β substring inside its parent chain and
     extract those residues' hidden states.
  3. Concatenate the α-CDR3 residues followed by the β-CDR3 residues
     into a single (≤40, hidden_dim) array, zero-padded to 40.

This matches the cross-attention model's `tcr_feature_dim` interface
exactly, so `models.cross_attention.train_immrep23 --esm_tcr` can drop
this in unchanged.

Outputs (per split, under outputs/embeddings/esm_immrep23/<split>/):

    cdr3_per_residue.npy    float16 (N, 40, D)  α residues then β residues, zero-padded
    cdr3_lengths.npy        int32   (N,)        len(cdr3a) + len(cdr3b)
    cdr3a_pooled.npy        float32 (N, D)      mean-pooled over CDR3a residues
    cdr3b_pooled.npy        float32 (N, D)      mean-pooled over CDR3b residues
    embedding_index.csv             (N, ...)    row-aligned cdr3a/cdr3b/peptide for join check

Run from repo root (after `python -m immrep23.fetch`):

    python -m immrep23.embed_esm                                # train + test, GPU strongly recommended
    python -m immrep23.embed_esm --splits test                  # just the test set
    python -m immrep23.embed_esm --limit 100                    # quick smoke-test
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.utils._config_module as _cm
from transformers import AutoModelForMaskedLM

from immrep23.dataset import load_train, load_test_with_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Synthyra/ESMplusplus_large"
DEFAULT_BATCH = 32
DEFAULT_LAYER = 24
TCR_MAX_LEN_PAIRED = 40
BASE_OUT = "outputs/embeddings/esm_immrep23"


# ── model / tokenizer plumbing (mirrors embeddings/esm/generate_embeddings.py) ──

def _load_model(model_name: str, device: str):
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


def _detect_bos_offset(tokenizer) -> int:
    probe = "ACDEFGHIKLM"
    ids = tokenizer([probe], return_tensors="pt")["input_ids"][0]
    return len(ids) - len(probe) - (1 if len(ids) > len(probe) + 1 else 0)


def _resolve_layer(layer_arg: int | None, num_blocks: int) -> int:
    if layer_arg is None:
        return num_blocks
    if not 1 <= layer_arg <= num_blocks:
        raise ValueError(f"--layer must be 1..{num_blocks} (got {layer_arg})")
    return layer_arg


# ── per-residue extraction ────────────────────────────────────────────────────

def _embed_unique_sequences(
    seqs: list[str],
    model, tokenizer, device: str,
    bos_offset: int, layer_idx: int, batch_size: int,
) -> dict[str, np.ndarray]:
    """Embed each unique sequence and return seq → (L, D) float16 array of
    its residue hidden states (no padding, no pooling)."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out: dict[str, np.ndarray] = {}
    n_batches = (len(order) + batch_size - 1) // batch_size

    for b, start in enumerate(range(0, len(order), batch_size)):
        batch_idx = order[start : start + batch_size]
        batch_seqs = [seqs[i] for i in batch_idx]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output = model(**enc, output_hidden_states=True)
        hidden = output.hidden_states[layer_idx]  # (B, L_pad, D)
        for local_i, seq in enumerate(batch_seqs):
            tok_start = bos_offset
            tok_end   = bos_offset + len(seq)
            residues  = hidden[local_i, tok_start:tok_end, :].cpu().float().numpy().astype(np.float16)
            out[seq] = residues
        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            logger.info("  embedded batch %d/%d  (%d unique sequences total)",
                        b + 1, n_batches, len(out))
    return out


def _locate_substring(haystack: str, needle: str) -> int:
    """Return start index of needle in haystack, or -1 if not found.
    For our use, haystack is the gap-stripped full TCR chain and needle
    is the gap-stripped CDR3 — they should always align, but we defend
    against malformed inputs."""
    if not needle:
        return -1
    return haystack.find(needle)


def _build_paired_per_residue(
    df: pd.DataFrame,
    tcra_emb: dict[str, np.ndarray],
    tcrb_emb: dict[str, np.ndarray],
    hidden_dim: int,
    max_len: int = TCR_MAX_LEN_PAIRED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Per row: extract CDR3α residues from TCRα embedding, then CDR3β residues
    from TCRβ embedding, concatenate, pad to max_len.

    Returns (per_residue, lengths, cdr3a_pooled, cdr3b_pooled, n_misaligned).
    """
    n = len(df)
    per_residue = np.zeros((n, max_len, hidden_dim), dtype=np.float16)
    lengths     = np.zeros(n, dtype=np.int32)
    cdr3a_pool  = np.full((n, hidden_dim), np.nan, dtype=np.float32)
    cdr3b_pool  = np.full((n, hidden_dim), np.nan, dtype=np.float32)
    n_misaligned = 0

    for i, row in df.reset_index(drop=True).iterrows():
        tcra, tcrb = str(row.get("tcra", "")), str(row.get("tcrb", ""))
        cdr3a, cdr3b = str(row.get("cdr3a", "")), str(row.get("cdr3b", ""))

        a_residues = b_residues = None

        if cdr3a and tcra in tcra_emb:
            s = _locate_substring(tcra, cdr3a)
            if s >= 0:
                a_residues = tcra_emb[tcra][s : s + len(cdr3a)]
            else:
                # Fall back: embed the CDR3 substring's positions as the FIRST
                # len(cdr3a) residues — better than dropping the row, but flag it.
                n_misaligned += 1

        if cdr3b and tcrb in tcrb_emb:
            s = _locate_substring(tcrb, cdr3b)
            if s >= 0:
                b_residues = tcrb_emb[tcrb][s : s + len(cdr3b)]
            else:
                n_misaligned += 1

        if a_residues is not None and a_residues.size:
            cdr3a_pool[i] = a_residues.astype(np.float32).mean(axis=0)
        if b_residues is not None and b_residues.size:
            cdr3b_pool[i] = b_residues.astype(np.float32).mean(axis=0)

        # Concatenate α residues, then β residues, into a single padded array.
        cursor = 0
        for chunk in (a_residues, b_residues):
            if chunk is None or chunk.size == 0:
                continue
            take = min(len(chunk), max_len - cursor)
            if take <= 0:
                break
            per_residue[i, cursor : cursor + take] = chunk[:take]
            cursor += take
        lengths[i] = cursor

    return per_residue, lengths, cdr3a_pool, cdr3b_pool, n_misaligned


# ── per-split driver ──────────────────────────────────────────────────────────

def embed_split(
    df: pd.DataFrame, split_name: str, out_root: Path,
    model, tokenizer, device: str, bos_offset: int,
    layer_idx: int, batch_size: int, hidden_dim: int,
) -> None:
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.reset_index(drop=True)
    logger.info("[%s] %d rows", split_name, len(df))

    unique_a = sorted(set(s for s in df["tcra"].astype(str) if s))
    unique_b = sorted(set(s for s in df["tcrb"].astype(str) if s))
    logger.info("[%s] embedding %d unique TCRα + %d unique TCRβ",
                split_name, len(unique_a), len(unique_b))

    tcra_emb = _embed_unique_sequences(unique_a, model, tokenizer, device,
                                       bos_offset, layer_idx, batch_size)
    tcrb_emb = _embed_unique_sequences(unique_b, model, tokenizer, device,
                                       bos_offset, layer_idx, batch_size)

    per_residue, lengths, cdr3a_pool, cdr3b_pool, n_mis = _build_paired_per_residue(
        df, tcra_emb, tcrb_emb, hidden_dim, max_len=TCR_MAX_LEN_PAIRED,
    )
    if n_mis:
        logger.warning("[%s] %d CDR3 substrings were not found inside their parent TCR — "
                       "those rows will have zero residues for the missing chain.",
                       split_name, n_mis)

    np.save(out_dir / "cdr3_per_residue.npy", per_residue)
    np.save(out_dir / "cdr3_lengths.npy",     lengths)
    np.save(out_dir / "cdr3a_pooled.npy",     cdr3a_pool)
    np.save(out_dir / "cdr3b_pooled.npy",     cdr3b_pool)

    keep = [c for c in ("cdr3a", "cdr3b", "tcra", "tcrb", "peptide", "hla", "label", "id")
            if c in df.columns]
    df[keep].to_csv(out_dir / "embedding_index.csv", index=True)

    n_nonzero = int((lengths > 0).sum())
    logger.info("[%s] wrote per_residue%s lengths%s pooled%s  (%d/%d non-empty rows)",
                split_name, per_residue.shape, lengths.shape,
                cdr3a_pool.shape, n_nonzero, len(df))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train",      default="immrep23_data/VDJdb_paired_chain.csv", type=Path)
    ap.add_argument("--test",       default="immrep23_data/test.csv", type=Path)
    ap.add_argument("--solutions",  default="immrep23_data/solutions.csv", type=Path)
    ap.add_argument("--out",        default=BASE_OUT, type=Path)
    ap.add_argument("--splits",     nargs="+", choices=["train", "test"],
                    default=["train", "test"])
    ap.add_argument("--model",      default=DEFAULT_MODEL)
    ap.add_argument("--layer",      type=int, default=DEFAULT_LAYER,
                    help="1-indexed transformer layer to extract from (default 24).")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--limit",      type=int, default=None,
                    help="Cap rows per split (smoke-test).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    logger.info("Loading %s ...", args.model)
    model = _load_model(args.model, device)
    tokenizer  = model.tokenizer
    num_blocks = len(model.transformer.blocks)
    hidden_dim = model.embed.embedding_dim
    layer_idx  = _resolve_layer(args.layer, num_blocks)
    bos_offset = _detect_bos_offset(tokenizer)
    logger.info("  blocks=%d  hidden_dim=%d  layer=%d  bos_offset=%d",
                num_blocks, hidden_dim, layer_idx, bos_offset)

    if "train" in args.splits:
        df_train = load_train(args.train)
        if args.limit:
            df_train = df_train.head(args.limit).reset_index(drop=True)
        embed_split(df_train, "train", args.out, model, tokenizer, device,
                    bos_offset, layer_idx, args.batch_size, hidden_dim)

    if "test" in args.splits:
        df_test = load_test_with_labels(args.test, args.solutions)
        if args.limit:
            df_test = df_test.head(args.limit).reset_index(drop=True)
        embed_split(df_test, "test", args.out, model, tokenizer, device,
                    bos_offset, layer_idx, args.batch_size, hidden_dim)

    logger.info("Done. Outputs under %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
