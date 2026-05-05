"""
Generate and save one-hot baseline embeddings for TCR-peptide pairs.

Run from the repo root:
    python embeddings/one_hot/embed.py
    python embeddings/one_hot/embed.py --max_samples 5000   # quick test

No training required — outputs are written immediately.

Outputs (written to outputs/embeddings/one_hot/):
    tcr_embeddings.npy        shape (N, 660)
    peptide_embeddings.npy    shape (N, 330)
    combined_embeddings.npy   shape (N, 990)
    embedding_index.csv       maps each row back to (cdr3, peptide)
"""

import argparse
import os
import numpy as np
import pandas as pd

from embeddings.one_hot.model import OneHotEmbedder

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs/embeddings/one_hot"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="processed/combined_trb_clean.csv")
    p.add_argument("--out",         default=OUTPUT_DIR)
    p.add_argument("--tcr_col",     default="cdr3")
    p.add_argument("--peptide_col", default="peptide")
    p.add_argument("--tcr_max_len",     type=int, default=30)
    p.add_argument("--peptide_max_len", type=int, default=15)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap rows for quick test (e.g. --max_samples 5000)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    logger.info(f"  {len(df)} rows")

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"  Subsampled to {len(df)} rows (--max_samples)")

    df = df.dropna(subset=[args.tcr_col, args.peptide_col])

    # Embed
    embedder = OneHotEmbedder(
        tcr_max_len=args.tcr_max_len,
        peptide_max_len=args.peptide_max_len,
        concat=False,
    )
    logger.info(f"Generating one-hot embeddings (TCR dim={embedder.tcr_dim}, peptide dim={embedder.peptide_dim})...")

    result = embedder.transform(df, tcr_col=args.tcr_col,
                                peptide_col=args.peptide_col)
    tcr_z    = result["tcr"]
    pep_z    = result["peptide"]
    combined = np.concatenate([tcr_z, pep_z], axis=1)

    np.save(os.path.join(args.out, "tcr_embeddings.npy"),      tcr_z)
    np.save(os.path.join(args.out, "peptide_embeddings.npy"),  pep_z)
    np.save(os.path.join(args.out, "combined_embeddings.npy"), combined)
    df[[args.tcr_col, args.peptide_col]].to_csv(
        os.path.join(args.out, "embedding_index.csv"), index=True)

    logger.info(f"TCR:      {tcr_z.shape}")
    logger.info(f"Peptide:  {pep_z.shape}")
    logger.info(f"Combined: {combined.shape}")
    logger.info(f"All outputs written to: {args.out}")


if __name__ == "__main__":
    main()