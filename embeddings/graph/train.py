"""
Train the TCR-peptide graph embedding model and save embeddings.

Run from the repo root:
    python embeddings/graph/train.py                          # full dataset
    python embeddings/graph/train.py --max_samples 5000 --epochs 10   # quick CPU test

Outputs are written to outputs/embeddings/graph/:
    checkpoints/graph_embedder.pt
    tcr_embeddings.npy        shape (N, latent_dim)
    peptide_embeddings.npy    shape (N, latent_dim)
    combined_embeddings.npy   shape (N, 2 * latent_dim)
    embedding_index.csv       maps each row back to (cdr3, peptide)
    metrics.txt               train/val loss + AUROC
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from embeddings.graph.model import GraphEmbedder, build_pair_tensors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/embeddings/graph"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="processed/combined_trb_clean.csv")
    p.add_argument("--out",         default=OUT_DIR)
    p.add_argument("--tcr_col",     default="cdr3")
    p.add_argument("--peptide_col", default="peptide")
    p.add_argument("--latent_dim",  type=int,   default=64)
    p.add_argument("--hidden_dim",  type=int,   default=64)
    p.add_argument("--num_layers",  type=int,   default=3)
    p.add_argument("--num_heads",   type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Cap rows for quick CPU runs")
    return p.parse_args()


def load_data(args) -> pd.DataFrame:
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    df = df.dropna(subset=[args.tcr_col, args.peptide_col]).reset_index(drop=True)
    logger.info(f"  {len(df)} rows after dropping NA {args.tcr_col}/{args.peptide_col}")
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"  Subsampled to {len(df)} rows (--max_samples)")
    return df


def evaluate_auc(embedder: GraphEmbedder, df: pd.DataFrame,
                 tcr_col: str, peptide_col: str, seed: int = 7) -> float:
    """AUROC against shuffled-peptide negatives — held-out evaluation."""
    rng = np.random.default_rng(seed)
    pos = df[[tcr_col, peptide_col]].copy()
    pos["label"] = 1
    neg = pos.copy()
    shuffled = neg[peptide_col].sample(frac=1, random_state=seed).reset_index(drop=True)
    neg[peptide_col] = shuffled.values
    neg["label"] = 0
    eval_df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    probs = embedder.predict_proba(eval_df, tcr_col=tcr_col, peptide_col=peptide_col)
    return float(roc_auc_score(eval_df["label"].values, probs))


def main():
    args = parse_args()
    df = load_data(args)

    df_train, df_val = train_test_split(df, test_size=args.val_frac, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    logger.info(f"  {len(df_train)} train / {len(df_val)} val pairs")

    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)

    embedder = GraphEmbedder(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    embedder.fit(
        df_train, df_val=df_val,
        tcr_col=args.tcr_col, peptide_col=args.peptide_col,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience,
    )

    ckpt_path = os.path.join(args.out, "checkpoints", "graph_embedder.pt")
    embedder.save(ckpt_path)

    logger.info("Generating embeddings for the full dataset...")
    combined = embedder.transform(df, tcr_col=args.tcr_col, peptide_col=args.peptide_col)
    half = embedder.latent_dim
    tcr_z = combined[:, :half]
    pep_z = combined[:, half:]

    np.save(os.path.join(args.out, "tcr_embeddings.npy"),      tcr_z)
    np.save(os.path.join(args.out, "peptide_embeddings.npy"),  pep_z)
    np.save(os.path.join(args.out, "combined_embeddings.npy"), combined)
    df[[args.tcr_col, args.peptide_col]].to_csv(
        os.path.join(args.out, "embedding_index.csv"), index=True)

    val_auc = evaluate_auc(embedder, df_val, args.tcr_col, args.peptide_col)
    logger.info(f"Validation AUROC vs shuffled negatives: {val_auc:.4f}")

    metrics_path = os.path.join(args.out, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"validation_auroc_vs_shuffled_negatives: {val_auc:.4f}\n")
        f.write(f"train_pairs: {len(df_train)}\n")
        f.write(f"val_pairs:   {len(df_val)}\n")
        f.write(f"latent_dim:  {args.latent_dim}\n")
        f.write(f"combined_embedding_dim: {combined.shape[1]}\n")

    logger.info(f"Outputs written to: {args.out}")


if __name__ == "__main__":
    main()
