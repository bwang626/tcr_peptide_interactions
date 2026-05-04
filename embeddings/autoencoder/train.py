"""
Train and save autoencoder embeddings for TCR and peptide sequences.

Run from the repo root:
    python embeddings/autoencoder/train.py                                              # plain AE
    python embeddings/autoencoder/train.py --vae                                        # VAE only
    python embeddings/autoencoder/train.py --compare                                    # both, side-by-side summary
    python embeddings/autoencoder/train.py --compare --max_samples 5000 --epochs 20     # quick CPU test

Outputs:
    Single mode  → outputs/embeddings/autoencoder/{plain_ae or vae}/
    Compare mode → outputs/embeddings/autoencoder/plain_ae/
                   outputs/embeddings/autoencoder/vae/
                   outputs/embeddings/autoencoder/comparison_summary.txt
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from embeddings.autoencoder.model import SequenceAutoencoder, encode_sequences

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_OUT = "outputs/embeddings/autoencoder"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="processed/combined_trb_clean.csv")
    p.add_argument("--out",         default=BASE_OUT,
                   help="Base output directory")
    p.add_argument("--tcr_col",     default="cdr3")
    p.add_argument("--peptide_col", default="peptide")
    p.add_argument("--latent_dim",  type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--vae",         action="store_true",
                   help="Train VAE instead of plain AE")
    p.add_argument("--compare",     action="store_true",
                   help="Train both plain AE and VAE and print a side-by-side summary")
    p.add_argument("--kl_weight",   type=float, default=0.01)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Cap rows for quick CPU test (e.g. --max_samples 5000)")
    return p.parse_args()


def load_data(args):
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    logger.info(f"  {len(df)} rows")
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"  Subsampled to {len(df)} rows (--max_samples)")
    return df


def train_one(df, args, vae: bool, out_dir: str) -> dict:
    """
    Train TCR + peptide autoencoders for one mode (plain or VAE).
    Saves all outputs to out_dir. Returns a summary dict for comparison.
    """
    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    df = df.dropna(subset=[args.tcr_col, args.peptide_col])
    tcr_seqs     = df[args.tcr_col].unique().tolist()
    peptide_seqs = df[args.peptide_col].unique().tolist()

    mode = "VAE" if vae else "Plain AE"
    logger.info(f"─── Training {mode} ───")

    # TCR
    tcr_train, tcr_val = train_test_split(tcr_seqs, test_size=args.val_frac, random_state=42)
    logger.info(f"  [{mode}] Training TCR autoencoder ({len(tcr_train)} train / {len(tcr_val)} val)...")
    tcr_ae = SequenceAutoencoder(
        seq_type="tcr", latent_dim=args.latent_dim, vae=vae, kl_weight=args.kl_weight,
    ).fit(tcr_train, val_sequences=tcr_val, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    tcr_ae.save(os.path.join(checkpoints_dir, "tcr_ae.pt"))

    # Peptide
    pep_train, pep_val = train_test_split(peptide_seqs, test_size=args.val_frac, random_state=42)
    logger.info(f"  [{mode}] Training peptide autoencoder ({len(pep_train)} train / {len(pep_val)} val)...")
    pep_ae = SequenceAutoencoder(
        seq_type="peptide", latent_dim=args.latent_dim, vae=vae, kl_weight=args.kl_weight,
    ).fit(pep_train, val_sequences=pep_val, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    pep_ae.save(os.path.join(checkpoints_dir, "peptide_ae.pt"))

    # Embed
    logger.info(f"  [{mode}] Generating embeddings...")
    tcr_z    = tcr_ae.transform(df[args.tcr_col].tolist())
    pep_z    = pep_ae.transform(df[args.peptide_col].tolist())
    combined = np.concatenate([tcr_z, pep_z], axis=1)

    np.save(os.path.join(out_dir, "tcr_embeddings.npy"),      tcr_z)
    np.save(os.path.join(out_dir, "peptide_embeddings.npy"),  pep_z)
    np.save(os.path.join(out_dir, "combined_embeddings.npy"), combined)
    df[[args.tcr_col, args.peptide_col]].to_csv(
        os.path.join(out_dir, "embedding_index.csv"), index=True)

    logger.info(f"  [{mode}] Outputs written to: {out_dir}")

    def val_loss(ae, seqs):
        ae.eval()
        X = torch.tensor(encode_sequences(seqs, ae.max_len))
        loader = DataLoader(TensorDataset(X), batch_size=args.batch_size)
        total, n = 0.0, 0
        with torch.no_grad():
            for (batch,) in loader:
                logits, mu, logvar = ae(batch)
                total += ae.loss(batch, logits, mu, logvar).item() * batch.size(0)
                n += batch.size(0)
        return total / n

    return {
        "mode":          mode,
        "tcr_val_loss":  val_loss(tcr_ae, tcr_val),
        "pep_val_loss":  val_loss(pep_ae, pep_val),
        "tcr_shape":     tcr_z.shape,
        "pep_shape":     pep_z.shape,
        "combined_shape": combined.shape,
        "out_dir":       out_dir,
    }


def print_summary(results: list, out_dir: str):
    lines = []
    lines.append("=" * 55)
    lines.append("  AUTOENCODER COMPARISON SUMMARY")
    lines.append("=" * 55)
    header = f"{'Metric':<28}" + "".join(f"{r['mode']:>12}" for r in results)
    lines.append(header)
    lines.append("-" * 55)
    lines.append(f"{'TCR val loss':<28}" + "".join(f"{r['tcr_val_loss']:>12.4f}" for r in results))
    lines.append(f"{'Peptide val loss':<28}" + "".join(f"{r['pep_val_loss']:>12.4f}" for r in results))
    lines.append(f"{'TCR embedding shape':<28}" + "".join(f"{str(r['tcr_shape']):>12}" for r in results))
    lines.append(f"{'Combined shape':<28}" + "".join(f"{str(r['combined_shape']):>12}" for r in results))
    lines.append("=" * 55)
    for r in results:
        lines.append(f"  {r['mode']} outputs → {r['out_dir']}")
    lines.append("=" * 55)

    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = os.path.join(out_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    logger.info(f"Summary saved to {summary_path}")


def main():
    args = parse_args()
    df = load_data(args)

    if args.compare:
        # Train both and compare
        results = []
        for vae, subdir in [(False, "plain_ae"), (True, "vae")]:
            out_dir = os.path.join(args.out, subdir)
            results.append(train_one(df, args, vae=vae, out_dir=out_dir))
        print_summary(results, args.out)
    else:
        # Single mode
        vae = args.vae
        subdir = "vae" if vae else "plain_ae"
        out_dir = os.path.join(args.out, subdir)
        train_one(df, args, vae=vae, out_dir=out_dir)


if __name__ == "__main__":
    main()