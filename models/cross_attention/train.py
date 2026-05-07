"""
Train the cross-attention TCR-peptide binding predictor.

Positives come from processed/combined_trb_clean.csv.
Negatives are generated on-the-fly each epoch by shuffling peptides
across TCRs (same strategy as the graph embedder).

V/J gene and MHC class are optionally appended as one-hot metadata
features after pooling (--use_metadata flag).

Run from repo root:
    python models/cross_attention/train.py
    python models/cross_attention/train.py --use_metadata
    python models/cross_attention/train.py --max_samples 5000 --epochs 10  # quick test

Outputs (outputs/models/cross_attention/):
    checkpoint.pt       best model weights + config
    metrics.txt         val AUROC / loss per epoch
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from models.cross_attention.model import (
    CrossAttentionTCRPep,
    TCR_MAX_LEN,
    PEP_MAX_LEN,
    collate_sequences,
)
from embeddings.feature_augment.one_hot import OneHotFeatureAugmenter
from embeddings.feature_augment.autoencoder import CatAEFeatureAugmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/cross_attention"


# ── dataset ───────────────────────────────────────────────────────────────────

class BindingDataset(Dataset):
    """
    Wraps positive pairs and generates shuffled-peptide negatives on the fly.
    Each __getitem__ returns one positive and one negative.
    """

    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self.tcrs = df["cdr3"].tolist()
        self.peps = df["peptide"].tolist()
        self.unique_peps = list(set(self.peps))
        self.tcr_to_peps = {}
        for t, p in zip(self.tcrs, self.peps):
            self.tcr_to_peps.setdefault(t, set()).add(p)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tcrs)

    def _neg_peptide(self, tcr: str) -> str:
        forbidden = self.tcr_to_peps[tcr]
        for _ in range(20):
            cand = self.unique_peps[self.rng.integers(len(self.unique_peps))]
            if cand not in forbidden:
                return cand
        return cand

    def __getitem__(self, i):
        return self.tcrs[i], self.peps[i], self._neg_peptide(self.tcrs[i])


# ── training helpers ──────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, all_labels, all_probs = 0.0, [], []
    with torch.no_grad():
        for ti, pi, tm, pm, labels, meta in loader:
            logits = model(ti, pi, tm, pm, meta)
            total_loss += loss_fn(logits, labels).item()
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
    n = len(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return total_loss / n, auc


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",        default="processed/combined_trb_clean.csv")
    p.add_argument("--out",         default=OUT_DIR)
    p.add_argument("--d_model",     type=int,   default=64)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--n_layers",    type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--max_samples", type=int,   default=None)
    p.add_argument("--use_metadata", action="store_true",
                   help="Append V/J + MHC metadata features after pooling")
    p.add_argument("--metadata_type", choices=["one_hot", "cat_ae"], default="one_hot",
                   help="Metadata encoder: one_hot (sparse, 164-dim) or cat_ae (dense, 65-dim). "
                        "Ignored unless --use_metadata is set.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data, low_memory=False).dropna(subset=["cdr3", "peptide"]).reset_index(drop=True)
    logger.info(f"  {len(df)} rows")

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"  Subsampled to {len(df)} rows")

    df_train, df_val = train_test_split(df, test_size=args.val_frac, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    logger.info(f"  {len(df_train)} train / {len(df_val)} val pairs")

    # ── optional metadata ──────────────────────────────────────────────────────
    meta_train = meta_val = None
    meta_dim = 0
    if args.use_metadata:
        if args.metadata_type == "cat_ae":
            aug = CatAEFeatureAugmenter()
        else:
            aug = OneHotFeatureAugmenter()
        aug.fit(df_train)
        meta_train = aug.transform(df_train).astype(np.float32)
        meta_val   = aug.transform(df_val).astype(np.float32)
        meta_dim   = aug.feature_dim
        logger.info(f"  Metadata type: {args.metadata_type}  dim={meta_dim}  {aug.feature_breakdown()}")

    # ── dataloaders ───────────────────────────────────────────────────────────
    ds_train = BindingDataset(df_train)
    ds_val   = BindingDataset(df_val, seed=1)

    if meta_train is not None:
        # Track indices through the dataloader using a wrapper dataset
        class IndexedDataset(Dataset):
            def __init__(self, ds): self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, i): return (i,) + self.ds[i]

        ds_train_idx = IndexedDataset(ds_train)
        ds_val_idx   = IndexedDataset(ds_val)

        def collate_indexed(meta):
            def fn(batch):
                idxs   = [b[0] for b in batch]
                tcrs   = [b[1] for b in batch]
                pos_p  = [b[2] for b in batch]
                neg_p  = [b[3] for b in batch]
                all_t  = tcrs + tcrs
                all_p  = pos_p + neg_p
                ti, tm = collate_sequences(all_t, TCR_MAX_LEN, device)
                pi, pm = collate_sequences(all_p, PEP_MAX_LEN, device)
                labels = torch.cat([torch.ones(len(tcrs)), torch.zeros(len(tcrs))]).to(device)
                m = torch.tensor(
                    np.concatenate([meta[idxs], meta[idxs]], axis=0),
                    dtype=torch.float32, device=device,
                )
                return ti, pi, tm, pm, labels, m
            return fn

        train_loader = DataLoader(ds_train_idx, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_indexed(meta_train))
        val_loader   = DataLoader(ds_val_idx,   batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_indexed(meta_val))
    else:
        def simple_collate(batch):
            tcrs, pos_p, neg_p = zip(*batch)
            all_t = list(tcrs) + list(tcrs)
            all_p = list(pos_p) + list(neg_p)
            ti, tm = collate_sequences(all_t, TCR_MAX_LEN, device)
            pi, pm = collate_sequences(all_p, PEP_MAX_LEN, device)
            labels = torch.cat([torch.ones(len(tcrs)), torch.zeros(len(tcrs))]).to(device)
            return ti, pi, tm, pm, labels, None

        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=simple_collate)
        val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                                  collate_fn=simple_collate)

    # ── model ─────────────────────────────────────────────────────────────────
    model = CrossAttentionTCRPep(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, meta_dim=meta_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, args.patience // 2), factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, bad_epochs = 0.0, None, 0
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    metrics_lines = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for ti, pi, tm, pm, labels, meta in train_loader:
            logits = model(ti, pi, tm, pm, meta)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * labels.size(0)
            n += labels.size(0)

        val_loss, val_auc = evaluate(model, val_loader, device)
        sched.step(val_loss)
        line = f"epoch={epoch:3d}  train_loss={train_loss/n:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}"
        logger.info(line)
        metrics_lines.append(line)

        if val_auc > best_auc + 1e-4:
            best_auc   = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (best val AUC={best_auc:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── save ──────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.out, "checkpoints", "checkpoint.pt")
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "config": {
            "d_model":   args.d_model,
            "n_heads":   args.n_heads,
            "n_layers":  args.n_layers,
            "dropout":   args.dropout,
            "meta_dim":  meta_dim,
        },
    }, ckpt_path)

    metrics_path = os.path.join(args.out, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_auc={best_auc:.4f}\n")
        f.write(f"use_metadata={args.use_metadata}\n")
        f.write(f"n_params={n_params}\n")

    logger.info(f"Best val AUC: {best_auc:.4f}")
    logger.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
