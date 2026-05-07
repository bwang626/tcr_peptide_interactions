"""
Train the cross-attention TCR-peptide binding predictor.

Expects pre-split, pre-labeled CSVs (cdr3, peptide, label, [v_gene, j_gene, mhc_class, ...]).
Data splitting and negative generation are handled upstream by a separate module.

Run from repo root:
    python -m models.cross_attention.train --train data/train.csv --val data/val.csv
    python -m models.cross_attention.train --train data/train.csv --val data/val.csv --use_metadata
    python -m models.cross_attention.train --train data/train.csv --val data/val.csv \\
        --use_metadata --metadata_type cat_ae

Outputs (outputs/models/cross_attention/ by default):
    checkpoints/checkpoint.pt    best model weights + config
    metrics.txt                  per-epoch train loss / val loss / val AUROC
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
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

class LabeledDataset(Dataset):
    """Reads pre-labeled (cdr3, peptide, label) rows. No negative generation."""

    def __init__(self, df: pd.DataFrame):
        self.tcrs   = df["cdr3"].tolist()
        self.peps   = df["peptide"].tolist()
        self.labels = df["label"].astype(float).tolist()

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, i):
        return i, self.tcrs[i], self.peps[i], self.labels[i]


# ── dataloaders ───────────────────────────────────────────────────────────────

def make_loader(df: pd.DataFrame, meta: np.ndarray | None, batch_size: int,
                shuffle: bool, device: str) -> DataLoader:
    ds = LabeledDataset(df)

    def collate(batch):
        idxs   = [b[0] for b in batch]
        tcrs   = [b[1] for b in batch]
        peps   = [b[2] for b in batch]
        labels = [b[3] for b in batch]
        ti, tm = collate_sequences(tcrs, TCR_MAX_LEN, device)
        pi, pm = collate_sequences(peps, PEP_MAX_LEN, device)
        lbl = torch.tensor(labels, dtype=torch.float32, device=device)
        m = (torch.tensor(meta[idxs], dtype=torch.float32, device=device)
             if meta is not None else None)
        return ti, pi, tm, pm, lbl, m

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# ── evaluation ────────────────────────────────────────────────────────────────

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",    required=True,
                   help="CSV with columns: cdr3, peptide, label (and optionally v_gene, j_gene, mhc_class)")
    p.add_argument("--val",      required=True,
                   help="Same format as --train")
    p.add_argument("--out",         default=OUT_DIR)
    p.add_argument("--d_model",     type=int,   default=64)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--n_layers",    type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--use_metadata", action="store_true",
                   help="Append V/J + MHC metadata features after pooling. "
                        "Requires v_gene, j_gene, mhc_class columns in the CSVs.")
    p.add_argument("--metadata_type", choices=["one_hot", "cat_ae"], default="one_hot",
                   help="one_hot (sparse, 164-dim) or cat_ae (dense learned, 65-dim). "
                        "Ignored unless --use_metadata is set.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    df_train = pd.read_csv(args.train, low_memory=False)
    df_val   = pd.read_csv(args.val,   low_memory=False)
    logger.info(f"Train: {len(df_train)} rows  |  Val: {len(df_val)} rows")

    for col in ("cdr3", "peptide", "label"):
        for name, df in [("train", df_train), ("val", df_val)]:
            if col not in df.columns:
                raise ValueError(f"--{name} CSV is missing required column '{col}'")

    # ── metadata ──────────────────────────────────────────────────────────────
    meta_train = meta_val = None
    meta_dim = 0
    if args.use_metadata:
        aug = CatAEFeatureAugmenter() if args.metadata_type == "cat_ae" else OneHotFeatureAugmenter()
        aug.fit(df_train)
        meta_train = aug.transform(df_train).astype(np.float32)
        meta_val   = aug.transform(df_val).astype(np.float32)
        meta_dim   = aug.feature_dim
        logger.info(f"Metadata: {args.metadata_type}  dim={meta_dim}  {aug.feature_breakdown()}")

    # ── loaders ───────────────────────────────────────────────────────────────
    train_loader = make_loader(df_train, meta_train, args.batch_size, shuffle=True,  device=device)
    val_loader   = make_loader(df_val,   meta_val,   args.batch_size, shuffle=False, device=device)

    # ── model ─────────────────────────────────────────────────────────────────
    model = CrossAttentionTCRPep(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, meta_dim=meta_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, args.patience // 2), factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, bad_epochs = 0.0, None, 0
    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    metrics_lines = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for ti, pi, tm, pm, labels, meta in train_loader:
            logits = model(ti, pi, tm, pm, meta)
            loss   = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * labels.size(0)
            n          += labels.size(0)

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
            "d_model":  args.d_model,
            "n_heads":  args.n_heads,
            "n_layers": args.n_layers,
            "dropout":  args.dropout,
            "meta_dim": meta_dim,
        },
    }, ckpt_path)

    with open(os.path.join(args.out, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_auc={best_auc:.4f}\n")
        f.write(f"metadata={args.metadata_type if args.use_metadata else 'none'}\n")
        f.write(f"n_params={n_params}\n")

    logger.info(f"Best val AUC: {best_auc:.4f}  |  Saved to {args.out}")


if __name__ == "__main__":
    main()
