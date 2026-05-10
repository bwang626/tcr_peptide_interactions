"""Train the cross-attention model on the IMMREP23 paired-chain dataset.

Mirrors models/cross_attention/train.py but uses the IMMREP23 schema and
the official Macro AUC0.1 metric. The TCR sequence fed to the model is
the concatenation of CDR3α + CDR3β (gap-stripped, max 40 residues total).

Pipeline:
    1. Load IMMREP23 train (positives only) and generate negatives — either
       pass --train pointing at a pre-built combined CSV from
       `python -m immrep23.build_negatives ...`, or pass --raw_train
       pointing at the raw VDJdb_paired_chain.csv and let this script
       generate negatives in-memory.
    2. Per-peptide stratified train/val split.
    3. Train with early stopping on val Macro AUC0.1.
    4. Evaluate on the IMMREP23 test set (Public + Private combined).

Run from repo root (after `python -m immrep23.fetch`):
    # one-shot: generate negatives in-memory
    python -m models.cross_attention.train_immrep23 \\
        --raw_train immrep23_data/VDJdb_paired_chain.csv

    # or use a pre-built negatives file
    python -m immrep23.build_negatives \\
        --train immrep23_data/VDJdb_paired_chain.csv \\
        --out   data/splits_immrep23/train.csv
    python -m models.cross_attention.train_immrep23 \\
        --train data/splits_immrep23/train.csv

    # with paired V/J + HLA metadata
    python -m models.cross_attention.train_immrep23 \\
        --raw_train immrep23_data/VDJdb_paired_chain.csv --use_metadata

Outputs (outputs/models/cross_attention_immrep23/<run>/):
    checkpoints/checkpoint.pt    state_dict + config
    metrics.txt                  per-epoch losses + final test metrics
    metrics.json                 final test metrics + run config
    per_peptide.csv              per-peptide AUC0.1 / AUROC / AUPRC on test
    feature_augment/             saved augmenter (only with --use_metadata)
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.cross_attention.model import (
    CrossAttentionTCRPep,
    PEP_MAX_LEN,
    collate_sequences,
)
from immrep23.dataset import load_train, load_test_with_labels, split_train_val
from immrep23.build_negatives import build_negatives
from immrep23.evaluate import macro_auc01, overall_metrics
from immrep23.feature_augment import PairedFeatureAugmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/cross_attention_immrep23"

# CDR3α + CDR3β concatenated. α typically ≤18, β typically ≤22 in the dataset.
TCR_MAX_LEN_PAIRED = 40


# ── data prep ─────────────────────────────────────────────────────────────────

def _attach_paired_cdr3(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `cdr3` column = CDR3α + CDR3β (gap-stripped already)."""
    df = df.copy()
    df["cdr3"] = (df["cdr3a"].fillna("").astype(str)
                  + df["cdr3b"].fillna("").astype(str))
    df = df[df["cdr3"].str.len() > 0].reset_index(drop=True)
    return df


def _load_train_with_negatives(args) -> pd.DataFrame:
    if args.train:
        df = pd.read_csv(args.train, low_memory=False)
        if "label" not in df.columns:
            raise ValueError(f"--train CSV must have a label column; got {list(df.columns)[:8]}")
        logger.info("Train (pre-built): %d rows  (%d pos / %d neg)",
                    len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
        return df

    if not args.raw_train:
        raise ValueError("Pass either --train (pre-built combined CSV) or --raw_train (positives-only).")
    pos = load_train(args.raw_train)
    pos["label"] = 1
    rng = random.Random(args.neg_seed)
    neg = build_negatives(pos, rng, neg_per_pos=args.neg_per_pos)
    logger.info("Generated %d negatives in-memory  (ratio 1:%.2f)",
                len(neg), len(neg) / max(1, len(pos)))
    shared = [c for c in pos.columns if c in neg.columns and c != "label"]
    return pd.concat(
        [pos[shared + ["label"]], neg[shared + ["label"]]], ignore_index=True,
    )


# ── dataset / loader ──────────────────────────────────────────────────────────

class PairedDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.tcrs   = df["cdr3"].tolist()       # CDR3a+CDR3b
        self.peps   = df["peptide"].tolist()
        self.labels = df["label"].astype(float).tolist()
        self.peptide_col = df["peptide"].tolist()  # for grouping at eval time

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, i):
        return i, self.tcrs[i], self.peps[i], self.labels[i]


def make_loader(df: pd.DataFrame, meta: np.ndarray | None,
                batch_size: int, shuffle: bool, device: str) -> DataLoader:
    ds = PairedDataset(df)

    def collate(batch):
        idxs   = [b[0] for b in batch]
        tcrs   = [b[1] for b in batch]
        peps   = [b[2] for b in batch]
        labels = [b[3] for b in batch]
        ti, tm = collate_sequences(tcrs, TCR_MAX_LEN_PAIRED, device)
        pi, pm = collate_sequences(peps, PEP_MAX_LEN,        device)
        lbl = torch.tensor(labels, dtype=torch.float32, device=device)
        m = (torch.tensor(meta[idxs], dtype=torch.float32, device=device)
             if meta is not None else None)
        return ti, pi, tm, pm, lbl, m

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# ── eval ──────────────────────────────────────────────────────────────────────

def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for ti, pi, tm, pm, labels, meta in loader:
            logits = model(ti, pi, tm, pm, meta)
            labels_all.append(labels.cpu().numpy())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def val_macro_auc(model, df_val: pd.DataFrame, loader) -> tuple[float, float]:
    """Returns (val_loss_dummy, macro_auc01) for early stopping signal.

    We use a dummy 0 for loss because the scheduler call site expects two
    values; the macro AUC0.1 is what we actually optimise against.
    """
    y, p = predict(model, loader)
    df_eval = df_val.copy()
    df_eval["score"] = p
    df_eval["label"] = y
    macro, _ = macro_auc01(df_eval)
    return 0.0, macro


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--train",     default=None, type=Path,
                    help="Pre-built combined positives+negatives CSV (from immrep23.build_negatives)")
    ap.add_argument("--raw_train", default=None, type=Path,
                    help="Raw IMMREP23 training file (positives only). Negatives are generated in-memory.")
    ap.add_argument("--test",      default=Path("immrep23_data/test.csv"), type=Path)
    ap.add_argument("--solutions", default=Path("immrep23_data/solutions.csv"), type=Path)
    ap.add_argument("--out",       default=None, type=Path)

    ap.add_argument("--val_frac",  type=float, default=0.1,
                    help="Per-peptide stratified fraction held out for early-stopping val")
    ap.add_argument("--neg_per_pos", type=int, default=5,
                    help="Negatives per positive for in-memory negative generation (--raw_train mode)")
    ap.add_argument("--neg_seed",  type=int, default=42)

    ap.add_argument("--d_model",     type=int,   default=64)
    ap.add_argument("--n_heads",     type=int,   default=4)
    ap.add_argument("--n_layers",    type=int,   default=2)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--batch_size",  type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--patience",    type=int,   default=5)
    ap.add_argument("--seed",        type=int,   default=42)

    ap.add_argument("--use_metadata", action="store_true",
                    help="Append paired V/J + HLA one-hot metadata after pooling")
    return ap.parse_args()


def _resolve_out(args) -> Path:
    if args.out:
        return args.out
    suffix = "_meta" if args.use_metadata else ""
    return Path(OUT_DIR) / f"cross_attention_immrep23{suffix}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = _resolve_out(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger.info("Device: %s  |  Output: %s", device, out_dir)

    # ── data ──────────────────────────────────────────────────────────────────
    df_full = _load_train_with_negatives(args)
    df_full = _attach_paired_cdr3(df_full)
    df_train, df_val = split_train_val(df_full, val_frac=args.val_frac, seed=args.seed)

    df_test = load_test_with_labels(args.test, args.solutions)
    df_test = _attach_paired_cdr3(df_test)
    logger.info("Test: %d rows over %d peptides",
                len(df_test), df_test["peptide"].nunique())

    # ── metadata ──────────────────────────────────────────────────────────────
    meta_train = meta_val = meta_test = None
    meta_dim = 0
    aug = None
    if args.use_metadata:
        aug = PairedFeatureAugmenter()
        aug.fit(df_train)
        meta_train = aug.transform(df_train).astype(np.float32)
        meta_val   = aug.transform(df_val).astype(np.float32)
        meta_test  = aug.transform(df_test).astype(np.float32)
        meta_dim   = aug.feature_dim
        logger.info("Metadata: dim=%d  (%s)", meta_dim, aug.feature_breakdown())

    # ── loaders ───────────────────────────────────────────────────────────────
    train_loader = make_loader(df_train, meta_train, args.batch_size, True,  device)
    val_loader   = make_loader(df_val,   meta_val,   args.batch_size, False, device)
    test_loader  = make_loader(df_test,  meta_test,  args.batch_size, False, device)

    # ── model ─────────────────────────────────────────────────────────────────
    model = CrossAttentionTCRPep(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, meta_dim=meta_dim,
        max_tcr_len=TCR_MAX_LEN_PAIRED,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %d", n_params)

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_pos = int(df_train["label"].sum())
    n_neg = len(df_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_macro, best_state, bad_epochs = -1.0, None, 0
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

        _, val_macro = val_macro_auc(model, df_val, val_loader)
        line = f"epoch={epoch:3d}  train_loss={train_loss/n:.4f}  val_macro_auc01={val_macro:.4f}"
        logger.info(line)
        metrics_lines.append(line)

        if val_macro > best_macro + 1e-4:
            best_macro = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("Early stopping at epoch %d (best val macro AUC0.1=%.4f)",
                            epoch, best_macro)
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── test eval ─────────────────────────────────────────────────────────────
    test_y, test_p = predict(model, test_loader)
    df_eval = df_test[["peptide"]].copy()
    df_eval["label"] = test_y
    df_eval["score"] = test_p

    macro, per_pep = macro_auc01(df_eval)
    pooled = overall_metrics(test_y, test_p)
    logger.info("Test  Macro AUC0.1 = %.4f", macro)
    logger.info("Test  Pooled  AUROC=%.4f  AUPRC=%.4f  AUC0.1=%.4f",
                pooled["auroc"], pooled["auprc"], pooled["auc01"])

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "config": {
            "d_model":     args.d_model,
            "n_heads":     args.n_heads,
            "n_layers":    args.n_layers,
            "dropout":     args.dropout,
            "meta_dim":    meta_dim,
            "max_tcr_len": TCR_MAX_LEN_PAIRED,
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, out_dir / "checkpoints" / "checkpoint.pt")

    if aug is not None:
        aug.save(out_dir / "feature_augment")

    per_pep.to_csv(out_dir / "per_peptide.csv", index=False)
    with open(out_dir / "metrics.txt", "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_macro_auc01={best_macro:.4f}\n")
        f.write(f"test_macro_auc01={macro:.4f}\n")
        f.write(f"test_pooled_auroc={pooled['auroc']:.4f}\n")
        f.write(f"test_pooled_auprc={pooled['auprc']:.4f}\n")
        f.write(f"test_pooled_auc01={pooled['auc01']:.4f}\n")
        f.write(f"metadata={'paired_one_hot' if args.use_metadata else 'none'}\n")
        f.write(f"n_params={n_params}\n")

    summary = {
        "experiment_name":        out_dir.name,
        "model":                  "cross_attention_paired",
        "metadata":               "paired_one_hot" if args.use_metadata else "none",
        "n_params":               int(n_params),
        "n_train":                int(len(df_train)),
        "n_val":                  int(len(df_val)),
        "n_test":                 int(len(df_test)),
        "best_val_macro_auc01":   float(best_macro),
        "test_macro_auc01":       macro,
        "test_pooled_auroc":      pooled["auroc"],
        "test_pooled_auprc":      pooled["auprc"],
        "test_pooled_auc01":      pooled["auc01"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    _update_summary(Path(OUT_DIR), summary)
    logger.info("Saved to %s", out_dir)


def _update_summary(root: Path, row: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "results_summary.csv"
    columns = list(row.keys())
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
        df = df[df["experiment_name"] != row["experiment_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    extra = [c for c in df.columns if c not in columns]
    df[columns + extra].to_csv(path, index=False)
    logger.info("Updated %s", path)


if __name__ == "__main__":
    main()
