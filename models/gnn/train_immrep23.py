"""Train the relational-GAT model on the IMMREP23 paired-chain dataset.

Mirrors models/gnn/train.py but uses the IMMREP23 schema and the official
Macro AUC0.1 metric. The TCR side of every pair-graph holds CDR3α + CDR3β
concatenated (gap-stripped, max 40 residues) — α residues come first,
followed by β. The GNN's existing position embedding gives the model an
implicit "where does α end and β begin" signal.

Pipeline:
    1. Load IMMREP23 train (positives only) + generate negatives.
    2. Per-peptide stratified train/val split.
    3. Train the existing TCRPeptideGNN with tcr_max_len=40, with early
       stopping on val Macro AUC0.1.
    4. Evaluate on the IMMREP23 test set and report per-peptide AUC0.1.

Run from repo root (after `python -m immrep23.fetch`):
    python -m models.gnn.train_immrep23 \\
        --raw_train immrep23_data/VDJdb_paired_chain.csv

    python -m models.gnn.train_immrep23 \\
        --raw_train immrep23_data/VDJdb_paired_chain.csv --use_metadata
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
from torch.utils.data import DataLoader

from embeddings.graph.model import build_pair_tensors
from embeddings.one_hot.model import DEFAULT_MAX_LEN
from models.gnn.train import (
    GNNBindingClassifier,
    LabeledPairDataset,
    make_collate,
    predict_proba,
)
from immrep23.dataset import load_train, load_test_with_labels, split_train_val
from immrep23.build_negatives import build_negatives
from immrep23.evaluate import macro_auc01, overall_metrics
from immrep23.feature_augment import PairedFeatureAugmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/gnn_immrep23"
TCR_MAX_LEN_PAIRED = 40


def _attach_paired_cdr3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cdr3"] = (df["cdr3a"].fillna("").astype(str)
                  + df["cdr3b"].fillna("").astype(str))
    return df[df["cdr3"].str.len() > 0].reset_index(drop=True)


def _load_train_with_negatives(args) -> pd.DataFrame:
    if args.train:
        df = pd.read_csv(args.train, low_memory=False)
        if "label" not in df.columns:
            raise ValueError(f"--train CSV must have a label column; got {list(df.columns)[:8]}")
        return df

    if not args.raw_train:
        raise ValueError("Pass either --train or --raw_train")
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


def _evaluate_macro(model, loader, device, peptides: list[str]) -> tuple[float, np.ndarray, np.ndarray]:
    """Run model over a loader and return (macro_auc01, y, p)."""
    y, p = predict_proba(model, loader, device)
    df = pd.DataFrame({"peptide": peptides, "label": y, "score": p})
    macro, _ = macro_auc01(df)
    return macro, y, p


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train",     default=None, type=Path,
                    help="Pre-built combined positives+negatives CSV")
    ap.add_argument("--raw_train", default=None, type=Path,
                    help="Raw IMMREP23 training file (positives-only); negatives generated in-memory.")
    ap.add_argument("--test",      default=Path("immrep23_data/test.csv"), type=Path)
    ap.add_argument("--solutions", default=Path("immrep23_data/solutions.csv"), type=Path)
    ap.add_argument("--out",       default=None, type=Path)

    ap.add_argument("--val_frac",   type=float, default=0.1)
    ap.add_argument("--neg_per_pos", type=int, default=5)
    ap.add_argument("--neg_seed",   type=int, default=42)

    ap.add_argument("--latent_dim", type=int,   default=64)
    ap.add_argument("--hidden_dim", type=int,   default=64)
    ap.add_argument("--num_layers", type=int,   default=3)
    ap.add_argument("--num_heads",  type=int,   default=4)
    ap.add_argument("--dropout",    type=float, default=0.1)

    ap.add_argument("--tcr_max_len", type=int, default=TCR_MAX_LEN_PAIRED)
    ap.add_argument("--pep_max_len", type=int, default=DEFAULT_MAX_LEN["peptide"])

    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--batch_size", type=int,   default=128)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--patience",   type=int,   default=5)
    ap.add_argument("--seed",       type=int,   default=42)

    ap.add_argument("--use_metadata", action="store_true",
                    help="Append paired V/J + HLA one-hot metadata after pooling")
    return ap.parse_args()


def _resolve_out(args) -> Path:
    if args.out:
        return args.out
    suffix = "_meta" if args.use_metadata else ""
    return Path(OUT_DIR) / f"gnn_immrep23{suffix}"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    y_tr = df_train["label"].astype(np.float32).to_numpy()
    y_va = df_val["label"].astype(np.float32).to_numpy()
    y_te = df_test["label"].astype(np.float32).to_numpy()

    # ── metadata ──────────────────────────────────────────────────────────────
    meta_tr = meta_va = meta_te = None
    meta_dim = 0
    aug = None
    if args.use_metadata:
        aug = PairedFeatureAugmenter()
        aug.fit(df_train)
        meta_tr = aug.transform(df_train).astype(np.float32)
        meta_va = aug.transform(df_val).astype(np.float32)
        meta_te = aug.transform(df_test).astype(np.float32)
        meta_dim = aug.feature_dim
        logger.info("Metadata: dim=%d  (%s)", meta_dim, aug.feature_breakdown())

    # ── datasets / loaders ────────────────────────────────────────────────────
    has_meta = meta_dim > 0
    collate = make_collate(args.tcr_max_len, args.pep_max_len, has_meta)

    train_ds = LabeledPairDataset(df_train["cdr3"].tolist(), df_train["peptide"].tolist(), y_tr, meta_tr)
    val_ds   = LabeledPairDataset(df_val["cdr3"].tolist(),   df_val["peptide"].tolist(),   y_va, meta_va)
    test_ds  = LabeledPairDataset(df_test["cdr3"].tolist(),  df_test["peptide"].tolist(),  y_te, meta_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)

    # ── model ─────────────────────────────────────────────────────────────────
    model = GNNBindingClassifier(
        tcr_max_len=args.tcr_max_len, pep_max_len=args.pep_max_len,
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, meta_dim=meta_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %d", n_params)

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_pos = int(df_train["label"].sum())
    n_neg = len(df_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_peptides = df_val["peptide"].tolist()
    test_peptides = df_test["peptide"].tolist()

    best_macro, best_state, bad_epochs = -1.0, None, 0
    metrics_lines = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss   = loss_fn(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * batch["labels"].size(0)
            n       += batch["labels"].size(0)

        val_macro, _, _ = _evaluate_macro(model, val_loader, device, val_peptides)
        line = f"epoch={epoch:3d}  train_loss={running/n:.4f}  val_macro_auc01={val_macro:.4f}"
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
    test_macro, test_y, test_p = _evaluate_macro(model, test_loader, device, test_peptides)
    df_eval = df_test[["peptide"]].copy()
    df_eval["label"] = test_y
    df_eval["score"] = test_p
    _, per_pep = macro_auc01(df_eval)
    pooled = overall_metrics(test_y, test_p)

    logger.info("Test  Macro AUC0.1 = %.4f", test_macro)
    logger.info("Test  Pooled  AUROC=%.4f  AUPRC=%.4f  AUC0.1=%.4f",
                pooled["auroc"], pooled["auprc"], pooled["auc01"])

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "config": {
            "tcr_max_len": args.tcr_max_len,
            "pep_max_len": args.pep_max_len,
            "latent_dim":  args.latent_dim,
            "hidden_dim":  args.hidden_dim,
            "num_layers":  args.num_layers,
            "num_heads":   args.num_heads,
            "dropout":     args.dropout,
            "meta_dim":    meta_dim,
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }, out_dir / "checkpoints" / "checkpoint.pt")

    if aug is not None:
        aug.save(out_dir / "feature_augment")

    per_pep.to_csv(out_dir / "per_peptide.csv", index=False)
    with open(out_dir / "metrics.txt", "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_macro_auc01={best_macro:.4f}\n")
        f.write(f"test_macro_auc01={test_macro:.4f}\n")
        f.write(f"test_pooled_auroc={pooled['auroc']:.4f}\n")
        f.write(f"test_pooled_auprc={pooled['auprc']:.4f}\n")
        f.write(f"test_pooled_auc01={pooled['auc01']:.4f}\n")
        f.write(f"metadata={'paired_one_hot' if args.use_metadata else 'none'}\n")
        f.write(f"n_params={n_params}\n")

    summary = {
        "experiment_name":        out_dir.name,
        "model":                  "gnn_paired",
        "metadata":               "paired_one_hot" if args.use_metadata else "none",
        "n_params":               int(n_params),
        "n_train":                int(len(df_train)),
        "n_val":                  int(len(df_val)),
        "n_test":                 int(len(df_test)),
        "best_val_macro_auc01":   float(best_macro),
        "test_macro_auc01":       test_macro,
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
