"""
Train the relational-GAT TCR-peptide binding predictor on the project splits.

Reads raw (cdr3, peptide, label) rows from data/splits/{train,val,test}.csv,
builds per-pair graphs (TCR backbone + peptide backbone + bipartite contact
edges), and trains TCRPeptideGNN end-to-end as a binding classifier. Writes
checkpoint, metrics.json, and a row in outputs/models/gnn/results_summary.csv
matching the MLP/CNN format.

The "graph embedding" the user-facing predictor consumes is produced inside
the GNN forward pass (per-pair masked-mean pooled R-GAT features). There is
no need to precompute it — the script trains the GNN directly on the raw
sequence pairs.

Run from repo root:
    # plain
    python -m models.gnn.train

    # with V/J + MHC metadata (one-hot)
    python -m models.gnn.train --use_feature_augment

    # with V/J + MHC metadata (cat-AE)
    python -m models.gnn.train --use_feature_augment --feature_augment_type cat_ae

Outputs (outputs/models/gnn/<run_name>/):
    checkpoints/checkpoint.pt    state_dict + config + args
    metrics.txt                  per-epoch losses + final test metrics
    metrics.json                 final test metrics + run config
    feature_augment/             saved augmenter (only when --use_feature_augment)

The shared file outputs/models/gnn/results_summary.csv is updated in place
after each run with one row per experiment.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from embeddings.graph.model import TCRPeptideGNN, build_pair_tensors
from embeddings.feature_augment.one_hot import OneHotFeatureAugmenter
from embeddings.feature_augment.autoencoder import CatAEFeatureAugmenter
from embeddings.one_hot.model import DEFAULT_MAX_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/gnn"


# ── Dataset ───────────────────────────────────────────────────────────────────

class LabeledPairDataset(Dataset):
    """Holds (cdr3, peptide, label[, meta]) rows. Graph tensors are built per batch."""

    def __init__(
        self,
        tcrs: list[str],
        peps: list[str],
        labels: np.ndarray,
        meta: np.ndarray | None = None,
    ):
        assert len(tcrs) == len(peps) == len(labels)
        if meta is not None:
            assert len(meta) == len(tcrs)
        self.tcrs   = tcrs
        self.peps   = peps
        self.labels = labels
        self.meta   = meta

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, i: int):
        item = (self.tcrs[i], self.peps[i], float(self.labels[i]))
        if self.meta is not None:
            return (*item, self.meta[i])
        return item


def make_collate(tcr_max_len: int, pep_max_len: int, has_meta: bool):
    def collate(batch):
        tcrs   = [b[0] for b in batch]
        peps   = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        out = build_pair_tensors(tcrs, peps, tcr_max_len, pep_max_len)
        out["labels"] = labels
        if has_meta:
            out["meta"] = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        return out
    return collate


# ── Wrapper model with optional metadata head ─────────────────────────────────

class GNNBindingClassifier(nn.Module):
    """TCRPeptideGNN encoder + (optional) metadata-aware classifier head."""

    def __init__(
        self,
        tcr_max_len: int,
        pep_max_len: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        meta_dim: int = 0,
    ):
        super().__init__()
        self.gnn = TCRPeptideGNN(
            tcr_max_len=tcr_max_len, pep_max_len=pep_max_len,
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
        )
        self.meta_dim = meta_dim
        head_in = 2 * latent_dim + meta_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        pair_emb, _, _ = self.gnn.encode(batch)
        if self.meta_dim > 0:
            pair_emb = torch.cat([pair_emb, batch["meta"]], dim=-1)
        return self.head(pair_emb).squeeze(-1)


# ── Eval ──────────────────────────────────────────────────────────────────────

def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            labels_all.append(batch["labels"].cpu().numpy())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def evaluate(model, loader, device) -> tuple[float, float]:
    """Return (mean BCE loss, AUROC)."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, n = 0.0, 0
    labels_all, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            total_loss += loss_fn(logits, batch["labels"]).item()
            n += batch["labels"].size(0)
            labels_all.append(batch["labels"].cpu().numpy())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(labels_all)
    p = np.concatenate(probs_all)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    return total_loss / max(1, n), auc


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    has_both = len(np.unique(y_true)) > 1
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if has_both else 0.5,
        "auprc": (float(average_precision_score(y_true, y_prob))
                  if has_both else float(y_true.mean())),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_csv", default="data/splits/train.csv")
    p.add_argument("--val_csv",   default="data/splits/val.csv")
    p.add_argument("--test_csv",  default="data/splits/test.csv")
    p.add_argument("--out",       default=None,
                   help=f"Output dir. Default: {OUT_DIR}/<run_name>")
    p.add_argument("--run_name",  default=None,
                   help="Subdir name. Default: gnn[_aug-<type>]")

    # Feature augmentation
    p.add_argument("--use_feature_augment", action="store_true",
                   help="Append V/J + MHC metadata after GNN pooling.")
    p.add_argument("--feature_augment_type", choices=["one_hot", "cat_ae"], default="one_hot")

    # Model
    p.add_argument("--latent_dim", type=int,   default=64)
    p.add_argument("--hidden_dim", type=int,   default=64)
    p.add_argument("--num_layers", type=int,   default=3)
    p.add_argument("--num_heads",  type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.1)

    p.add_argument("--tcr_max_len", type=int, default=DEFAULT_MAX_LEN["tcr"])
    p.add_argument("--pep_max_len", type=int, default=DEFAULT_MAX_LEN["peptide"])

    # Training
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=5)
    p.add_argument("--num_workers", type=int,  default=0)
    return p.parse_args()


def _resolve_out_dir(args) -> str:
    if args.out:
        return args.out
    name = args.run_name
    if not name:
        suffix = f"_aug-{args.feature_augment_type}" if args.use_feature_augment else ""
        name = f"gnn{suffix}"
    return os.path.join(OUT_DIR, name)


def _load_split(path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    needed = {"cdr3", "peptide", "label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"--{split}_csv ({path}) is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["cdr3", "peptide"]).reset_index(drop=True)
    logger.info(f"  {split}: {len(df)} rows")
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _resolve_out_dir(args)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    logger.info(f"Device: {device}  |  Output: {out_dir}")

    # ── load splits ───────────────────────────────────────────────────────────
    df_tr = _load_split(args.train_csv, "train")
    df_va = _load_split(args.val_csv,   "val")
    df_te = _load_split(args.test_csv,  "test")

    y_tr = df_tr["label"].astype(np.float32).to_numpy()
    y_va = df_va["label"].astype(np.float32).to_numpy()
    y_te = df_te["label"].astype(np.float32).to_numpy()

    # ── feature augmentation (V/J + MHC) ──────────────────────────────────────
    meta_tr = meta_va = meta_te = None
    meta_dim = 0
    aug = None
    if args.use_feature_augment:
        aug = (CatAEFeatureAugmenter() if args.feature_augment_type == "cat_ae"
               else OneHotFeatureAugmenter())
        aug.fit(df_tr)
        meta_tr  = aug.transform(df_tr).astype(np.float32)
        meta_va  = aug.transform(df_va).astype(np.float32)
        meta_te  = aug.transform(df_te).astype(np.float32)
        meta_dim = aug.feature_dim
        logger.info(f"Feature augment: {args.feature_augment_type}  "
                    f"dim={meta_dim}  {aug.feature_breakdown()}")

    # ── datasets / loaders ────────────────────────────────────────────────────
    has_meta = meta_dim > 0
    collate  = make_collate(args.tcr_max_len, args.pep_max_len, has_meta)

    train_ds = LabeledPairDataset(df_tr["cdr3"].tolist(), df_tr["peptide"].tolist(), y_tr, meta_tr)
    val_ds   = LabeledPairDataset(df_va["cdr3"].tolist(), df_va["peptide"].tolist(), y_va, meta_va)
    test_ds  = LabeledPairDataset(df_te["cdr3"].tolist(), df_te["peptide"].tolist(), y_te, meta_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate)

    # ── model ─────────────────────────────────────────────────────────────────
    model = GNNBindingClassifier(
        tcr_max_len=args.tcr_max_len, pep_max_len=args.pep_max_len,
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, meta_dim=meta_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=max(1, args.patience // 2), factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    # ── train loop ────────────────────────────────────────────────────────────
    best_auc, best_state, bad_epochs = 0.0, None, 0
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

        val_loss, val_auc = evaluate(model, val_loader, device)
        sched.step(val_loss)
        line = (f"epoch={epoch:3d}  train_loss={running/n:.4f}  "
                f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")
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

    # ── test eval ─────────────────────────────────────────────────────────────
    test_labels, test_probs = predict_proba(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs)
    logger.info(
        "Test  AUROC=%.4f AUPRC=%.4f ACC=%.4f F1=%.4f",
        test_metrics["auroc"], test_metrics["auprc"],
        test_metrics["accuracy"], test_metrics["f1"],
    )

    # ── save ──────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "checkpoints", "checkpoint.pt")
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
        "args": vars(args),
    }, ckpt_path)

    if aug is not None and hasattr(aug, "save"):
        aug.save(os.path.join(out_dir, "feature_augment"))

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_auc={best_auc:.4f}\n")
        f.write(f"feature_augment="
                f"{args.feature_augment_type if args.use_feature_augment else 'none'}\n")
        f.write(f"n_params={n_params}\n")
        f.write(f"test_auroc={test_metrics['auroc']:.4f}\n")
        f.write(f"test_auprc={test_metrics['auprc']:.4f}\n")
        f.write(f"test_accuracy={test_metrics['accuracy']:.4f}\n")
        f.write(f"test_f1={test_metrics['f1']:.4f}\n")

    experiment_name = os.path.basename(out_dir.rstrip("/"))
    feature_augment = (args.feature_augment_type
                       if args.use_feature_augment else "none")
    summary_row = {
        "experiment_name":  experiment_name,
        "feature_augment":  feature_augment,
        "n_params":         int(n_params),
        "n_train":          int(len(df_tr)),
        "n_val":            int(len(df_va)),
        "n_test":           int(len(df_te)),
        "best_val_auroc":   float(best_auc),
        "auroc":            test_metrics["auroc"],
        "auprc":            test_metrics["auprc"],
        "accuracy":         test_metrics["accuracy"],
        "f1":               test_metrics["f1"],
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({**summary_row, "model_type": "gnn"}, f, indent=2, sort_keys=True)

    _update_results_summary(OUT_DIR, summary_row)

    logger.info(f"Best val AUC: {best_auc:.4f}  |  Saved to {out_dir}")


def _update_results_summary(root: str, row: dict) -> None:
    """Read outputs/models/gnn/results_summary.csv, replace any existing row
    with the same experiment_name, append this run's row, write back."""
    path = os.path.join(root, "results_summary.csv")
    columns = list(row.keys())
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = df[df["experiment_name"] != row["experiment_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    extra = [c for c in df.columns if c not in columns]
    df = df[columns + extra]
    os.makedirs(root, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Updated {path}")


if __name__ == "__main__":
    main()
