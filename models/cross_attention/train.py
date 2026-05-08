"""
Train the cross-attention TCR-peptide binding predictor.

Expects pre-split, pre-labeled CSVs (cdr3, peptide, label, [v_gene, j_gene, mhc_class, ...]).
Data splitting and negative generation are handled upstream by a separate module.

Run from repo root:
    # raw sequence input (TCR side uses learned AA embedding)
    python -m models.cross_attention.train
    python -m models.cross_attention.train --use_metadata --metadata_type cat_ae

    # ESM CDR3 input (TCR side uses per-residue ESM hidden states; peptide side
    # still uses the learned AA embedding). Requires
    #   outputs/embeddings/esm/cdr3_per_residue.npy
    #   outputs/embeddings/esm/cdr3_lengths.npy
    #   outputs/embeddings/esm/embedding_index.csv
    # produced by `embeddings/esm/generate_embeddings.py --save_per_residue`.
    python -m models.cross_attention.train --esm_tcr
    python -m models.cross_attention.train --esm_tcr --use_metadata --metadata_type cat_ae

Outputs (outputs/models/cross_attention/<run_name>/):
    checkpoints/checkpoint.pt    best model weights + config
    metrics.txt                  per-epoch losses + final test metrics
    metrics.json                 final test metrics + run config
    feature_augment/             saved augmenter (only when --use_metadata)

The shared file outputs/models/cross_attention/results_summary.csv is
updated in place after each run with one row per experiment.
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

    def __init__(self, df: pd.DataFrame, tcr_feature_idx: np.ndarray | None = None):
        self.tcrs   = df["cdr3"].tolist()
        self.peps   = df["peptide"].tolist()
        self.labels = df["label"].astype(float).tolist()
        # When the TCR side consumes pre-computed features, this is a parallel
        # array of int row indices into a (N_unique, L, D) feature tensor.
        self.tcr_feature_idx = tcr_feature_idx

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, i):
        if self.tcr_feature_idx is not None:
            return i, int(self.tcr_feature_idx[i]), self.peps[i], self.labels[i]
        return i, self.tcrs[i], self.peps[i], self.labels[i]


# ── ESM per-residue lookup ────────────────────────────────────────────────────

class ESMTcrLookup:
    """First-occurrence cdr3 → (per-residue feature row, length) lookup."""

    def __init__(self, esm_dir: str):
        index_path     = os.path.join(esm_dir, "embedding_index.csv")
        per_residue    = os.path.join(esm_dir, "cdr3_per_residue.npy")
        lengths_path   = os.path.join(esm_dir, "cdr3_lengths.npy")
        for p in (index_path, per_residue, lengths_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing {p}. Run `python embeddings/esm/generate_embeddings.py "
                    f"--save_per_residue` first."
                )

        idx = pd.read_csv(index_path, index_col=0)
        if "cdr3" not in idx.columns:
            raise ValueError(f"{index_path} is missing required column 'cdr3'")

        self.features = np.load(per_residue)         # (N_unique, L, D) float16
        self.lengths  = np.load(lengths_path)        # (N_unique,) int32
        self.feature_dim = int(self.features.shape[2])
        self.max_len     = int(self.features.shape[1])

        seen: dict[str, int] = {}
        for row, cdr3 in enumerate(idx["cdr3"].astype(str).tolist()):
            if cdr3 not in seen and self.lengths[row] > 0:
                seen[cdr3] = row
        self.cdr3_to_row = seen
        logger.info(
            "ESM CDR3 lookup: %d unique cdr3s, features (%d,%d,%d) float16",
            len(self.cdr3_to_row), len(self.features), self.max_len, self.feature_dim,
        )

    def __contains__(self, cdr3: str) -> bool:
        return cdr3 in self.cdr3_to_row

    def filter_df(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        keep = df["cdr3"].astype(str).isin(self.cdr3_to_row).to_numpy()
        n_drop = (~keep).sum()
        if n_drop:
            logger.warning(
                "Dropping %d/%d rows from %s missing ESM CDR3 features",
                int(n_drop), len(df), split,
            )
        return df.loc[keep].reset_index(drop=True)

    def row_indices(self, df: pd.DataFrame) -> np.ndarray:
        return df["cdr3"].astype(str).map(self.cdr3_to_row).to_numpy(dtype=np.int64)


# ── dataloaders ───────────────────────────────────────────────────────────────

def make_loader(df: pd.DataFrame, meta: np.ndarray | None, batch_size: int,
                shuffle: bool, device: str,
                tcr_lookup: ESMTcrLookup | None = None,
                tcr_feature_idx: np.ndarray | None = None) -> DataLoader:
    ds = LabeledDataset(df, tcr_feature_idx=tcr_feature_idx)
    use_features = tcr_lookup is not None

    def collate(batch):
        idxs   = [b[0] for b in batch]
        peps   = [b[2] for b in batch]
        labels = [b[3] for b in batch]
        pi, pm = collate_sequences(peps, PEP_MAX_LEN, device)
        lbl = torch.tensor(labels, dtype=torch.float32, device=device)
        m = (torch.tensor(meta[idxs], dtype=torch.float32, device=device)
             if meta is not None else None)

        if use_features:
            feat_rows = np.asarray([b[1] for b in batch], dtype=np.int64)
            feats = tcr_lookup.features[feat_rows]                    # (B, L, D) float16
            tlen  = tcr_lookup.lengths[feat_rows]                     # (B,)
            ti = torch.from_numpy(feats.astype(np.float32, copy=False)).to(device)
            tm = torch.zeros((len(batch), tcr_lookup.max_len),
                             dtype=torch.float32, device=device)
            for j, L in enumerate(tlen):
                tm[j, :int(L)] = 1.0
        else:
            tcrs = [b[1] for b in batch]
            ti, tm = collate_sequences(tcrs, TCR_MAX_LEN, device)
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


def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for ti, pi, tm, pm, labels, meta in loader:
            logits = model(ti, pi, tm, pm, meta)
            labels_all.append(labels.cpu().numpy())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


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
    p.add_argument("--train",    default="data/splits/train.csv",
                   help="CSV with columns: cdr3, peptide, label (and optionally v_gene, j_gene, mhc_class)")
    p.add_argument("--val",      default="data/splits/val.csv",
                   help="Same format as --train")
    p.add_argument("--test",     default="data/splits/test.csv")
    p.add_argument("--out",      default=None,
                   help=f"Output dir. Default: {OUT_DIR}/<run_name>")
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
    p.add_argument("--esm_tcr", action="store_true",
                   help="Feed per-residue ESM CDR3 hidden states into the TCR side "
                        "instead of learning AA embeddings. Peptide side still uses "
                        "the learned AA embedding.")
    p.add_argument("--esm_dir", default="outputs/embeddings/esm",
                   help="Directory containing cdr3_per_residue.npy / cdr3_lengths.npy "
                        "/ embedding_index.csv. Ignored unless --esm_tcr is set.")
    return p.parse_args()


def _resolve_out_dir(args) -> str:
    if args.out:
        return args.out
    src    = "esm-cdr3" if args.esm_tcr else "raw"
    suffix = f"_aug-{args.metadata_type}" if args.use_metadata else ""
    return os.path.join(OUT_DIR, f"cross_attention_{src}{suffix}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = _resolve_out_dir(args)
    logger.info(f"Device: {device}  |  Output: {out_dir}")

    df_train = pd.read_csv(args.train, low_memory=False)
    df_val   = pd.read_csv(args.val,   low_memory=False)
    df_test  = pd.read_csv(args.test,  low_memory=False)
    logger.info(f"Train: {len(df_train)} rows  |  Val: {len(df_val)} rows  |  Test: {len(df_test)} rows")

    for col in ("cdr3", "peptide", "label"):
        for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            if col not in df.columns:
                raise ValueError(f"--{name} CSV is missing required column '{col}'")

    # ── ESM CDR3 lookup (drops rows whose CDR3 lacks ESM coverage) ────────────
    tcr_lookup = None
    tcr_feature_dim = 0
    if args.esm_tcr:
        tcr_lookup = ESMTcrLookup(args.esm_dir)
        tcr_feature_dim = tcr_lookup.feature_dim
        df_train = tcr_lookup.filter_df(df_train, "train")
        df_val   = tcr_lookup.filter_df(df_val,   "val")
        df_test  = tcr_lookup.filter_df(df_test,  "test")

    # ── metadata ──────────────────────────────────────────────────────────────
    meta_train = meta_val = meta_test = None
    meta_dim = 0
    aug = None
    if args.use_metadata:
        aug = CatAEFeatureAugmenter() if args.metadata_type == "cat_ae" else OneHotFeatureAugmenter()
        aug.fit(df_train)
        meta_train = aug.transform(df_train).astype(np.float32)
        meta_val   = aug.transform(df_val).astype(np.float32)
        meta_test  = aug.transform(df_test).astype(np.float32)
        meta_dim   = aug.feature_dim
        logger.info(f"Metadata: {args.metadata_type}  dim={meta_dim}  {aug.feature_breakdown()}")

    # ── per-row TCR feature row indices when in ESM mode ──────────────────────
    tr_feat = va_feat = te_feat = None
    if tcr_lookup is not None:
        tr_feat = tcr_lookup.row_indices(df_train)
        va_feat = tcr_lookup.row_indices(df_val)
        te_feat = tcr_lookup.row_indices(df_test)

    # ── loaders ───────────────────────────────────────────────────────────────
    train_loader = make_loader(df_train, meta_train, args.batch_size, shuffle=True,
                               device=device, tcr_lookup=tcr_lookup, tcr_feature_idx=tr_feat)
    val_loader   = make_loader(df_val,   meta_val,   args.batch_size, shuffle=False,
                               device=device, tcr_lookup=tcr_lookup, tcr_feature_idx=va_feat)
    test_loader  = make_loader(df_test,  meta_test,  args.batch_size, shuffle=False,
                               device=device, tcr_lookup=tcr_lookup, tcr_feature_idx=te_feat)

    # ── model ─────────────────────────────────────────────────────────────────
    model = CrossAttentionTCRPep(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, meta_dim=meta_dim,
        tcr_feature_dim=tcr_feature_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, args.patience // 2), factor=0.5)
    n_pos = int(df_train["label"].sum())
    n_neg = len(df_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state, bad_epochs = 0.0, None, 0
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
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
            "d_model":         args.d_model,
            "n_heads":         args.n_heads,
            "n_layers":        args.n_layers,
            "dropout":         args.dropout,
            "meta_dim":        meta_dim,
            "tcr_feature_dim": tcr_feature_dim,
        },
        "args": vars(args),
    }, ckpt_path)

    if aug is not None and hasattr(aug, "save"):
        aug.save(os.path.join(out_dir, "feature_augment"))

    metadata_label = args.metadata_type if args.use_metadata else "none"
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_auc={best_auc:.4f}\n")
        f.write(f"metadata={metadata_label}\n")
        f.write(f"n_params={n_params}\n")
        f.write(f"test_auroc={test_metrics['auroc']:.4f}\n")
        f.write(f"test_auprc={test_metrics['auprc']:.4f}\n")
        f.write(f"test_accuracy={test_metrics['accuracy']:.4f}\n")
        f.write(f"test_f1={test_metrics['f1']:.4f}\n")

    experiment_name = os.path.basename(out_dir.rstrip("/"))
    summary_row = {
        "experiment_name":  experiment_name,
        "tcr_input":        "esm/cdr3" if args.esm_tcr else "raw",
        "metadata":         metadata_label,
        "n_params":         int(n_params),
        "n_train":          int(len(df_train)),
        "n_val":            int(len(df_val)),
        "n_test":           int(len(df_test)),
        "best_val_auroc":   float(best_auc),
        "auroc":            test_metrics["auroc"],
        "auprc":            test_metrics["auprc"],
        "accuracy":         test_metrics["accuracy"],
        "f1":               test_metrics["f1"],
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({**summary_row, "model_type": "cross_attention"}, f, indent=2, sort_keys=True)

    _update_results_summary(OUT_DIR, summary_row)

    logger.info(f"Best val AUC: {best_auc:.4f}  |  Saved to {out_dir}")


def _update_results_summary(root: str, row: dict) -> None:
    """Row-replace update of results_summary.csv keyed by experiment_name."""
    path = os.path.join(root, "results_summary.csv")
    columns = list(row.keys())
    if os.path.exists(path):
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
