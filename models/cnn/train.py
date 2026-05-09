"""
Train the CNN TCR-peptide binding predictor over pre-computed embeddings.

Looks up per-row TCR and peptide vectors from the combined-format embedding
artifacts (the same files MLP/RF read), optionally augments them with V/J +
MHC metadata, and trains a dual-branch 1D-CNN classifier.

Embedding artifacts (one row per unique sequence, looked up via the split CSV):
    outputs/embeddings/one_hot/{tcr,peptide}_embeddings.npy + embedding_index.csv
    outputs/embeddings/autoencoder/plain_ae/{tcr,peptide}_embeddings.npy + index
    outputs/embeddings/autoencoder/vae/{tcr,peptide}_embeddings.npy      + index
    outputs/embeddings/esm/{cdr3,full}_embeddings.npy                     + index
        (ESM only contains TCR vectors; --peptide_embedding is required.)

Run from repo root:
    # one_hot
    python -m models.cnn.train --embedding one_hot

    # plain autoencoder + V/J + MHC (one-hot encoded)
    python -m models.cnn.train --embedding autoencoder --use_feature_augment

    # ESM CDR3 TCR + autoencoder peptide + V/J + MHC (cat-AE encoded)
    python -m models.cnn.train --embedding esm/cdr3 \\
        --peptide_embedding autoencoder/plain_ae \\
        --use_feature_augment --feature_augment_type cat_ae

Outputs (outputs/models/cnn/<run_name>/):
    checkpoints/checkpoint.pt    best model weights + config
    metrics.txt                  per-epoch train loss / val loss / val AUROC
    metrics.json                 final test metrics + run config (matches MLP shape)
    feature_augment/             saved augmenter (only when --use_feature_augment)

The shared file outputs/models/cnn/results_summary.csv is updated in place
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
from torch.utils.data import DataLoader, TensorDataset

from models.cnn.model import EmbeddingCNN
from models.embedding_loader import canonicalize_embedding_name, load_split_separated
from embeddings.feature_augment.one_hot import OneHotFeatureAugmenter
from embeddings.feature_augment.autoencoder import CatAEFeatureAugmenter
from embeddings.one_hot.model import VOCAB_SIZE, DEFAULT_MAX_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/cnn"


# ── Embedding shape config ────────────────────────────────────────────────────

# Per-residue embeddings unfold their flat vector to (channels, seq_len) where
# channels = vocab dim. Vector embeddings are treated as 1 channel × D length.
PER_RESIDUE_LAYOUT = {
    "one_hot": {
        "tcr":     {"in_channels": VOCAB_SIZE, "seq_len": DEFAULT_MAX_LEN["tcr"]},
        "peptide": {"in_channels": VOCAB_SIZE, "seq_len": DEFAULT_MAX_LEN["peptide"]},
    },
}

# Short aliases the user can pass on the command line. The right-hand side is
# the canonical embedding name accepted by models.embedding_loader.
EMBEDDING_ALIASES = {
    "one_hot":     "one_hot",
    "autoencoder": "autoencoder/plain_ae",
    "vae":         "autoencoder/vae",
    "esm/cdr3":    "esm/cdr3",
    "esm/full":    "esm/full",
}


def _resolve_embedding(name: str) -> str:
    """Map CLI name → canonical embedding name used by the loader."""
    if name in EMBEDDING_ALIASES:
        return EMBEDDING_ALIASES[name]
    return canonicalize_embedding_name(name)


def _shape_inputs(embedding: str, tcr_arr: np.ndarray, pep_arr: np.ndarray):
    """
    Reshape flat embedding arrays into (N, C, L) for Conv1D and return
    the in_channels / seq_len config used to build the model.
    """
    if embedding in PER_RESIDUE_LAYOUT:
        layout = PER_RESIDUE_LAYOUT[embedding]
        tcr_cfg = layout["tcr"]
        pep_cfg = layout["peptide"]

        n_t, t_dim = tcr_arr.shape
        n_p, p_dim = pep_arr.shape
        if t_dim != tcr_cfg["in_channels"] * tcr_cfg["seq_len"]:
            raise ValueError(
                f"TCR embedding dim {t_dim} != L*V "
                f"({tcr_cfg['seq_len']}*{tcr_cfg['in_channels']}). "
                f"Did you save with reshape settings the model doesn't expect?"
            )
        if p_dim != pep_cfg["in_channels"] * pep_cfg["seq_len"]:
            raise ValueError(f"Peptide embedding dim {p_dim} != expected "
                             f"{pep_cfg['seq_len']}*{pep_cfg['in_channels']}")

        # (N, L*V) → (N, L, V) → (N, V, L)
        tcr = tcr_arr.reshape(n_t, tcr_cfg["seq_len"], tcr_cfg["in_channels"]).transpose(0, 2, 1)
        pep = pep_arr.reshape(n_p, pep_cfg["seq_len"], pep_cfg["in_channels"]).transpose(0, 2, 1)
        return tcr.copy(), pep.copy(), tcr_cfg, pep_cfg

    # Vector embedding: 1 channel × D length
    tcr_cfg = {"in_channels": 1, "seq_len": tcr_arr.shape[1]}
    pep_cfg = {"in_channels": 1, "seq_len": pep_arr.shape[1]}
    tcr = tcr_arr[:, None, :]  # (N, 1, D)
    pep = pep_arr[:, None, :]
    return tcr, pep, tcr_cfg, pep_cfg


# ── eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    labels_all, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            tcr, pep, meta, labels = _unpack(batch, device)
            logits = model(tcr, pep, meta)
            total_loss += loss_fn(logits, labels).item()
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(torch.sigmoid(logits).cpu().tolist())
    n = len(labels_all)
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.5
    return total_loss / n, auc


def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for batch in loader:
            tcr, pep, meta, labels = _unpack(batch, device)
            logits = model(tcr, pep, meta)
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


def _unpack(batch, device):
    """Move a batch to device. Last tensor is labels; the optional 3rd is meta."""
    if len(batch) == 4:
        tcr, pep, meta, labels = batch
        return tcr.to(device), pep.to(device), meta.to(device), labels.to(device)
    tcr, pep, labels = batch
    return tcr.to(device), pep.to(device), None, labels.to(device)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedding", required=True,
                   choices=sorted(EMBEDDING_ALIASES.keys()),
                   help="TCR embedding to look up. ESM choices require --peptide_embedding.")
    p.add_argument("--peptide_embedding", default=None,
                   help="Peptide embedding source. Defaults to the same source as --embedding "
                        "for non-ESM. Required when --embedding is esm/cdr3 or esm/full because "
                        "ESM artifacts have no peptide vectors. Accepts e.g. one_hot, "
                        "autoencoder/plain_ae, autoencoder/vae.")
    p.add_argument("--splits_dir",     default="data/splits")
    p.add_argument("--embeddings_dir", default="outputs/embeddings")
    p.add_argument("--out",            default=None,
                   help=f"Output dir. Default: {OUT_DIR}/<run_name>")

    # Feature augmentation
    p.add_argument("--use_feature_augment", action="store_true",
                   help="Append V/J + MHC metadata after CNN pooling.")
    p.add_argument("--feature_augment_type", choices=["one_hot", "cat_ae"], default="one_hot")

    # Model
    p.add_argument("--conv_channels",  type=int, default=64)
    p.add_argument("--kernel_size",    type=int, default=5)
    p.add_argument("--branch_out_dim", type=int, default=64)
    p.add_argument("--dropout",        type=float, default=0.1)

    # Training
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=15)

    return p.parse_args()


def _resolve_out_dir(args) -> str:
    if args.out:
        return args.out
    # Slashes are valid in canonical names (esm/cdr3, autoencoder/plain_ae) but
    # not friendly as directory components — flatten with the CLI alias instead.
    base = args.embedding.replace("/", "_")
    suffix = ""
    if args.use_feature_augment:
        suffix = f"_aug-{args.feature_augment_type}"
    return os.path.join(OUT_DIR, f"{base}{suffix}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _resolve_out_dir(args)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    tcr_method = _resolve_embedding(args.embedding)
    pep_method = (canonicalize_embedding_name(args.peptide_embedding)
                  if args.peptide_embedding else None)
    if tcr_method.startswith("esm/") and pep_method is None:
        raise SystemExit(
            "--peptide_embedding is required when --embedding is esm/cdr3 or esm/full "
            "(ESM artifacts contain only TCR vectors). "
            "Try: --peptide_embedding autoencoder/plain_ae"
        )

    logger.info(f"Device: {device}  |  Output: {out_dir}")
    logger.info(f"TCR embedding: {tcr_method}  |  Peptide embedding: "
                f"{pep_method or tcr_method}")

    # ── load embeddings + labels (sequence-keyed lookup) ──────────────────────
    tcr_tr, pep_tr, y_tr_int, df_tr = load_split_separated(
        "train", tcr_method, pep_method,
        splits_dir=args.splits_dir, embeddings_dir=args.embeddings_dir,
        drop_missing=True,
    )
    tcr_va, pep_va, y_va_int, df_va = load_split_separated(
        "val",   tcr_method, pep_method,
        splits_dir=args.splits_dir, embeddings_dir=args.embeddings_dir,
        drop_missing=True,
    )
    tcr_te, pep_te, y_te_int, df_te = load_split_separated(
        "test",  tcr_method, pep_method,
        splits_dir=args.splits_dir, embeddings_dir=args.embeddings_dir,
        drop_missing=True,
    )

    # The reshape helper keys off the CLI name so it knows when to unfold one_hot.
    tcr_tr, pep_tr, tcr_cfg, pep_cfg = _shape_inputs(args.embedding, tcr_tr, pep_tr)
    tcr_va, pep_va, _,       _       = _shape_inputs(args.embedding, tcr_va, pep_va)
    tcr_te, pep_te, _,       _       = _shape_inputs(args.embedding, tcr_te, pep_te)
    logger.info(f"  Conv inputs: TCR (C={tcr_cfg['in_channels']}, L={tcr_cfg['seq_len']})  "
                f"pep (C={pep_cfg['in_channels']}, L={pep_cfg['seq_len']})")

    y_tr = y_tr_int.astype(np.float32)
    y_va = y_va_int.astype(np.float32)
    y_te = y_te_int.astype(np.float32)

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

    # ── tensors / loaders ─────────────────────────────────────────────────────
    def make_loader(tcr, pep, meta, y, shuffle):
        tensors = [
            torch.from_numpy(tcr).float(),
            torch.from_numpy(pep).float(),
        ]
        if meta is not None:
            tensors.append(torch.from_numpy(meta).float())
        tensors.append(torch.from_numpy(y).float())
        return DataLoader(TensorDataset(*tensors), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader(tcr_tr, pep_tr, meta_tr, y_tr, shuffle=True)
    val_loader   = make_loader(tcr_va, pep_va, meta_va, y_va, shuffle=False)
    test_loader  = make_loader(tcr_te, pep_te, meta_te, y_te, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = EmbeddingCNN(
        tcr_in_channels=tcr_cfg["in_channels"], tcr_seq_len=tcr_cfg["seq_len"],
        pep_in_channels=pep_cfg["in_channels"], pep_seq_len=pep_cfg["seq_len"],
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        branch_out_dim=args.branch_out_dim,
        meta_dim=meta_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=max(1, args.patience // 2), factor=0.5)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── train loop ────────────────────────────────────────────────────────────
    best_auc, best_state, bad_epochs = 0.0, None, 0
    metrics_lines = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            tcr, pep, meta, labels = _unpack(batch, device)
            logits = model(tcr, pep, meta)
            loss   = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * labels.size(0)
            n       += labels.size(0)

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

    # ── test eval (best checkpoint) ───────────────────────────────────────────
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
            "tcr_in_channels": tcr_cfg["in_channels"],
            "tcr_seq_len":     tcr_cfg["seq_len"],
            "pep_in_channels": pep_cfg["in_channels"],
            "pep_seq_len":     pep_cfg["seq_len"],
            "conv_channels":   args.conv_channels,
            "kernel_size":     args.kernel_size,
            "branch_out_dim":  args.branch_out_dim,
            "meta_dim":        meta_dim,
            "dropout":         args.dropout,
        },
        "args": vars(args),
    }, ckpt_path)

    if aug is not None and hasattr(aug, "save"):
        aug.save(os.path.join(out_dir, "feature_augment"))

    pep_label = pep_method or tcr_method
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_auc={best_auc:.4f}\n")
        f.write(f"embedding={args.embedding}\n")
        f.write(f"tcr_embedding={tcr_method}\n")
        f.write(f"peptide_embedding={pep_label}\n")
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
        "experiment_name":   experiment_name,
        "embedding":         args.embedding,
        "tcr_embedding":     tcr_method,
        "peptide_embedding": pep_label,
        "feature_augment":   feature_augment,
        "n_params":          int(n_params),
        "n_train":           int(len(df_tr)),
        "n_val":             int(len(df_va)),
        "n_test":            int(len(df_te)),
        "best_val_auroc":    float(best_auc),
        "auroc":             test_metrics["auroc"],
        "auprc":             test_metrics["auprc"],
        "accuracy":          test_metrics["accuracy"],
        "f1":                test_metrics["f1"],
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({**summary_row, "model_type": "cnn"}, f, indent=2, sort_keys=True)

    _update_results_summary(OUT_DIR, summary_row)

    logger.info(f"Best val AUC: {best_auc:.4f}  |  Saved to {out_dir}")


def _update_results_summary(root: str, row: dict) -> None:
    """Read outputs/models/cnn/results_summary.csv, replace any existing row
    with the same experiment_name, append this run's row, write back."""
    path = os.path.join(root, "results_summary.csv")
    columns = list(row.keys())
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = df[df["experiment_name"] != row["experiment_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    # Keep our column order, then any pre-existing extra cols at the end
    extra = [c for c in df.columns if c not in columns]
    df = df[columns + extra]
    os.makedirs(root, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Updated {path}")


if __name__ == "__main__":
    main()
