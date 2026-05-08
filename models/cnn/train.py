"""
Train the CNN TCR-peptide binding predictor over pre-computed embeddings.

Reads pre-saved per-split embeddings and pre-labeled split CSVs, optionally
augments them with V/J + MHC metadata, and trains a dual-branch 1D-CNN
classifier.

Expected layout (default --embedding_dir is outputs/embeddings/<embedding>):
    outputs/embeddings/<embedding>/train/tcr_embeddings.npy
    outputs/embeddings/<embedding>/train/peptide_embeddings.npy
    outputs/embeddings/<embedding>/train/embedding_index.csv     (cdr3, peptide order check)
    outputs/embeddings/<embedding>/val/...                        (same)

Run from repo root:
    # one_hot only
    python -m models.cnn.train --embedding one_hot

    # autoencoder + V/J + MHC (one-hot encoded)
    python -m models.cnn.train --embedding autoencoder --use_feature_augment

    # ESM + V/J + MHC (cat-AE encoded)
    python -m models.cnn.train --embedding esm \\
        --use_feature_augment --feature_augment_type cat_ae

Outputs (outputs/models/cnn/<run_name>/):
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
from torch.utils.data import DataLoader, TensorDataset

from models.cnn.model import EmbeddingCNN
from embeddings.feature_augment.one_hot import OneHotFeatureAugmenter
from embeddings.feature_augment.autoencoder import CatAEFeatureAugmenter
from embeddings.one_hot.model import VOCAB_SIZE, DEFAULT_MAX_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/cnn"


# ── Embedding shape config ────────────────────────────────────────────────────

# For each supported embedding, describe how to interpret its flattened tensor
# as a (channels, seq_len) input to a Conv1D.
#   one_hot:    flat (L*V,) → reshape to (L, V) → permute to (V, L) so channels=V.
#   autoencoder/esm: vector (D,) → unsqueeze to (1, D) so channels=1.
EMBEDDING_LAYOUTS = {
    "one_hot": {
        "tcr":     {"in_channels": VOCAB_SIZE, "seq_len": DEFAULT_MAX_LEN["tcr"]},
        "peptide": {"in_channels": VOCAB_SIZE, "seq_len": DEFAULT_MAX_LEN["peptide"]},
        "per_residue": True,
    },
    "autoencoder": {"per_residue": False},
    "vae":         {"per_residue": False},
    "esm":         {"per_residue": False},
}


def _load_embedding_array(path_dir: str, kind: str) -> np.ndarray:
    """
    Load tcr_embeddings.npy or peptide_embeddings.npy from a split dir.
    Falls back to cdr3_embeddings.npy / full_embeddings.npy for ESM-style layouts.
    """
    candidates = {
        "tcr":     ["tcr_embeddings.npy", "cdr3_embeddings.npy"],
        "peptide": ["peptide_embeddings.npy", "full_embeddings.npy"],
    }[kind]
    for name in candidates:
        p = os.path.join(path_dir, name)
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(
        f"None of {candidates} found in {path_dir}. "
        f"Generate the embeddings first."
    )


def _shape_inputs(embedding: str, tcr_arr: np.ndarray, pep_arr: np.ndarray):
    """
    Reshape flat embedding arrays into (N, C, L) for Conv1D and return
    the in_channels / seq_len config used to build the model.
    """
    layout = EMBEDDING_LAYOUTS[embedding]

    if layout["per_residue"]:
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

    # Vector embedding: treat as 1 channel × D length
    tcr_cfg = {"in_channels": 1, "seq_len": tcr_arr.shape[1]}
    pep_cfg = {"in_channels": 1, "seq_len": pep_arr.shape[1]}
    tcr = tcr_arr[:, None, :]  # (N, 1, D)
    pep = pep_arr[:, None, :]
    return tcr, pep, tcr_cfg, pep_cfg


def _check_alignment(split_df: pd.DataFrame, index_path: str, split_name: str) -> None:
    """Verify embedding rows match the split CSV row-for-row on (cdr3, peptide)."""
    if not os.path.exists(index_path):
        logger.warning(f"  No embedding_index.csv at {index_path}; skipping alignment check.")
        return
    idx = pd.read_csv(index_path, index_col=0)
    if len(idx) != len(split_df):
        raise ValueError(f"[{split_name}] index has {len(idx)} rows, split has {len(split_df)}")
    if not (idx["cdr3"].values == split_df["cdr3"].values).all():
        raise ValueError(f"[{split_name}] cdr3 column mismatch between embedding index and split CSV")
    if not (idx["peptide"].values == split_df["peptide"].values).all():
        raise ValueError(f"[{split_name}] peptide column mismatch between embedding index and split CSV")


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
    p.add_argument("--embedding", required=True, choices=sorted(EMBEDDING_LAYOUTS.keys()),
                   help="Which embedding to read from outputs/embeddings/<embedding>/")
    p.add_argument("--embedding_dir", default=None,
                   help="Override the default outputs/embeddings/<embedding>/ root.")
    p.add_argument("--train_csv", default="data/splits/train.csv")
    p.add_argument("--val_csv",   default="data/splits/val.csv")
    p.add_argument("--out",       default=None,
                   help=f"Output dir. Default: {OUT_DIR}/<embedding>[_aug-<type>]")

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
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=5)

    return p.parse_args()


def _resolve_out_dir(args) -> str:
    if args.out:
        return args.out
    suffix = ""
    if args.use_feature_augment:
        suffix = f"_aug-{args.feature_augment_type}"
    return os.path.join(OUT_DIR, f"{args.embedding}{suffix}")


def _load_split(args, split: str, split_csv: str):
    """Load embeddings + label CSV for one split, return (tcr_arr, pep_arr, df)."""
    embed_root = args.embedding_dir or os.path.join("outputs/embeddings", args.embedding)
    split_dir  = os.path.join(embed_root, split)

    df = pd.read_csv(split_csv, low_memory=False)
    if "label" not in df.columns:
        raise ValueError(f"--{split}_csv ({split_csv}) is missing required 'label' column")

    tcr_arr = _load_embedding_array(split_dir, "tcr")
    pep_arr = _load_embedding_array(split_dir, "peptide")
    if len(tcr_arr) != len(df) or len(pep_arr) != len(df):
        raise ValueError(
            f"[{split}] embedding row count ({len(tcr_arr)} TCR / {len(pep_arr)} pep) "
            f"!= split rows ({len(df)})"
        )

    _check_alignment(df, os.path.join(split_dir, "embedding_index.csv"), split)
    logger.info(f"  {split}: {len(df)} rows  TCR{tcr_arr.shape}  pep{pep_arr.shape}")
    return tcr_arr, pep_arr, df


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = _resolve_out_dir(args)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    logger.info(f"Device: {device}  |  Output: {out_dir}")

    # ── load embeddings + labels ──────────────────────────────────────────────
    logger.info(f"Loading {args.embedding} embeddings …")
    tcr_tr, pep_tr, df_tr = _load_split(args, "train", args.train_csv)
    tcr_va, pep_va, df_va = _load_split(args, "val",   args.val_csv)

    # ── reshape inputs for Conv1D ─────────────────────────────────────────────
    tcr_tr, pep_tr, tcr_cfg, pep_cfg = _shape_inputs(args.embedding, tcr_tr, pep_tr)
    tcr_va, pep_va, _,       _       = _shape_inputs(args.embedding, tcr_va, pep_va)
    logger.info(f"  Conv inputs: TCR (C={tcr_cfg['in_channels']}, L={tcr_cfg['seq_len']})  "
                f"pep (C={pep_cfg['in_channels']}, L={pep_cfg['seq_len']})")

    y_tr = df_tr["label"].astype(np.float32).to_numpy(copy=True)
    y_va = df_va["label"].astype(np.float32).to_numpy(copy=True)

    # ── feature augmentation (V/J + MHC) ──────────────────────────────────────
    meta_tr = meta_va = None
    meta_dim = 0
    if args.use_feature_augment:
        aug = (CatAEFeatureAugmenter() if args.feature_augment_type == "cat_ae"
               else OneHotFeatureAugmenter())
        aug.fit(df_tr)
        meta_tr  = aug.transform(df_tr).astype(np.float32)
        meta_va  = aug.transform(df_va).astype(np.float32)
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

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        f.write(f"\nbest_val_auc={best_auc:.4f}\n")
        f.write(f"embedding={args.embedding}\n")
        f.write(f"feature_augment="
                f"{args.feature_augment_type if args.use_feature_augment else 'none'}\n")
        f.write(f"n_params={n_params}\n")

    logger.info(f"Best val AUC: {best_auc:.4f}  |  Saved to {out_dir}")


if __name__ == "__main__":
    main()
