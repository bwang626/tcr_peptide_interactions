"""Train the cross-attention model on the IMMREP23 paired-chain dataset.

Two input modes:

    Token mode (default):
        TCR side = CDR3α + CDR3β concatenated (max 40 residues), encoded
        with the model's learned AA embedding.

    ESM mode (--esm_tcr):
        TCR side = per-residue ESMplusplus_large hidden states for the
        same CDR3α + CDR3β concatenation. Run `python -m immrep23.embed_esm`
        first to produce the per-residue cache under
        outputs/embeddings/esm_immrep23/{train,test}/.

Optional metadata: --use_metadata appends a paired V/J + HLA encoding
(--metadata_type one_hot or cat_ae). cat_ae mirrors the best-performing
config from the main pipeline benchmark.

Pipeline:
    1. Load IMMREP23 train (positives only) and generate negatives — either
       pass --train pointing at a pre-built combined CSV, or pass
       --raw_train pointing at the raw VDJdb_paired_chain.csv and let this
       script generate negatives in-memory.
    2. Per-peptide stratified train/val split.
    3. Train with early stopping on val Macro AUC0.1.
    4. Evaluate on the IMMREP23 test set with per-peptide breakdown.

Run from repo root:

    # raw sequence + cat_ae metadata
    python -m models.cross_attention.train_immrep23 \\
        --raw_train immrep23_data/VDJdb_paired_chain.csv \\
        --use_metadata --metadata_type cat_ae

    # ESM CDR3 + cat_ae metadata (best-performing main-pipeline config)
    python -m immrep23.embed_esm                        # one-time, expensive
    python -m models.cross_attention.train_immrep23 \\
        --raw_train immrep23_data/VDJdb_paired_chain.csv \\
        --esm_tcr --use_metadata --metadata_type cat_ae

Outputs (outputs/models/cross_attention_immrep23/<run>/):
    checkpoints/checkpoint.pt    state_dict + config
    metrics.txt / metrics.json   per-epoch losses + final test metrics
    per_peptide.csv              per-peptide AUC0.1 / AUROC / AUPRC on test
    feature_augment/             saved augmenter (only with --use_metadata)
"""

import argparse
import json
import logging
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
from immrep23.feature_augment import make_paired_augmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/cross_attention_immrep23"
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


# ── ESM per-residue lookup ────────────────────────────────────────────────────

class ESMPairedLookup:
    """Per-split (tcra, tcrb) → row index lookup for ESM CDR3 per-residue features.

    Reads outputs/embeddings/esm_immrep23/<split>/ produced by
    `python -m immrep23.embed_esm`. For the training+val pool use
    split='train'; for the test pool use split='test'.
    """

    def __init__(self, esm_split_dir: Path):
        esm_split_dir = Path(esm_split_dir)
        per_residue = esm_split_dir / "cdr3_per_residue.npy"
        lengths     = esm_split_dir / "cdr3_lengths.npy"
        index       = esm_split_dir / "embedding_index.csv"
        for p in (per_residue, lengths, index):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p}. Run `python -m immrep23.embed_esm --splits {esm_split_dir.name}` first."
                )

        self.features = np.load(per_residue)            # (N, 40, D) float16
        self.lengths  = np.load(lengths)                # (N,) int32
        self.feature_dim = int(self.features.shape[2])
        self.max_len     = int(self.features.shape[1])

        idx = pd.read_csv(index, index_col=0)
        if "tcra" not in idx.columns or "tcrb" not in idx.columns:
            raise ValueError(f"{index} must have 'tcra' and 'tcrb' columns; got {list(idx.columns)}")

        seen: dict[tuple[str, str], int] = {}
        for row, (a, b) in enumerate(zip(idx["tcra"].astype(str), idx["tcrb"].astype(str))):
            if (a, b) not in seen and self.lengths[row] > 0:
                seen[(a, b)] = row
        self.key_to_row = seen
        logger.info("ESMPairedLookup(%s): %d unique (tcra, tcrb) pairs, hidden_dim=%d",
                    esm_split_dir.name, len(seen), self.feature_dim)

    def filter_df(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        keys = list(zip(df["tcra"].astype(str), df["tcrb"].astype(str)))
        keep = np.array([k in self.key_to_row for k in keys])
        n_drop = int((~keep).sum())
        if n_drop:
            logger.warning("Dropping %d/%d %s rows missing ESM TCR features",
                           n_drop, len(df), split)
        return df.loc[keep].reset_index(drop=True)

    def row_indices(self, df: pd.DataFrame) -> np.ndarray:
        keys = list(zip(df["tcra"].astype(str), df["tcrb"].astype(str)))
        return np.asarray([self.key_to_row[k] for k in keys], dtype=np.int64)


# ── dataset / loader ──────────────────────────────────────────────────────────

class PairedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tcr_feature_idx: np.ndarray | None = None):
        self.tcrs   = df["cdr3"].tolist()
        self.peps   = df["peptide"].tolist()
        self.labels = df["label"].astype(float).tolist()
        self.tcr_feature_idx = tcr_feature_idx

    def __len__(self):
        return len(self.tcrs)

    def __getitem__(self, i):
        if self.tcr_feature_idx is not None:
            return i, int(self.tcr_feature_idx[i]), self.peps[i], self.labels[i]
        return i, self.tcrs[i], self.peps[i], self.labels[i]


def make_loader(df: pd.DataFrame, meta: np.ndarray | None,
                batch_size: int, shuffle: bool, device: str,
                esm_lookup: ESMPairedLookup | None = None,
                tcr_feature_idx: np.ndarray | None = None) -> DataLoader:
    ds = PairedDataset(df, tcr_feature_idx=tcr_feature_idx)
    use_features = esm_lookup is not None

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
            feats = esm_lookup.features[feat_rows]                  # (B, L, D) float16
            tlen  = esm_lookup.lengths[feat_rows]                   # (B,)
            ti = torch.from_numpy(feats.astype(np.float32, copy=False)).to(device)
            tm = torch.zeros((len(batch), esm_lookup.max_len), dtype=torch.float32, device=device)
            for j, L in enumerate(tlen):
                tm[j, :int(L)] = 1.0
        else:
            tcrs = [b[1] for b in batch]
            ti, tm = collate_sequences(tcrs, TCR_MAX_LEN_PAIRED, device)
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


def val_macro_auc(model, df_val: pd.DataFrame, loader) -> float:
    y, p = predict(model, loader)
    df_eval = df_val[["peptide"]].copy()
    df_eval["label"] = y
    df_eval["score"] = p
    macro, _ = macro_auc01(df_eval)
    return macro


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--train",     default=None, type=Path,
                    help="Pre-built combined positives+negatives CSV (from immrep23.build_negatives)")
    ap.add_argument("--raw_train", default=None, type=Path,
                    help="Raw IMMREP23 training file (positives only). Negatives generated in-memory.")
    ap.add_argument("--test",      default=Path("immrep23_data/test.csv"), type=Path)
    ap.add_argument("--solutions", default=Path("immrep23_data/solutions.csv"), type=Path)
    ap.add_argument("--out",       default=None, type=Path)

    ap.add_argument("--val_frac",    type=float, default=0.1)
    ap.add_argument("--neg_per_pos", type=int,   default=5)
    ap.add_argument("--neg_seed",    type=int,   default=42)

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
                    help="Append paired V/J + HLA metadata after pooling")
    ap.add_argument("--metadata_type", choices=["one_hot", "cat_ae"], default="cat_ae",
                    help="Metadata encoder. cat_ae matches the best-performing main-pipeline config.")

    ap.add_argument("--esm_tcr", action="store_true",
                    help="Use per-residue ESMplusplus_large features for the TCR side. "
                         "Requires `python -m immrep23.embed_esm` to have been run first.")
    ap.add_argument("--esm_dir", default=Path("outputs/embeddings/esm_immrep23"), type=Path,
                    help="Root directory containing train/ and test/ subdirs with "
                         "cdr3_per_residue.npy / cdr3_lengths.npy / embedding_index.csv.")
    return ap.parse_args()


def _resolve_out(args) -> Path:
    if args.out:
        return args.out
    src = "esm-cdr3" if args.esm_tcr else "raw"
    suffix = f"_aug-{args.metadata_type}" if args.use_metadata else ""
    return Path(OUT_DIR) / f"cross_attention_immrep23_{src}{suffix}"


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

    # ── ESM lookups ───────────────────────────────────────────────────────────
    train_lookup = test_lookup = None
    tcr_feature_dim = 0
    max_tcr_len = TCR_MAX_LEN_PAIRED
    tr_feat = va_feat = te_feat = None
    if args.esm_tcr:
        train_lookup = ESMPairedLookup(args.esm_dir / "train")
        test_lookup  = ESMPairedLookup(args.esm_dir / "test")
        if train_lookup.feature_dim != test_lookup.feature_dim:
            raise ValueError(
                f"Train/test ESM hidden dims differ: "
                f"{train_lookup.feature_dim} vs {test_lookup.feature_dim}"
            )
        tcr_feature_dim = train_lookup.feature_dim
        max_tcr_len     = train_lookup.max_len

        df_train = train_lookup.filter_df(df_train, "train")
        df_val   = train_lookup.filter_df(df_val,   "val")
        df_test  = test_lookup.filter_df(df_test,   "test")

        tr_feat = train_lookup.row_indices(df_train)
        va_feat = train_lookup.row_indices(df_val)
        te_feat = test_lookup.row_indices(df_test)

    # ── metadata ──────────────────────────────────────────────────────────────
    meta_train = meta_val = meta_test = None
    meta_dim = 0
    aug = None
    if args.use_metadata:
        aug = make_paired_augmenter(args.metadata_type)
        aug.fit(df_train)
        meta_train = aug.transform(df_train).astype(np.float32)
        meta_val   = aug.transform(df_val).astype(np.float32)
        meta_test  = aug.transform(df_test).astype(np.float32)
        meta_dim   = aug.feature_dim
        logger.info("Metadata (%s): dim=%d  (%s)",
                    args.metadata_type, meta_dim, aug.feature_breakdown())

    # ── loaders ───────────────────────────────────────────────────────────────
    train_loader = make_loader(df_train, meta_train, args.batch_size, True,  device,
                               esm_lookup=train_lookup, tcr_feature_idx=tr_feat)
    val_loader   = make_loader(df_val,   meta_val,   args.batch_size, False, device,
                               esm_lookup=train_lookup, tcr_feature_idx=va_feat)
    test_loader  = make_loader(df_test,  meta_test,  args.batch_size, False, device,
                               esm_lookup=test_lookup,  tcr_feature_idx=te_feat)

    # ── model ─────────────────────────────────────────────────────────────────
    model = CrossAttentionTCRPep(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, meta_dim=meta_dim,
        tcr_feature_dim=tcr_feature_dim,
        max_tcr_len=max_tcr_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %d", n_params)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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

        val_macro = val_macro_auc(model, df_val, val_loader)
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
    metadata_label = (f"paired_{args.metadata_type}" if args.use_metadata else "none")
    tcr_input_label = "esm-cdr3" if args.esm_tcr else "raw"

    torch.save({
        "state_dict": best_state or model.state_dict(),
        "config": {
            "d_model":         args.d_model,
            "n_heads":         args.n_heads,
            "n_layers":        args.n_layers,
            "dropout":         args.dropout,
            "meta_dim":        meta_dim,
            "tcr_feature_dim": tcr_feature_dim,
            "max_tcr_len":     max_tcr_len,
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
        f.write(f"tcr_input={tcr_input_label}\n")
        f.write(f"metadata={metadata_label}\n")
        f.write(f"n_params={n_params}\n")

    summary = {
        "experiment_name":        out_dir.name,
        "model":                  "cross_attention_paired",
        "tcr_input":              tcr_input_label,
        "metadata":               metadata_label,
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
