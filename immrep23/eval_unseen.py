"""Evaluate the three best main-pipeline models on the 3 IMMREP23 unseen peptides.

Unseen peptides (absent from VDJdb and IEDB at challenge time):
    SALPTNADLY, TSDACMMTMY, FTDALGIDEY

Models evaluated:
    1. Cross-attention  (cross_attention_esm-cdr3_aug-cat_ae)
       Requires ESM per-residue features.  If outputs/embeddings/esm/
       cdr3_per_residue.npy is missing this model is skipped with
       instructions for generating it on Colab.

    2. GNN R-GAT        (gnn_aug-cat_ae)
       Uses raw CDR3b sequences and saved CatAE augmenter. Runs locally.

    3. Random Forest    (onehot_meta, retrained inline)
       One-hot encodes CDR3b + peptide on-the-fly. Retrains in ~2 min
       on the combined train+val splits. Runs locally.

All models are trained on the main-pipeline combined corpus (TRBb CDR3 only)
and evaluated on the IMMREP23 test data.  Input mapping:
    IMMREP23 CDR3b  -> cdr3    (after gap-stripping)
    IMMREP23 Vb/Jb  -> v_gene / j_gene
    IMMREP23 HLA    -> mhc_a   (all class I -> mhc_class = MHCI)

Run from repo root:
    python -m immrep23.eval_unseen
    python -m immrep23.eval_unseen --ca_dir outputs/models/cross_attention/cross_attention_esm-cdr3_aug-cat_ae
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UNSEEN = {"SALPTNADLY", "TSDACMMTMY", "FTDALGIDEY"}

DEFAULT_CA_DIR  = Path("outputs/models/cross_attention/cross_attention_esm-cdr3_aug-cat_ae")
DEFAULT_GNN_DIR = Path("outputs/models/gnn/gnn_aug-cat_ae")
# embed_esm.py saves to esm_immrep23/test/; fall back to the main esm/ dir
DEFAULT_ESM_DIR = Path("outputs/embeddings/esm_immrep23/test")
DEFAULT_OH_DIR  = Path("outputs/embeddings/one_hot")
DEFAULT_SPLITS  = Path("data/splits")
IMMREP_DIR      = Path("immrep23_data")


# ── IMMREP23 data loading ─────────────────────────────────────────────────────

def _load_unseen_test() -> pd.DataFrame:
    test_path = IMMREP_DIR / "test.csv"
    sol_path  = IMMREP_DIR / "solutions.csv"
    if not test_path.exists() or not sol_path.exists():
        sys.exit(
            "IMMREP23 data not found. Run:\n"
            "    python -m immrep23.fetch\n"
            "then re-run this script."
        )
    test = pd.read_csv(test_path)
    sols = pd.read_csv(sol_path)
    sols.columns = [c.lower() for c in sols.columns]
    test.columns = [c.lower() for c in test.columns]
    df = test.merge(sols[["id", "label"]], on="id", how="inner")

    # Normalise sequences (strip ANARCI gaps)
    for col in ("cdr3b", "cdr3a", "tcra", "tcrb", "peptide"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.replace("-", "").str.replace(".", "").str.upper()

    # Map to our schema
    df["cdr3"]      = df["cdr3b"]
    df["v_gene"]    = df.get("vb", pd.Series("", index=df.index)).fillna("").astype(str)
    df["j_gene"]    = df.get("jb", pd.Series("", index=df.index)).fillna("").astype(str)
    df["mhc_a"]     = df.get("hla", pd.Series("", index=df.index)).fillna("").astype(str)
    df["mhc_class"] = "MHCI"

    # Filter to unseen peptides only
    df = df[df["peptide"].str.upper().isin(UNSEEN)].reset_index(drop=True)
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    logger.info("Unseen test: %d rows  (%d pos / %d neg)  over peptides: %s",
                len(df), n_pos, n_neg, sorted(df["peptide"].unique()))
    return df


# ── evaluation metric ─────────────────────────────────────────────────────────

def _report(model_name: str, df: pd.DataFrame, scores: np.ndarray) -> dict:
    from immrep23.evaluate import macro_auc01
    df = df.copy()
    df["score"] = scores
    macro, per_pep = macro_auc01(df, label_col="label")
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  {'Peptide':<25}  {'n_pos':>5}  {'n_neg':>5}  {'AUC0.1':>7}")
    print(f"  {'-'*50}")
    for _, row in per_pep.iterrows():
        marker = " *" if str(row['peptide']) in UNSEEN else ""
        auc_str = f"{row['auc01']:.4f}" if not np.isnan(row['auc01']) else "  n/a "
        print(f"  {str(row['peptide']):<25}  {int(row['n_pos']):>5}  {int(row['n_neg']):>5}  {auc_str}{marker}")
    print(f"  {'-'*50}")
    print(f"  {'Macro AUC0.1 (unseen mean)':<43}  {macro:.4f}")
    return {
        "model": model_name,
        "macro_auc01": macro,
        "per_peptide": per_pep.set_index("peptide")["auc01"].to_dict(),
    }


# ── Model 1: Cross-attention ESM ──────────────────────────────────────────────

def eval_cross_attention(df: pd.DataFrame, ca_dir: Path, esm_dir: Path) -> dict | None:
    ckpt_path = ca_dir / "checkpoints" / "checkpoint.pt"
    if not ckpt_path.exists():
        logger.warning("Cross-attention checkpoint not found at %s — skipping.", ckpt_path)
        return None

    pr_path = esm_dir / "cdr3_per_residue.npy"
    if not pr_path.exists():
        print("\n[Cross-attention ESM] SKIPPED — ESM per-residue features not found.")
        print("  176 unique TCRb sequences (304-324 residues) need embedding.")
        print("  Run locally (~35-45 min on CPU, ~3-5 min on GPU):")
        print("    python -m immrep23.embed_esm --splits test --batch_size 8")
        print("  Then re-run this script.")
        return None

    from models.cross_attention.model import CrossAttentionTCRPep, PEP_MAX_LEN, collate_sequences
    from immrep23.eval_unseen_esm import ESMImmrepLookup

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CrossAttentionTCRPep(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    max_tcr = ckpt["config"]["max_tcr_len"]

    try:
        lookup = ESMImmrepLookup(esm_dir, max_len=max_tcr)
    except FileNotFoundError as e:
        print(f"\n[Cross-attention ESM] SKIPPED — {e}")
        return None

    df_filtered, feat_idx = lookup.filter_and_index(df)
    if len(df_filtered) == 0:
        print("\n[Cross-attention ESM] SKIPPED — no IMMREP23 CDR3b sequences in ESM lookup.")
        return None

    aug_dir = ca_dir / "feature_augment"
    meta = None
    if aug_dir.exists():
        from embeddings.feature_augment.autoencoder import CatAEFeatureAugmenter
        aug = CatAEFeatureAugmenter.load(str(aug_dir))
        meta = torch.tensor(aug.transform(df_filtered), dtype=torch.float32)

    probs = []
    with torch.no_grad():
        for i in range(0, len(df_filtered), 128):
            rows  = feat_idx[i:i+128]
            feats = torch.from_numpy(lookup.features[rows].astype(np.float32, copy=False))
            tlen  = lookup.lengths[rows]
            tm = torch.zeros(len(rows), max_tcr, dtype=torch.float32)
            for j, L in enumerate(tlen):
                tm[j, :int(L)] = 1.0
            peps   = df_filtered["peptide"].iloc[i:i+128].tolist()
            pi, pm = collate_sequences(peps, PEP_MAX_LEN, "cpu")
            m = meta[i:i+128] if meta is not None else None
            logits = model(feats, pi, tm, pm, m)
            probs.append(torch.sigmoid(logits).numpy())

    return _report("Cross-attention  ESM/CDR3 + cat_ae", df_filtered, np.concatenate(probs))


# ── Model 2: GNN aug-cat_ae ───────────────────────────────────────────────────

def eval_gnn(df: pd.DataFrame, gnn_dir: Path) -> dict | None:
    ckpt_path = gnn_dir / "checkpoints" / "checkpoint.pt"
    if not ckpt_path.exists():
        logger.warning("GNN checkpoint not found at %s — skipping.", ckpt_path)
        return None

    from models.gnn.train import GNNBindingClassifier, LabeledPairDataset, make_collate, predict_proba
    from embeddings.feature_augment.autoencoder import CatAEFeatureAugmenter

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = GNNBindingClassifier(**cfg)
    model.load_state_dict(ckpt["state_dict"])

    aug_dir = gnn_dir / "feature_augment"
    meta = None
    meta_dim = cfg.get("meta_dim", 0)
    if meta_dim > 0 and aug_dir.exists():
        aug  = CatAEFeatureAugmenter.load(str(aug_dir))
        meta = aug.transform(df).astype(np.float32)

    device   = torch.device("cpu")
    has_meta = meta is not None
    collate  = make_collate(cfg["tcr_max_len"], cfg["pep_max_len"], has_meta)
    from models.gnn.train import LabeledPairDataset
    ds = LabeledPairDataset(
        df["cdr3"].tolist(), df["peptide"].tolist(),
        df["label"].astype(np.float32).to_numpy(), meta,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    _, scores = predict_proba(model, loader, device)
    return _report("GNN R-GAT  + cat_ae", df, scores)


# ── Model 3: Random Forest one-hot + meta (retrain) ───────────────────────────

def _onehot_for_df(df: pd.DataFrame, oh_dir: Path,
                   splits_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build one-hot + one-hot-meta features for an arbitrary DataFrame.
    TCR and peptide one-hot are encoded on the fly from raw sequences;
    OneHotFeatureAugmenter is fit on training split."""
    from embeddings.one_hot.model import encode_sequence, DEFAULT_MAX_LEN, VOCAB_SIZE
    from embeddings.feature_augment.one_hot import OneHotFeatureAugmenter

    tcr_L = DEFAULT_MAX_LEN["tcr"]
    pep_L = DEFAULT_MAX_LEN["peptide"]
    V     = VOCAB_SIZE

    tcr = np.stack([encode_sequence(s, tcr_L) for s in df["cdr3"].astype(str)]).reshape(len(df), -1)
    pep = np.stack([encode_sequence(s, pep_L) for s in df["peptide"].astype(str)]).reshape(len(df), -1)
    X   = np.concatenate([tcr, pep], axis=1).astype(np.float32)

    train_df = pd.read_csv(splits_dir / "train.csv", low_memory=False)
    aug = OneHotFeatureAugmenter()
    aug.fit(train_df)
    meta = aug.transform(df).astype(np.float32)
    return X, meta, aug


def eval_rf(df: pd.DataFrame, oh_dir: Path, splits_dir: Path) -> dict | None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_curve

    logger.info("RF: loading train + val one-hot embeddings for retraining...")

    # Build train+val feature matrices from pre-computed one-hot files
    idx  = pd.read_csv(oh_dir / "embedding_index.csv", index_col=0)
    tcr_emb = np.load(oh_dir / "tcr_embeddings.npy").astype(np.float32)
    pep_emb = np.load(oh_dir / "peptide_embeddings.npy").astype(np.float32)
    tcr_lk  = {str(seq): vec for seq, vec in zip(idx["cdr3"], tcr_emb)}
    pep_lk  = {str(pep): vec for pep, vec in zip(idx["peptide"], pep_emb)}

    from embeddings.feature_augment.one_hot import OneHotFeatureAugmenter

    def _build_split(split_csv: Path) -> tuple[np.ndarray, np.ndarray]:
        df_s = pd.read_csv(split_csv, low_memory=False)
        aug  = OneHotFeatureAugmenter()
        aug.fit(pd.read_csv(splits_dir / "train.csv", low_memory=False))
        rows = []
        for _, row in df_s.iterrows():
            tcr = tcr_lk.get(str(row["cdr3"]))
            pep = pep_lk.get(str(row["peptide"]))
            if tcr is None or pep is None:
                continue
            feat = np.concatenate([tcr, pep, aug.transform(pd.DataFrame([row]))[0]])
            rows.append((feat, int(row["label"])))
        X = np.stack([r[0] for r in rows])
        y = np.array([r[1] for r in rows], dtype=np.int64)
        return X, y

    logger.info("RF: building train split features...")
    X_tr, y_tr = _build_split(splits_dir / "train.csv")
    logger.info("RF: building val split features...")
    X_va, y_va = _build_split(splits_dir / "val.csv")
    X_fit = np.concatenate([X_tr, X_va])
    y_fit = np.concatenate([y_tr, y_va])
    logger.info("RF: fitting on %d rows...", len(X_fit))

    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                 random_state=42, class_weight="balanced",
                                 oob_score=True)
    clf.fit(X_fit, y_fit)
    logger.info("RF: fitted. OOB accuracy: %.4f", clf.oob_score_)

    # Encode IMMREP23 unseen test rows on-the-fly
    aug  = OneHotFeatureAugmenter()
    aug.fit(pd.read_csv(splits_dir / "train.csv", low_memory=False))
    from embeddings.one_hot.model import encode_sequence, DEFAULT_MAX_LEN
    tcr_L = DEFAULT_MAX_LEN["tcr"]
    pep_L = DEFAULT_MAX_LEN["peptide"]

    rows = []
    for _, row in df.iterrows():
        tcr_vec = encode_sequence(str(row["cdr3"]),     tcr_L).flatten()
        pep_vec = encode_sequence(str(row["peptide"]),  pep_L).flatten()
        meta    = aug.transform(pd.DataFrame([row]))[0]
        rows.append(np.concatenate([tcr_vec, pep_vec, meta]))
    X_test = np.stack(rows).astype(np.float32)

    scores = clf.predict_proba(X_test)[:, 1]
    return _report("Random Forest  one-hot + meta", df, scores)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ca_dir",     default=DEFAULT_CA_DIR,  type=Path)
    ap.add_argument("--gnn_dir",    default=DEFAULT_GNN_DIR, type=Path)
    ap.add_argument("--esm_dir",    default=DEFAULT_ESM_DIR, type=Path)
    ap.add_argument("--oh_dir",     default=DEFAULT_OH_DIR,  type=Path)
    ap.add_argument("--splits_dir", default=DEFAULT_SPLITS,  type=Path)
    ap.add_argument("--out",        default=None, type=Path,
                    help="Optional path to write results CSV")
    return ap.parse_args()


def main():
    args = parse_args()

    df = _load_unseen_test()
    results = []

    r = eval_cross_attention(df, args.ca_dir, args.esm_dir)
    if r:
        results.append(r)

    r = eval_gnn(df, args.gnn_dir)
    if r:
        results.append(r)

    r = eval_rf(df, args.oh_dir, args.splits_dir)
    if r:
        results.append(r)

    if not results:
        print("\nNo models were evaluated successfully.")
        return

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY — Macro AUC0.1 on 3 IMMREP23 unseen peptides")
    print(f"  (SALPTNADLY, TSDACMMTMY, FTDALGIDEY — zero-shot)")
    print(f"{'='*60}")
    print(f"  {'Model':<45}  {'Macro AUC0.1':>12}")
    print(f"  {'-'*58}")
    for r in sorted(results, key=lambda x: -x["macro_auc01"]):
        print(f"  {r['model']:<45}  {r['macro_auc01']:>12.4f}")

    # Per-peptide breakdown
    print(f"\n  Per-peptide AUC0.1:")
    print(f"  {'Peptide':<25}", end="")
    for r in results:
        print(f"  {r['model'][:18]:>18}", end="")
    print()
    print(f"  {'-' * (25 + 20 * len(results))}")
    for pep in sorted(UNSEEN):
        print(f"  {pep:<25}", end="")
        for r in results:
            val = r["per_peptide"].get(pep, float("nan"))
            print(f"  {'n/a':>18}" if np.isnan(val) else f"  {val:>18.4f}", end="")
        print()

    if args.out:
        rows = []
        for r in results:
            for pep, auc in r["per_peptide"].items():
                rows.append({"model": r["model"], "peptide": pep, "auc01": auc})
        pd.DataFrame(rows).to_csv(args.out, index=False)
        logger.info("Results written to %s", args.out)


if __name__ == "__main__":
    main()
