"""
Train Random Forest baselines over the predefined embedding experiment grid.

Run from repo root:
    python -m models.random_forest.train
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from models.embedding_loader import experiment_dir_name, load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/random_forest"

EXPERIMENTS = [
    ("one_hot", None, "one_hot"),
    ("one_hot", None, None),
    ("autoencoder/plain_ae", None, "cat_ae"),
    ("autoencoder/plain_ae", None, None),
    ("autoencoder/vae", None, "cat_ae"),
    ("autoencoder/vae", None, None),
    ("esm/cdr3", "autoencoder/plain_ae", "cat_ae"),
    ("esm/cdr3", "autoencoder/plain_ae", None),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--embeddings_dir", default="outputs/embeddings")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--only_tcr", nargs="+", default=None,
                   help="Only run experiments whose TCR embedding matches one of these names.")
    return p.parse_args()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float(y_true.mean())
    return {
        "auroc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_experiment(
    tcr_embedding: str,
    peptide_embedding: str | None,
    categorical: str | None,
    splits_dir: str,
    embeddings_dir: str,
    out_dir: str,
) -> dict | None:
    name = experiment_dir_name(tcr_embedding, peptide_embedding, categorical)
    logger.info("Running Random Forest experiment: %s", name)

    try:
        X_train, y_train = load_split(
            "train",
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            splits_dir=splits_dir,
            embeddings_dir=embeddings_dir,
            drop_missing=True,
        )
        X_val, y_val = load_split(
            "val",
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            splits_dir=splits_dir,
            embeddings_dir=embeddings_dir,
            drop_missing=True,
        )
        X_test, y_test = load_split(
            "test",
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            splits_dir=splits_dir,
            embeddings_dir=embeddings_dir,
            drop_missing=True,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.warning("Skipping experiment %s: %s", name, exc)
        return None

    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    clf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_fit, y_fit)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)

    experiment_out = os.path.join(out_dir, name)
    os.makedirs(experiment_out, exist_ok=True)
    joblib.dump(clf, os.path.join(experiment_out, "model.joblib"))
    save_json(
        os.path.join(experiment_out, "metrics.json"),
        {
            "experiment_name": name,
            "model_type": "random_forest",
            "tcr_embedding": tcr_embedding,
            "peptide_embedding": peptide_embedding or tcr_embedding,
            "categorical": categorical,
            "input_dim": int(X_fit.shape[1]),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_train_combined": int(len(X_fit)),
            "n_test": int(len(X_test)),
            "metrics": metrics,
        },
    )
    logger.info(
        "Completed %s | AUROC=%.4f AUPRC=%.4f ACC=%.4f F1=%.4f",
        name,
        metrics["auroc"],
        metrics["auprc"],
        metrics["accuracy"],
        metrics["f1"],
    )

    return {
        "experiment_name": name,
        "tcr_embedding": tcr_embedding,
        "peptide_embedding": peptide_embedding or tcr_embedding,
        "categorical": categorical or "none",
        "input_dim": int(X_fit.shape[1]),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        **metrics,
    }


def _update_results_summary(out_dir: str, new_rows: list[dict]) -> None:
    """Row-replace update of results_summary.csv keyed by experiment_name."""
    if not new_rows:
        return
    path = os.path.join(out_dir, "results_summary.csv")
    new_df = pd.DataFrame(new_rows)
    new_names = set(new_df["experiment_name"])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old["experiment_name"].isin(new_names)]
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(path, index=False)
    logger.info("Updated %s (%d new rows merged)", path, len(new_df))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only_tcr:
        wanted = set(args.only_tcr)
        experiments = [e for e in EXPERIMENTS if e[0] in wanted]
        logger.info("Filtering to TCR embeddings %s -> %d experiment(s)",
                    sorted(wanted), len(experiments))

    results: list[dict] = []
    for tcr_embedding, peptide_embedding, categorical in experiments:
        result = run_experiment(
            tcr_embedding=tcr_embedding,
            peptide_embedding=peptide_embedding,
            categorical=categorical,
            splits_dir=args.splits_dir,
            embeddings_dir=args.embeddings_dir,
            out_dir=args.out_dir,
        )
        if result is not None:
            results.append(result)

    _update_results_summary(args.out_dir, results)


if __name__ == "__main__":
    main()
