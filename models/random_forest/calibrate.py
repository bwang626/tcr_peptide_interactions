"""
Train RF onehot_meta and compare F1 under several threshold-calibration strategies.

The existing train.py trains on train+val combined and calibrates the threshold
via out-of-bag predictions on the same set. Under peptide_holdout, the val
peptides and test peptides come from disjoint pools, so an OOB-calibrated
threshold can fail to transfer to test. This script isolates the calibration
question by:

  1. Training on train only.
  2. Predicting on val and test, saving probabilities to disk.
  3. Reporting test metrics under four threshold rules:
       - default 0.5
       - F1-optimal on val   (the honest production rule)
       - F1-optimal on OOB   (matches the existing train.py rule, train-only)
       - F1-optimal on test  (oracle upper bound — diagnostic only)

Run from repo root:
    python -m models.random_forest.calibrate
"""

from __future__ import annotations

import json
import logging
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from models.embedding_loader import load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "outputs/models/random_forest/onehot_meta_calibration"


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.argmax(f1s))
    return float(thr[i]), float(f1s[i])


def metrics_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_pred_positive": int(y_pred.sum()),
        "n_true_positive": int(y_true.sum()),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Loading splits (one_hot + cat_ae metadata)")
    X_train, y_train = load_split(
        "train", tcr_embedding="one_hot", categorical="cat_ae",
    )
    X_val, y_val = load_split(
        "val", tcr_embedding="one_hot", categorical="cat_ae",
    )
    X_test, y_test = load_split(
        "test", tcr_embedding="one_hot", categorical="cat_ae",
    )
    logger.info(
        "Shapes: train=%s val=%s test=%s | pos rates: train=%.3f val=%.3f test=%.3f",
        X_train.shape, X_val.shape, X_test.shape,
        y_train.mean(), y_val.mean(), y_test.mean(),
    )

    logger.info("Fitting RandomForestClassifier on train only (n_estimators=500)")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
        oob_score=True,
    )
    clf.fit(X_train, y_train)
    logger.info("Fit took %.1fs", time.time() - t0)

    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]
    oob_prob = clf.oob_decision_function_[:, 1]

    np.save(os.path.join(OUT_DIR, "val_prob.npy"), val_prob)
    np.save(os.path.join(OUT_DIR, "test_prob.npy"), test_prob)
    np.save(os.path.join(OUT_DIR, "val_label.npy"), y_val)
    np.save(os.path.join(OUT_DIR, "test_label.npy"), y_test)
    joblib.dump(clf, os.path.join(OUT_DIR, "model.joblib"))

    auroc = float(roc_auc_score(y_test, test_prob))
    auprc = float(average_precision_score(y_test, test_prob))

    thr_val, val_f1 = best_f1_threshold(y_val, val_prob)
    thr_oob, oob_f1 = best_f1_threshold(y_train, oob_prob)
    thr_test, test_oracle_f1 = best_f1_threshold(y_test, test_prob)

    report = {
        "auroc": auroc,
        "auprc": auprc,
        "calibration": {
            "default_0.5":   metrics_at(y_test, test_prob, 0.5),
            "val_optimal":   {**metrics_at(y_test, test_prob, thr_val),   "val_f1_at_threshold": val_f1},
            "oob_optimal":   {**metrics_at(y_test, test_prob, thr_oob),   "oob_f1_at_threshold": oob_f1},
            "test_oracle":   {**metrics_at(y_test, test_prob, thr_test), "test_f1_at_threshold": test_oracle_f1},
        },
        "probability_summary": {
            "test_prob_mean": float(test_prob.mean()),
            "test_prob_median": float(np.median(test_prob)),
            "test_prob_p90": float(np.quantile(test_prob, 0.90)),
            "test_prob_p99": float(np.quantile(test_prob, 0.99)),
        },
    }

    with open(os.path.join(OUT_DIR, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== RF onehot_meta calibration ===")
    print(f"AUROC = {auroc:.4f}   AUPRC = {auprc:.4f}\n")
    print(f"{'rule':<14} {'threshold':>10} {'test_acc':>10} {'test_f1':>10} {'n_pred+':>10}")
    for name, m in report["calibration"].items():
        print(f"{name:<14} {m['threshold']:>10.4f} {m['accuracy']:>10.4f} "
              f"{m['f1']:>10.4f} {m['n_pred_positive']:>10d}")
    print(f"\ntest prob: mean={test_prob.mean():.3f} "
          f"median={np.median(test_prob):.3f} "
          f"p90={np.quantile(test_prob, 0.90):.3f} "
          f"p99={np.quantile(test_prob, 0.99):.3f}")
    print(f"\nArtifacts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
