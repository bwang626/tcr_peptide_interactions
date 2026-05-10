"""IMMREP23 official evaluation metric: Macro AUC0.1 with McClish standardisation.

From the lessons-learned paper (Nielsen et al., 2024):

    "We calculate this AUC0.1 independently for each peptide in the test set
    and then calculate the arithmetic mean of these peptide-specific AUC0.1
    scores."

The partial AUC up to FPR=0.1 is McClish-standardised so a random ranker
scores 0.5 and a perfect ranker scores 1.0:

    pAUC_McClish = 0.5 * (1 + (pAUC - min_pAUC) / (max_pAUC - min_pAUC))

with min_pAUC = 0.5 * fpr_max² (random) and max_pAUC = fpr_max (perfect).

For fpr_max=0.1: min=0.005, max=0.1.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def partial_auc_mcclish(y_true: Iterable[int], y_score: Iterable[float],
                        max_fpr: float = 0.1) -> float:
    """Partial AUC up to max_fpr, McClish-standardised to [0.5, 1.0].

    Returns 0.5 if y_true has fewer than 2 classes (undefined → random baseline).
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5

    # sklearn returns the McClish-standardised partial AUC when max_fpr < 1.
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def macro_auc01(df: pd.DataFrame,
                peptide_col: str = "peptide",
                label_col: str = "label",
                score_col: str = "score",
                max_fpr: float = 0.1,
                min_examples: int = 2,
                min_positives: int = 1) -> tuple[float, pd.DataFrame]:
    """Macro Mean AUC0.1: arithmetic mean of per-peptide McClish pAUC.

    Skips peptides with fewer than `min_examples` rows or fewer than
    `min_positives` positives — these are degenerate cases where AUC is
    undefined.

    Returns:
        (macro_score, per_peptide_df)
    """
    rows = []
    for pep, grp in df.groupby(peptide_col, sort=False):
        n_pos = int((grp[label_col] == 1).sum())
        n_neg = int((grp[label_col] == 0).sum())
        if len(grp) < min_examples or n_pos < min_positives or n_neg < 1:
            rows.append({
                "peptide": pep, "n": len(grp), "n_pos": n_pos, "n_neg": n_neg,
                "auc01": np.nan, "auroc": np.nan, "auprc": np.nan,
            })
            continue
        auc01 = partial_auc_mcclish(grp[label_col], grp[score_col], max_fpr=max_fpr)
        auroc = float(roc_auc_score(grp[label_col], grp[score_col]))
        auprc = float(average_precision_score(grp[label_col], grp[score_col]))
        rows.append({
            "peptide": pep, "n": len(grp), "n_pos": n_pos, "n_neg": n_neg,
            "auc01": auc01, "auroc": auroc, "auprc": auprc,
        })

    per_peptide = pd.DataFrame(rows).sort_values("auc01", ascending=False)
    macro = float(per_peptide["auc01"].dropna().mean())
    return macro, per_peptide


def overall_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Pooled (not per-peptide) AUROC, AUPRC, and McClish AUC0.1."""
    return {
        "auroc":  float(roc_auc_score(y_true, y_score)),
        "auprc":  float(average_precision_score(y_true, y_score)),
        "auc01":  partial_auc_mcclish(y_true, y_score, max_fpr=0.1),
    }
