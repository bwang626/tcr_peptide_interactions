#!/usr/bin/env python3
"""Generate paired-chain negatives for the IMMREP23 training set.

IMMREP23's official negative-generation rule (used for the test set, and
recommended for training negatives by the organisers):

    For each positive (TCR, peptide_A), sample N negatives from TCRs
    binding peptides at Levenshtein distance > 3 from peptide_A. The
    full TCR (both α and β chains, V/J genes, and CDRs) is swapped —
    not just the CDR3.

    The official ratio is 1:5.

This is the paired-chain analogue of the repo-level build_negatives.py,
but simplified: IMMREP23 has only 20 epitopes and every one is well-
populated, so we don't split the donor pool into "test-group" vs
"left-out-group" by pair count. We just sample from any peptide with
Levenshtein > 3.

Run from repo root:
    python -m immrep23.build_negatives \\
        --train immrep23_data/VDJdb_paired_chain.csv \\
        --out   data/splits_immrep23/train.csv

Output: a CSV with the same columns as the input plus a `label` column
(1=positive, 0=negative). The negative rows carry the donor TCR's full
metadata (Va/Ja/CDR1a/CDR2a/CDR3a/TCRa + β counterparts) and the target
peptide / HLA from the original positive.
"""

import argparse
import logging
import random
from pathlib import Path

import pandas as pd

try:
    from Levenshtein import distance as _lev  # type: ignore[import-untyped]
except ImportError:
    def _lev(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                prev, dp[j] = dp[j], prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
        return dp[n]

from immrep23.dataset import load_train, ALL_COMMON_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LEV_THRESHOLD = 3
NEG_PER_POS = 5

# Columns that travel with the TCR when we swap it into a different peptide row.
# These are the donor TCR's full identity. The receptor peptide and HLA come
# from the ORIGINAL positive, so we keep them as-is.
TCR_DONOR_COLS = (
    "va", "ja", "tcra", "cdr1a", "cdr2a", "cdr3a", "cdr3a_extended",
    "vb", "jb", "tcrb", "cdr1b", "cdr2b", "cdr3b", "cdr3b_extended",
)


def _tcr_key(row) -> tuple:
    """Identity hash for a paired TCR (used to deduplicate the donor pool)."""
    return tuple(row.get(c, "") for c in ("cdr3a", "cdr3b", "va", "vb", "ja", "jb"))


def build_negatives(positives: pd.DataFrame,
                    rng: random.Random,
                    neg_per_pos: int = NEG_PER_POS,
                    lev_threshold: int = LEV_THRESHOLD) -> pd.DataFrame:
    unique_peptides = sorted(positives["peptide"].unique())
    logger.info("Unique peptides: %d", len(unique_peptides))

    # Precompute Levenshtein distances between every peptide pair.
    lev: dict[tuple[str, str], int] = {}
    for i, p1 in enumerate(unique_peptides):
        for p2 in unique_peptides[i + 1:]:
            d = _lev(p1, p2)
            lev[(p1, p2)] = lev[(p2, p1)] = d

    # Donor TCRs grouped by their bound peptide. Each donor row stays as a
    # full record (we'll merge it with the target peptide later).
    donors_by_peptide: dict[str, list[dict]] = {}
    for pep, grp in positives.groupby("peptide", sort=False):
        # Deduplicate identical paired TCRs within the same peptide group.
        seen: set[tuple] = set()
        donor_list: list[dict] = []
        for _, row in grp.iterrows():
            key = _tcr_key(row)
            if key in seen:
                continue
            seen.add(key)
            donor_list.append({c: row.get(c, "") for c in TCR_DONOR_COLS})
        donors_by_peptide[str(pep)] = donor_list

    # The set of true positive (TCR, peptide) pairs — used to reject any
    # "negative" that happens to be a real binder. Key on (cdr3a, cdr3b, peptide).
    positive_set = frozenset(
        (row["cdr3a"], row["cdr3b"], row["peptide"])
        for _, row in positives.iterrows()
    )

    rows: list[dict] = []
    for peptide_a, pep_group in positives.groupby("peptide", sort=False):
        peptide_a = str(peptide_a)

        # Donor pool: TCRs from peptides at Levenshtein > threshold from peptide_a.
        pool: list[dict] = []
        seen_keys: set[tuple] = set()
        for src_pep, donor_list in donors_by_peptide.items():
            if src_pep == peptide_a:
                continue
            if lev.get((peptide_a, src_pep), _lev(peptide_a, src_pep)) <= lev_threshold:
                continue
            for d in donor_list:
                key = (d.get("cdr3a", ""), d.get("cdr3b", ""), d.get("va", ""), d.get("vb", ""))
                if (d.get("cdr3a", ""), d.get("cdr3b", ""), peptide_a) in positive_set:
                    continue
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                pool.append(d)

        if not pool:
            logger.warning("Empty donor pool for peptide %s — skipping", peptide_a)
            continue

        for _, pos_row in pep_group.iterrows():
            sampled = rng.sample(pool, min(neg_per_pos, len(pool)))
            for donor in sampled:
                rows.append({
                    **donor,
                    "peptide": peptide_a,
                    "hla":     pos_row.get("hla", ""),
                    "label":   0,
                })

    neg_df = pd.DataFrame(rows)
    # The same donor TCR may have been sampled for multiple positives of the
    # same peptide — collapse to unique (TCR, peptide) pairs.
    neg_df = neg_df.drop_duplicates(
        subset=["cdr3a", "cdr3b", "va", "vb", "peptide"]
    ).reset_index(drop=True)
    return neg_df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True,
                    help="IMMREP23 training file (e.g. immrep23_data/VDJdb_paired_chain.csv)")
    ap.add_argument("--out", type=Path,
                    default=Path("data/splits_immrep23/train.csv"),
                    help="Output combined positives+negatives CSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-per-pos", type=int, default=NEG_PER_POS,
                    help=f"Negatives per positive (default: {NEG_PER_POS} — IMMREP23 official ratio is 1:5)")
    ap.add_argument("--lev-threshold", type=int, default=LEV_THRESHOLD,
                    help=f"Negatives require peptide Levenshtein > this (default: {LEV_THRESHOLD})")
    args = ap.parse_args()

    pos = load_train(args.train)
    pos["label"] = 1
    logger.info("Loaded %d positives", len(pos))

    rng = random.Random(args.seed)
    neg = build_negatives(pos, rng, args.neg_per_pos, args.lev_threshold)
    logger.info("Generated %d unique negatives  (overall ratio 1:%.2f)",
                len(neg), len(neg) / max(1, len(pos)))

    shared_cols = [c for c in pos.columns if c in neg.columns and c != "label"]
    combined = pd.concat(
        [pos[shared_cols + ["label"]], neg[shared_cols + ["label"]]],
        ignore_index=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    logger.info("Wrote %d rows → %s", len(combined), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
