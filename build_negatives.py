#!/usr/bin/env python3
"""Generate negative TCR-peptide pairs by swapping TCRs across distant peptides.

Methodology (from paper §2.2.2):
  For each positive (TCR, peptide_A):
    - 3 negatives from TCRs that bind peptides with >=10 positive pairs
      AND Levenshtein distance > 3 from peptide_A.
    - 2 negatives from TCRs that bind peptides with <10 positive pairs
      AND Levenshtein distance > 3 from peptide_A.
  Target positive-to-negative ratio: 1:5.
  When the left-out pool is too small, fewer than 2 negatives are generated
  (reproducing the 1:4.x exceptions reported in the paper).
"""

import argparse
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


MIN_PAIRS_THRESHOLD = 10
N_NEG_FROM_TEST = 3
N_NEG_FROM_LEFTOUT = 2
LEV_THRESHOLD = 3
TCR_KEY = ["cdr3", "v_gene", "j_gene"]


def precompute_lev(peptides: list[str]) -> dict[tuple[str, str], int]:
    cache: dict[tuple[str, str], int] = {}
    total = len(peptides) * (len(peptides) - 1) // 2
    done = 0
    for i, p1 in enumerate(peptides):
        for p2 in peptides[i + 1:]:
            d = _lev(p1, p2)
            cache[(p1, p2)] = d
            cache[(p2, p1)] = d
            done += 1
            if done % 200_000 == 0:
                print(f"  {done:,}/{total:,} pairs computed...", flush=True)
    return cache


def build_pool(
    peptide_a: str,
    source_peptides: set[str],
    tcr_by_peptide: dict[str, list[dict]],
    lev_cache: dict[tuple[str, str], int],
    positive_set: frozenset[tuple[str, str]],
    lev_threshold: int,
) -> list[dict]:
    """Return deduplicated TCRs from source_peptides eligible to swap into peptide_a."""
    seen_tcrs: set[tuple[str, str, str]] = set()
    pool: list[dict] = []
    for src in source_peptides:
        if src == peptide_a:
            continue
        if lev_cache.get((peptide_a, src), _lev(peptide_a, src)) <= lev_threshold:
            continue
        for tcr in tcr_by_peptide[src]:
            key = (tcr["cdr3"], tcr["v_gene"], tcr["j_gene"])
            if (tcr["cdr3"], peptide_a) in positive_set:
                continue
            if key in seen_tcrs:
                continue
            seen_tcrs.add(key)
            pool.append(tcr)
    return pool


def build_negatives(
    positives: pd.DataFrame,
    rng: random.Random,
    n_test: int = N_NEG_FROM_TEST,
    n_leftout: int = N_NEG_FROM_LEFTOUT,
    min_pairs: int = MIN_PAIRS_THRESHOLD,
    lev_threshold: int = LEV_THRESHOLD,
) -> pd.DataFrame:
    peptide_counts = positives.groupby("peptide").size()
    test_peptides: set[str] = set(peptide_counts[peptide_counts >= min_pairs].index)
    leftout_peptides: set[str] = set(peptide_counts[peptide_counts < min_pairs].index)
    unique_peptides: list[str] = list(peptide_counts.index)

    print(f"Unique peptides total:        {len(unique_peptides):,}")
    print(f"Test peptides (>={min_pairs}):      {len(test_peptides):,}")
    print(f"Left-out peptides (<{min_pairs}):   {len(leftout_peptides):,}")

    print(f"Precomputing {len(unique_peptides) * (len(unique_peptides) - 1) // 2:,} Levenshtein pairs...", flush=True)
    lev_cache = precompute_lev(unique_peptides)
    print("Done.", flush=True)

    # Unique (cdr3, v_gene, j_gene) per peptide — each TCR in the dataset binds exactly one
    # peptide (enforced by _drop_multi_peptide_tcrs in build_dataset.py), but the same
    # (cdr3, v/j) may appear across multiple sources, so we deduplicate here.
    tcr_by_peptide: dict[str, list[dict]] = {}
    for peptide, grp in positives.groupby("peptide"):
        tcr_by_peptide[str(peptide)] = (
            grp[TCR_KEY].drop_duplicates().to_dict("records")
        )

    positive_set = frozenset(zip(positives["cdr3"], positives["peptide"]))

    rows: list[dict] = []
    n_peptides = len(test_peptides | leftout_peptides)
    processed = 0

    for peptide_a, pep_group in positives.groupby("peptide"):
        peptide_a = str(peptide_a)
        test_pool = build_pool(
            peptide_a, test_peptides, tcr_by_peptide, lev_cache, positive_set, lev_threshold
        )
        lo_pool = build_pool(
            peptide_a, leftout_peptides, tcr_by_peptide, lev_cache, positive_set, lev_threshold
        )

        for _, pos_row in pep_group.iterrows():
            ctx = {
                "tcr_chain": pos_row["tcr_chain"],
                "mhc_a": pos_row["mhc_a"],
                "mhc_b": pos_row["mhc_b"],
                "mhc_class": pos_row["mhc_class"],
            }

            # Sample without replacement up to the target count.
            # If pool is smaller than target, take all available (reproduces the 1:4.x exceptions).
            sampled: list[dict] = rng.sample(test_pool, min(n_test, len(test_pool)))
            sampled += rng.sample(lo_pool, min(n_leftout, len(lo_pool)))

            for tcr in sampled:
                rows.append({**tcr, "peptide": peptide_a, **ctx, "label": 0})

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{n_peptides} peptides...", flush=True)

    neg_df = pd.DataFrame(rows)
    # Remove duplicates: the same (TCR, peptide) pair may be sampled for multiple positives.
    neg_df = neg_df.drop_duplicates(subset=[*TCR_KEY, "peptide"]).reset_index(drop=True)
    return neg_df


def print_ratio_report(positives: pd.DataFrame, negatives: pd.DataFrame) -> None:
    pos_counts = positives.groupby("peptide").size().rename("n_pos")
    neg_counts = negatives.groupby("peptide").size().rename("n_neg")
    report = pd.concat([pos_counts, neg_counts], axis=1).fillna(0).astype({"n_neg": int})
    report["ratio"] = (report["n_neg"] / report["n_pos"]).round(2)
    below_target = report[report["ratio"] < N_NEG_FROM_TEST + N_NEG_FROM_LEFTOUT].sort_values("ratio")
    print("\nPeptides with ratio < target (1:5):")
    if below_target.empty:
        print("  None — all peptides reached full 1:5 ratio.")
    else:
        print(below_target.to_string())
    overall = len(negatives) / len(positives)
    print(f"\nOverall positive-to-negative ratio: 1:{overall:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", type=Path,
        default=Path("processed/combined_trb_clean.csv"),
        help="Path to the cleaned positive TCR-peptide CSV (default: processed/combined_trb_clean.csv)",
    )
    ap.add_argument(
        "--out-negatives", type=Path,
        default=Path("processed/negatives_trb.csv"),
        help="Output path for generated negatives only",
    )
    ap.add_argument(
        "--out-combined", type=Path,
        default=Path("processed/combined_with_negatives_trb.csv"),
        help="Output path for positives + negatives combined",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS_THRESHOLD,
                    help="Peptide pair count threshold separating test / left-out groups")
    ap.add_argument("--n-test", type=int, default=N_NEG_FROM_TEST,
                    help="Negatives per positive from test-group peptides")
    ap.add_argument("--n-leftout", type=int, default=N_NEG_FROM_LEFTOUT,
                    help="Negatives per positive from left-out-group peptides")
    ap.add_argument("--lev-threshold", type=int, default=LEV_THRESHOLD,
                    help="Levenshtein distance threshold; negatives require dist > this value")
    args = ap.parse_args()

    print(f"Loading positives from {args.input}...")
    positives = pd.read_csv(args.input, low_memory=False)
    positives = positives.assign(label=1)
    print(f"Loaded {len(positives):,} positive pairs\n")

    rng = random.Random(args.seed)
    negatives = build_negatives(
        positives,
        rng,
        n_test=args.n_test,
        n_leftout=args.n_leftout,
        min_pairs=args.min_pairs,
        lev_threshold=args.lev_threshold,
    )

    print(f"\nGenerated {len(negatives):,} unique negative pairs")
    print_ratio_report(positives, negatives)

    for path in [args.out_negatives, args.out_combined]:
        path.parent.mkdir(parents=True, exist_ok=True)

    negatives.to_csv(args.out_negatives, index=False)
    print(f"\nNegatives written to {args.out_negatives}")

    shared_cols = [c for c in positives.columns if c in negatives.columns]
    combined = pd.concat(
        [positives[shared_cols + ["label"]], negatives[shared_cols + ["label"]]],
        ignore_index=True,
    )
    combined.to_csv(args.out_combined, index=False)
    print(f"Combined dataset written to {args.out_combined} ({len(combined):,} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
