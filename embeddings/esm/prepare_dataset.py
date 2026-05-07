"""
prepare_dataset.py

Reads combined_trb_clean.csv, generates full-length mature TRB amino acid
sequences via stitchr, and writes a Parquet dataset ready for ESMplusplus
embedding.

Output schema (processed/esm_trb_dataset.parquet)
---------------------------------------------------
full_aa      str   Mature VDJ+C amino acid sequence — the X passed to ESM++.
                   Signal peptide (leader) is excluded (no_leader=True); sequence
                   starts at the first residue of the V gene mature region through
                   the constant region.
peptide      str   Epitope sequence — the y target for binding classification.
                   NOT concatenated with full_aa; pairing is preserved by row.
cdr3_start   int   0-indexed position in full_aa where CDR3 begins.
cdr3_end     int   0-indexed exclusive end of CDR3 in full_aa.
                   full_aa[cdr3_start:cdr3_end] == the embedded CDR3 residues.
                   These positions mark the TCR fingerprint (D-region) for
                   downstream attention or feature masking.
cdr3         str   Original CDR3 from the database.
v_gene       str   IMGT V gene (e.g. TRBV19*01).
j_gene       str   IMGT J gene (e.g. TRBJ2-1*01).
mhc_a        str   MHC alpha chain.
mhc_b        str   MHC beta chain (NaN for most MHCI entries).
mhc_class    str   MHCI or MHCII.
source       str   Data source (VDJdb, IEDB, McPAS).
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "embeddings" / "esm"))

from stitchr_utils import get_full_aa_sequence  # noqa: E402 — path must be set first

DATA_PATH = ROOT / "processed" / "combined_trb_clean.csv"
OUT_PATH  = ROOT / "processed" / "esm_trb_dataset.csv"

# Terminal residues to try when the original CDR3 is absent from the stitched
# sequence (covers the most common fallback mutations applied by stitchr_utils).
_FALLBACK_TERMINALS = ("F", "W", "L", "V")


def _find_cdr3_span(full_aa: str, cdr3: str) -> tuple[int | None, int | None]:
    """
    Locate the CDR3 substring inside the full AA sequence.

    Returns (start, end) as a 0-indexed half-open interval so that
    full_aa[start:end] reproduces the embedded CDR3. Returns (None, None)
    if no variant of the CDR3 can be found.

    Search order:
      1. original CDR3 as stored in the database
      2. 'C' + original (fallback: conserved Cys was prepended)
      3. original + terminal (fallback: junction residue was appended)
      4. 'C' + original + terminal (both fixes applied)
    """
    cdr3 = cdr3.upper()  # stitchr uppercases all input; search must match
    has_c = cdr3.startswith("C")

    candidates: list[str] = [cdr3]
    if not has_c:
        candidates.append("C" + cdr3)
    for term in _FALLBACK_TERMINALS:
        candidates.append(cdr3 + term)
        if not has_c:
            candidates.append("C" + cdr3 + term)

    for candidate in candidates:
        idx = full_aa.find(candidate)
        if idx != -1:
            return idx, idx + len(candidate)

    return None, None


def main() -> None:
    print(f"Reading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required = ["v_gene", "j_gene", "cdr3", "peptide"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped_na = before - len(df)
    if dropped_na:
        print(f"  Dropped {dropped_na} rows with missing required fields.")
    print(f"  {len(df):,} rows to process.\n")

    full_aa_col:    list[str | None]  = []
    cdr3_start_col: list[int | None]  = []
    cdr3_end_col:   list[int | None]  = []

    t0 = time.perf_counter()
    span_misses = 0
    stitch_fails = 0
    log_every = 5_000

    for i, row in df.iterrows():
        seq = get_full_aa_sequence(
            row.v_gene, row.j_gene, row.cdr3, no_leader=True  # mature sequence, no signal peptide
        )
        full_aa_col.append(seq)

        if seq is None:
            stitch_fails += 1
            cdr3_start_col.append(None)
            cdr3_end_col.append(None)
        else:
            start, end = _find_cdr3_span(seq, row.cdr3)
            cdr3_start_col.append(start)
            cdr3_end_col.append(end)
            if start is None:
                span_misses += 1

        if (i + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (len(df) - i - 1) / rate
            print(
                f"  {i + 1:>6,}/{len(df):,}  "
                f"[{elapsed:5.1f}s elapsed  ~{remaining:5.1f}s remaining]"
            )

    elapsed = time.perf_counter() - t0
    print(f"\nStitching complete in {elapsed:.1f}s.")

    df["full_aa"]     = full_aa_col
    df["cdr3_start"]  = cdr3_start_col
    df["cdr3_end"]    = cdr3_end_col

    # Drop rows where stitching failed entirely
    before = len(df)
    df = df.dropna(subset=["full_aa"]).reset_index(drop=True)
    failure_pct = stitch_fails / before * 100
    print(f"Stitching failures dropped : {stitch_fails} ({failure_pct:.2f}%)")

    if failure_pct > 50:
        print(
            "\nWARNING: >50% of sequences failed stitching. If running on a new machine,\n"
            "stitchr's IMGT reference data may not have been downloaded. Fix with:\n"
            "  python -c \"from Stitchr import stitchrfunctions as fxn; "
            "fxn.get_ref_data('TRB', list(fxn.regions.values()), 'HUMAN')\"\n"
            "or check the stitchr docs: https://github.com/JamieHeather/stitchr"
        )
    if len(df) == 0:
        raise RuntimeError(
            "No sequences survived stitching — output would be empty. "
            "See the warning above for likely cause."
        )

    if span_misses:
        print(f"CDR3 span not found in    : {span_misses} sequences (span set to None)")

    # Reorder columns: X and span first, y second, then metadata
    cols = [
        "full_aa",
        "cdr3_start",
        "cdr3_end",
        "peptide",
        "cdr3",
        "v_gene",
        "j_gene",
        "mhc_a",
        "mhc_b",
        "mhc_class",
        "source",
    ]
    df = df[cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")

    # Quick sanity check: verify CDR3 round-trips for the first valid row
    sample = df.dropna(subset=["cdr3_start"]).iloc[0]
    extracted = sample.full_aa[int(sample.cdr3_start) : int(sample.cdr3_end)]
    print(f"\nSanity check (row 0):")
    print(f"  full_aa[:30]  : {sample.full_aa[:30]}...")
    print(f"  cdr3_start    : {sample.cdr3_start}")
    print(f"  cdr3_end      : {sample.cdr3_end}")
    print(f"  extracted CDR3: {extracted}")
    print(f"  original CDR3 : {sample.cdr3}")
    print(f"  span OK       : {sample.cdr3 in extracted}")


if __name__ == "__main__":
    main()
