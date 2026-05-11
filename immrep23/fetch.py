#!/usr/bin/env python3
"""Download the IMMREP23 challenge data from the official GitHub mirror.

The IMMREP23 Kaggle competition's data and ground-truth solutions are
re-published openly at https://github.com/justin-barton/IMMREP23/, which
means we can fetch them via raw.githubusercontent.com without Kaggle
credentials.

Files downloaded into immrep23_data/:
    VDJdb_paired_chain.csv   training positives (Target column may be
                             always 1, since negatives are an exercise
                             for competitors)
    test.csv                 test pairs (no labels) with an `ID` column
    sample_submission.csv    submission format reference
    solutions.csv            test data + `Label` (0/1) + `Usage` (Public/Private)

Run from repo root:
    python -m immrep23.fetch
    python -m immrep23.fetch --out immrep23_data --force
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO_RAW = "https://raw.githubusercontent.com/justin-barton/IMMREP23/main/data"
FILES = (
    "VDJdb_paired_chain.csv",
    "test.csv",
    "sample_submission.csv",
    "solutions.csv",
)


def _download(url: str, dest: Path) -> None:
    print(f"  -> {dest.name} ... ", end="", flush=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"{dest.stat().st_size/1024:.1f} KB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("immrep23_data"),
                    help="Destination directory (default: immrep23_data/)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the file already exists")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IMMREP23 data into {args.out}/")

    for name in FILES:
        dest = args.out / name
        if dest.exists() and not args.force:
            print(f"  [exists] {name} (use --force to redownload)")
            continue
        try:
            _download(f"{REPO_RAW}/{name}", dest)
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            return 1

    print("\nDone. Next:")
    print(f"  python -m immrep23.build_negatives --train {args.out}/VDJdb_paired_chain.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
