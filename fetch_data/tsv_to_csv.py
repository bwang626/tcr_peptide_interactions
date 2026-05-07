#!/usr/bin/env python3
"""Convert all VDJdb TSV/TXT tables to CSV with proper quoting."""

import argparse
import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SKIP = {"LICENSE.txt", "latest-version.txt"}


def convert(src: Path, dst: Path) -> int:
    n = 0
    with src.open(newline="") as fin, dst.open("w", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            writer.writerow(row)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--in-dir", type=Path, default=Path("vdjdb_data"))
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("vdjdb_csv"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.in_dir.iterdir() if p.is_file() and p.name not in SKIP)
    if not files:
        print(f"No files found in {args.in_dir}", file=sys.stderr)
        return 1

    for src in files:
        dst = args.out_dir / (src.stem + ".csv")
        rows = convert(src, dst)
        size_mb = dst.stat().st_size / 1e6
        print(f"  {src.name:40s} -> {dst.name:40s} {rows:>9,} rows  ({size_mb:.2f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
