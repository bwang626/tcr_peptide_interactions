#!/usr/bin/env python3
"""Download the McPAS-TCR database (manually-curated pathology-associated TCRs).

The canonical source is the Shiny app at https://friedmanlab.weizmann.ac.il/McPAS-TCR/,
whose download button is wired through a session-scoped Shiny endpoint and is
not directly fetchable over plain HTTP. This script defaults to the immunarch
gitlab mirror, which tracks the same CSV. Override with --url if you have a
fresher archive.
"""

import argparse
import gzip
import io
import sys
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_URL = "https://gitlab.com/immunomind/immunarch/raw/dev-0.5.0/private/McPAS-TCR.csv.gz"
UA = {"User-Agent": "mcpas-fetch/1.0"}


def http_get(url: str) -> bytes:
    req = Request(url, headers={**UA, "Accept": "application/octet-stream,*/*"})
    with urlopen(req) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o", "--out", type=Path, default=Path("mcpas_data"),
        help="output directory (default: ./mcpas_data)",
    )
    ap.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"download URL (default: {DEFAULT_URL})",
    )
    ap.add_argument(
        "--keep-gz", action="store_true",
        help="keep the .csv.gz alongside the decompressed CSV",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url}", file=sys.stderr)
    blob = http_get(args.url)

    is_gz = args.url.endswith(".gz") or blob[:2] == b"\x1f\x8b"
    csv_path = args.out / "McPAS-TCR.csv"
    if is_gz:
        data = gzip.decompress(blob)
        if args.keep_gz:
            (args.out / "McPAS-TCR.csv.gz").write_bytes(blob)
    else:
        data = blob

    csv_path.write_bytes(data)

    n = sum(1 for _ in io.BytesIO(data)) - 1
    size_mb = len(data) / 1e6
    print(f"Wrote {csv_path} ({size_mb:.2f} MB, {n:,} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
