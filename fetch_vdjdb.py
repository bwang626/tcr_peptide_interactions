#!/usr/bin/env python3
"""Download the latest VDJdb release and extract the full TCR-epitope table."""

import argparse
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

GITHUB_API = "https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest"
UA = {"User-Agent": "vdjdb-fetch/1.0"}


def http_get(url: str, accept: str = "application/json") -> bytes:
    req = Request(url, headers={**UA, "Accept": accept})
    with urlopen(req) as resp:
        return resp.read()


def latest_release_asset() -> tuple[str, str]:
    import json

    meta = json.loads(http_get(GITHUB_API))
    tag = meta["tag_name"]
    for asset in meta.get("assets", []):
        name = asset["name"].lower()
        if name.endswith(".zip") and "vdjdb" in name:
            return tag, asset["browser_download_url"]
    return tag, meta["zipball_url"]


def download_and_extract(url: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}", file=sys.stderr)
    blob = http_get(url, accept="application/octet-stream")
    written: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.endswith((".tsv", ".txt", ".meta")):
                continue
            target = out_dir / name
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            written.append(target)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o", "--out", type=Path, default=Path("vdjdb_data"),
        help="output directory (default: ./vdjdb_data)",
    )
    ap.add_argument(
        "--url", help="override download URL (defaults to latest GitHub release)",
    )
    args = ap.parse_args()

    if args.url:
        tag, url = "custom", args.url
    else:
        tag, url = latest_release_asset()
        print(f"Latest VDJdb release: {tag}", file=sys.stderr)

    files = download_and_extract(url, args.out)
    if not files:
        print("No TSV files found in the archive.", file=sys.stderr)
        return 1

    print(f"Extracted {len(files)} file(s) to {args.out}/:")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.2f} MB)")

    main_table = args.out / "vdjdb.slim.txt"
    full_table = args.out / "vdjdb_full.txt"
    if main_table.exists() or full_table.exists():
        target = full_table if full_table.exists() else main_table
        with target.open() as fh:
            n = sum(1 for _ in fh) - 1
        print(f"\n{target.name}: {n:,} records")

    return 0


if __name__ == "__main__":
    sys.exit(main())
