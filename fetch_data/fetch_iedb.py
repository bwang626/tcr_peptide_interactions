#!/usr/bin/env python3
"""Download IEDB bulk exports (zipped CSVs) and extract them.

IEDB publishes full database exports at iedb.org/database_export_v3.php.
This script streams each zip to disk, then extracts the CSVs.

By default only the two files used by build_dataset.py are downloaded:
  tcr    — TCR receptor sequences (tcr_full_v3.csv)
  tcell  — T cell assay methods, joined for confidence filtering (tcell_full_v3.csv)

Use --only to fetch other exports if needed (see EXPORTS below for all keys).
"""

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://www.iedb.org/downloader.php?file_name=doc/"

# All available bulk exports. Only "tcr" and "tcell" are used by this pipeline.
EXPORTS = {
    "epitope":      ("epitope_full_v3.zip",                  "all epitopes"),
    "tcell":        ("tcell_full_v3.zip",                    "T cell assays"),
    "bcell":        ("bcell_full_v3_single_file.zip",        "B cell assays"),
    "mhc_ligand":   ("mhc_ligand_full_single_file.zip",      "MHC ligand assays (large)"),
    "tcr":          ("tcr_full_v3.zip",                      "TCR receptors"),
    "bcr":          ("bcr_full_v3.zip",                      "BCR receptors"),
    "receptor":     ("receptor_full_v3.zip",                 "all receptors"),
    "reference":    ("reference_full_v3.zip",                "references"),
}

UA = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/zip,application/octet-stream,*/*",
}
MAX_RETRIES = 5
RETRY_BACKOFF_S = 60  # IEDB's mod_qos throttle window is ~minute-scale


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def head_size(url: str) -> int | None:
    """HEAD-style probe via Range request — returns Content-Length if known."""
    try:
        req = Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urlopen(req) as resp:
            cr = resp.headers.get("Content-Range")
            if cr and "/" in cr:
                total = cr.rsplit("/", 1)[-1]
                if total.isdigit():
                    return int(total)
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None


def _stream_once(url: str, dst: Path, chunk: int) -> int:
    req = Request(url, headers=UA)
    with urlopen(req) as resp, dst.open("wb") as fh:
        total = resp.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        n = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            n += len(buf)
            if total:
                pct = 100 * n / total
                sys.stderr.write(f"\r  {dst.name}: {human(n)} / {human(total)} ({pct:.1f}%)")
            else:
                sys.stderr.write(f"\r  {dst.name}: {human(n)}")
            sys.stderr.flush()
    sys.stderr.write("\n")
    return n


def stream_download(url: str, dst: Path, chunk: int = 1 << 20) -> int:
    """Download with retries — IEDB's mod_qos returns 200 with empty body when throttled."""
    import time
    for attempt in range(1, MAX_RETRIES + 1):
        n = _stream_once(url, dst, chunk)
        if n > 0:
            return n
        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_S * attempt
            print(f"  empty response (mod_qos throttle?). retry {attempt}/{MAX_RETRIES - 1} in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return 0


def extract(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            target = out_dir / name
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)
            written.append(target)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o", "--out", type=Path, default=Path("iedb_data"),
        help="output directory (default: ./iedb_data)",
    )
    ap.add_argument(
        "--only", nargs="+", choices=list(EXPORTS), metavar="KEY",
        help=f"download only these exports (default: all). Choices: {', '.join(EXPORTS)}",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="only probe sizes; don't download",
    )
    ap.add_argument(
        "--keep-zip", action="store_true",
        help="keep the .zip after extraction (default: delete)",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="skip exports whose extracted CSVs already exist",
    )
    args = ap.parse_args()

    keys = args.only or ["tcr", "tcell"]
    args.out.mkdir(parents=True, exist_ok=True)
    zip_dir = args.out / "_zips"
    zip_dir.mkdir(exist_ok=True)

    if args.check:
        print(f"{'export':<14} {'size':>12}  description")
        for k in keys:
            fname, desc = EXPORTS[k]
            size = head_size(BASE + fname)
            shown = human(size) if size else "?"
            print(f"{k:<14} {shown:>12}  {desc}")
        return 0

    summary: list[tuple[str, int, list[Path]]] = []
    for k in keys:
        fname, desc = EXPORTS[k]
        url = BASE + fname
        zip_path = zip_dir / fname

        if args.skip_existing:
            stem = fname.removesuffix(".zip")
            existing = list(args.out.glob(f"{stem}*.csv"))
            if existing:
                print(f"[skip] {k}: {len(existing)} CSV(s) already present")
                summary.append((k, sum(p.stat().st_size for p in existing), existing))
                continue

        print(f"\n[{k}] {desc} -> {url}")
        n = stream_download(url, zip_path)
        if n == 0:
            print(f"  ERROR: server returned empty body after {MAX_RETRIES} attempts (rate-limited)")
            zip_path.unlink(missing_ok=True)
            continue
        try:
            files = extract(zip_path, args.out)
        except zipfile.BadZipFile:
            print(f"  ERROR: {zip_path} is not a valid zip — leaving in place for inspection")
            continue
        if not args.keep_zip:
            zip_path.unlink()
        summary.append((k, n, files))
        for f in files:
            print(f"  extracted {f.name} ({human(f.stat().st_size)})")

    if not args.keep_zip:
        try:
            zip_dir.rmdir()
        except OSError:
            pass

    print("\n=== summary ===")
    for k, n, files in summary:
        print(f"  {k:<14} {human(n):>12}  {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
