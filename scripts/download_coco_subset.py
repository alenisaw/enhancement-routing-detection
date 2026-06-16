"""Download COCO val2017 images and instances_val2017.json into data/raw/coco/.

Uses only Python standard library (urllib, zipfile).  Safe to re-run: already
present files are skipped.  Zip archives are cached in data/raw/coco/.cache/ so
they can be reused without re-downloading.

Usage
-----
  # Full download (~6.4 GB images + ~241 MB annotations)
  python scripts/download_coco_subset.py

  # Annotations only (skip 6 GB image download)
  python scripts/download_coco_subset.py --skip-images

  # Extract only 50 images (for a quick smoke test)
  python scripts/download_coco_subset.py --limit 50

  # Custom destination
  python scripts/download_coco_subset.py --dest path/to/coco
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
_INSTANCES_ENTRY = "annotations/instances_val2017.json"


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    mb = downloaded / 1_048_576
    if total_size > 0:
        pct = min(downloaded / total_size * 100.0, 100.0)
        total_mb = total_size / 1_048_576
        sys.stdout.write(f"\r  {pct:5.1f}%  {mb:,.1f} / {total_mb:,.1f} MB")
    else:
        sys.stdout.write(f"\r  {mb:,.1f} MB downloaded")
    sys.stdout.flush()


def _done() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, label: str) -> Path:
    """Download *url* to *dest*, skipping if the file already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Already downloaded: {dest.name}  (skipping)")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  Downloading {label}")
    print(f"  URL: {url}")
    try:
        urlretrieve(url, tmp, reporthook=_reporthook)
        _done()
        tmp.rename(dest)
        print(f"  Saved: {dest}")
    except BaseException:
        _done()
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_annotations(zip_path: Path, dest_dir: Path) -> Path:
    out = dest_dir / _INSTANCES_ENTRY
    if out.exists() and out.stat().st_size > 0:
        print(f"  Already extracted: {out}")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {_INSTANCES_ENTRY} …")
    with zipfile.ZipFile(zip_path) as zf:
        if _INSTANCES_ENTRY not in zf.namelist():
            raise KeyError(f"{_INSTANCES_ENTRY!r} not found inside {zip_path.name}")
        zf.extract(_INSTANCES_ENTRY, dest_dir)
    print(f"  -> {out}")
    return out


def extract_images(zip_path: Path, dest_dir: Path, limit: int | None) -> int:
    """Extract image files from *zip_path* into *dest_dir*/val2017/.

    When *limit* is given only the first *limit* entries are extracted, which
    is enough for a smoke test without unpacking the full 6 GB archive.
    """
    with zipfile.ZipFile(zip_path) as zf:
        all_entries = [
            name for name in zf.namelist()
            if name.startswith("val2017/") and not name.endswith("/")
        ]

    if not all_entries:
        raise RuntimeError(f"No val2017/* image entries found in {zip_path.name}")

    entries = all_entries[:limit] if limit is not None else all_entries
    missing = [e for e in entries if not (dest_dir / e).exists()]

    if not missing:
        note = f" (first {limit})" if limit is not None else ""
        print(f"  All {len(entries)}{note} images already extracted  (skipping)")
        return len(entries)

    target = dest_dir / "val2017"
    target.mkdir(parents=True, exist_ok=True)

    note = f" (first {limit} of {len(all_entries)})" if limit is not None else f" of {len(all_entries)}"
    print(f"  Extracting {len(missing)} images{note} …")

    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(missing, 1):
            sys.stdout.write(f"\r  {i}/{len(missing)}")
            sys.stdout.flush()
            zf.extract(name, dest_dir)

    _done()
    return len(entries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COCO val2017 images and instances_val2017.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        default="data/raw/coco",
        help="Root directory for COCO data (default: data/raw/coco).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Extract only the first N images from the archive (for smoke tests).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Download annotations only; skip the 6.4 GB image archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)
    cache_dir = dest / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=== COCO val2017 downloader ===")
    print(f"Destination : {dest.resolve()}")
    if args.limit:
        print(f"Image limit : {args.limit} (smoke-test mode)")
    print()

    # Step 1: annotations (~241 MB zip, extract one JSON)
    print("[1/2] Annotations")
    ann_zip = cache_dir / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, ann_zip, "annotations_trainval2017.zip  (~241 MB)")
    extract_annotations(ann_zip, dest)

    # Step 2: images (~6.4 GB zip)
    if args.skip_images:
        print("\n[2/2] Images  — skipped (--skip-images)")
    else:
        print("\n[2/2] Images")
        img_zip = cache_dir / "val2017.zip"
        download_file(IMAGES_URL, img_zip, "val2017.zip  (~6.4 GB)")
        n = extract_images(img_zip, dest, limit=args.limit)
        print(f"  {n} image(s) ready in {dest / 'val2017'}")

    print("\nDone.  Expected layout:")
    print(f"  {dest}/")
    print(f"  ├── val2017/                    ← JPEG images")
    print(f"  ├── annotations/")
    print(f"  │   └── instances_val2017.json  ← required annotation file")
    print(f"  └── .cache/                     ← zip archives (can be deleted)")


if __name__ == "__main__":
    main()
