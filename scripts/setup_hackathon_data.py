"""Unpack the professor's Hackathon_export.zip into data/hackathon (inner zips included).

    python scripts/setup_hackathon_data.py path/to/Hackathon_export.zip
    python scripts/setup_hackathon_data.py path/to/already-extracted-folder

Then build the knowledge base:  python -m backend.rag.ingest
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "hackathon"
EXPECTED = ["FoodSecurity", "Clinical", "UnitedStatesSecuritiesandExchangeCommission(SEC)", "Retail"]


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if root != target and root not in target.parents:  # zip-slip guard
            raise ValueError(f"unsafe path in archive: {member.filename}")
    zf.extractall(dest)


def unpack_inner(dest: Path) -> None:
    for inner in sorted(dest.glob("*.zip")):
        print(f"  extracting {inner.name}")
        with zipfile.ZipFile(inner) as zf:
            safe_extract(zf, dest)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1]).expanduser()
    DEST.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        print(f"Copying {src} → {DEST}")
        shutil.copytree(src, DEST, dirs_exist_ok=True)
    elif src.suffix.lower() == ".zip" and src.exists():
        print(f"Extracting {src.name} → {DEST}")
        with zipfile.ZipFile(src) as zf:
            safe_extract(zf, DEST)
    else:
        print(f"Not found: {src}")
        return 1
    # The export may wrap everything in one top-level folder; flatten it.
    subdirs = [d for d in DEST.iterdir() if d.is_dir()]
    if not any((DEST / e).exists() for e in EXPECTED) and len(subdirs) == 1:
        for item in subdirs[0].iterdir():
            shutil.move(str(item), DEST / item.name)
        subdirs[0].rmdir()
    unpack_inner(DEST)
    missing = [e for e in EXPECTED if not (DEST / e).exists()]
    print("Found:", ", ".join(sorted(p.name for p in DEST.iterdir())))
    if missing:
        print("Warning, missing folders:", missing)
    print("\nNext: python -m backend.rag.ingest      (add --no-embed to skip the embedding model download)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
