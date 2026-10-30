"""Re-encode every SIDTD template image as JPEG quality 75 with 4:2:0 subsampling.

Genuine SIDTD templates are stored at JPEG quality 92 (4:4:4), forged ones at quality 75
(4:2:0, the Pillow default). A classifier can separate them from the compression alone.
Re-encoding all images with the forgeries' settings removes this shortcut: forged images
barely change, genuine images get the same compression as the forgeries.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "sidtd"
DST = ROOT / "data" / "sidtd_q75"


def main() -> None:
    files = sorted((SRC / "templates" / "Images").rglob("*.jpg"))
    for i, src in enumerate(files, 1):
        dst = DST / src.relative_to(SRC)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.open(src).convert("RGB").save(dst, "JPEG", quality=75, subsampling=2)
        if i % 500 == 0:
            print(f"{i}/{len(files)}")
    print(f"Done: {len(files)} images in {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
