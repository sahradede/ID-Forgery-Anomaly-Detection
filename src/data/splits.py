"""Build a grouped train/val/test split for the SIDTD template images.

Each genuine document and every forgery derived from it end up in the same split.
Crop & Replace forgeries take a field from a second source document, so documents
linked this way are grouped together and always share a split. The official SIDTD
split shares 252 source documents between train and test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIDTD_DIR = ROOT / "data" / "sidtd"
OUT_DIR = ROOT / "splits"

SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


def load_sidtd_index(sidtd_dir: Path = SIDTD_DIR) -> pd.DataFrame:
    """Return one row per template image with label, document type and source info."""
    rows = []
    for path in sorted((sidtd_dir / "templates/Images/reals").glob("*.jpg")):
        source = path.stem  # e.g. esp_id_42
        rows.append({
            "image_path": str(path.relative_to(sidtd_dir)),
            "label": 0,
            "doc_type": source.rsplit("_", 1)[0],
            "source": source,
            "forgery_type": "none",
            "field": "none",
            "second_source": "none",
        })

    ann_dir = sidtd_dir / "templates/Annotations/fakes"
    for path in sorted((sidtd_dir / "templates/Images/fakes").glob("*.jpg")):
        meta = json.loads((ann_dir / f"{path.stem}.json").read_text(encoding="utf-8"))
        source = Path(meta["src"]).stem
        second = meta.get("second_src", "None")
        rows.append({
            "image_path": str(path.relative_to(sidtd_dir)),
            "label": 1,
            "doc_type": source.rsplit("_", 1)[0],
            "source": source,
            "forgery_type": meta["ctype"],
            "field": meta.get("field", "none"),
            "second_source": Path(second).stem if second not in (None, "None") else "none",
        })
    return pd.DataFrame(rows)


def group_sources(index: pd.DataFrame) -> dict[str, str]:
    """Map each genuine document to a group id.

    Documents linked by a forgery (source and second source) share a group (union-find).
    """
    parent = {s: s for s in index.loc[index["label"] == 0, "source"]}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    linked = index[(index["label"] == 1) & (index["second_source"] != "none")]
    for a, b in zip(linked["source"], linked["second_source"]):
        if a in parent and b in parent:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    return {s: find(s) for s in parent}


def assign_splits(index: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign whole groups to splits, aiming at the target ratios per document type."""
    rng = np.random.default_rng(seed)
    group_of = group_sources(index)
    reals = index[index["label"] == 0].copy()
    reals["group"] = reals["source"].map(group_of)

    group_split: dict[str, str] = {}
    for _, docs in reals.groupby("doc_type"):
        sizes = docs.groupby("group").size()
        groups = sorted(g for g in sizes.index if g not in group_split)
        rng.shuffle(groups)
        n_train = round(len(docs) * SPLIT_RATIOS["train"])
        n_val = round(len(docs) * SPLIT_RATIOS["val"])
        filled = {"train": 0, "val": 0}
        for g in groups:
            if filled["train"] < n_train:
                split = "train"
            elif filled["val"] < n_val:
                split = "val"
            else:
                split = "test"
            group_split[g] = split
            if split in filled:
                filled[split] += int(sizes[g])

    source_split = {s: group_split[g] for s, g in group_of.items()}
    out = index.copy()
    out["group"] = out["source"].map(group_of)
    out["split"] = out["source"].map(source_split)
    out["second_source_split"] = out["second_source"].map(source_split).fillna("none")
    out["cross_split_source"] = (
        (out["second_source_split"] != "none") & (out["second_source_split"] != out["split"])
    )
    return out


def check_split(df: pd.DataFrame) -> None:
    """Fail loudly if any source document is shared between splits."""
    missing = int(df["split"].isna().sum())
    assert missing == 0, f"{missing} images have a source without a genuine document"
    assert (df.groupby("source")["split"].nunique() == 1).all(), "a source appears in more than one split"
    assert not df["cross_split_source"].any(), "a second source lies in another split"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = assign_splits(load_sidtd_index(), seed=args.seed)
    check_split(df)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "sidtd_grouped.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path.relative_to(ROOT)}")
    print(pd.crosstab([df["split"], df["label"]], df["forgery_type"]).to_string())
    group_sizes = df[df["label"] == 0].groupby("group").size()
    print("Linked groups with more than one document:", int((group_sizes > 1).sum()),
          "| largest group:", int(group_sizes.max()))


if __name__ == "__main__":
    main()
