"""Build an index of FantasyID images and card-grouped folds per template.

The file stem identifies the physical card (e.g. turkiye-041_03): the same stem is used
for the three capture devices and for every manipulation of that card. Folds are
assigned per card, so a card's genuine images and all of its manipulations share a fold.
Cards that only appear as attacks (no genuine images) get fold -1; they can be added to
every test fold without leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FANTASY_DIR = ROOT / "data" / "fantasyid" / "FantasyID"
OUT_DIR = ROOT / "splits"


def load_fantasyid_index(root: Path = FANTASY_DIR) -> pd.DataFrame:
    """Return one row per image with official split, label, attack, device and card id."""
    rows = []
    for path in sorted(root.rglob("*.jpg")):
        parts = path.relative_to(root).parts
        if parts[0] not in ("train", "test"):
            continue
        is_genuine = parts[1] == "bonafide"
        rows.append({
            "image_path": str(path.relative_to(root)),
            "official_split": parts[0],
            "label": 0 if is_genuine else 1,
            "attack": "none" if is_genuine else parts[2],
            "device": parts[-2],
            "template": path.stem.split("-")[0],
            "card_id": path.stem,
        })
    return pd.DataFrame(rows)


def assign_card_folds(index: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign each card with genuine images to a fold, separately for each template."""
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    genuine_cards = index[index["label"] == 0].groupby("template")["card_id"].unique()
    for template in sorted(genuine_cards.index):
        cards = sorted(genuine_cards[template])
        rng.shuffle(cards)
        for i, card in enumerate(cards):
            fold_of[card] = i % n_folds

    out = index.copy()
    out["fold"] = out["card_id"].map(fold_of).fillna(-1).astype(int)
    return out


def check_folds(df: pd.DataFrame) -> None:
    """Fail loudly if a card spans folds or a genuine image has no fold."""
    assert (df.groupby("card_id")["fold"].nunique() == 1).all(), "a card appears in more than one fold"
    assert (df.loc[df["label"] == 0, "fold"] >= 0).all(), "a genuine image has no fold"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = assign_card_folds(load_fantasyid_index(), n_folds=args.folds, seed=args.seed)
    check_folds(df)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "fantasyid_index.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path.relative_to(ROOT)}")

    has_genuine = df.groupby(["template", "card_id"])["label"].min() == 0
    summary = has_genuine.groupby("template").agg(
        cards_with_genuine="sum",
        attack_only_cards=lambda s: int((~s).sum()),
    )
    print(summary.to_string())

    turkish = df[df["template"] == "turkiye"]
    print("\nTurkish template, images per fold and attack:")
    print(pd.crosstab(turkish["fold"], turkish["attack"]).to_string())


if __name__ == "__main__":
    main()
