"""PyTorch datasets for SIDTD and FantasyID with a shared interface.

Rows come from the split files in `splits/`. Every item is a dict:
{"image": tensor, "label": int, "path": str}, with label 0 = genuine and 1 = forged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[2]
SIDTD_DIR = ROOT / "data" / "sidtd"
SIDTD_Q75_DIR = ROOT / "data" / "sidtd_q75"  # all images re-encoded at JPEG quality 75, 4:2:0
FANTASY_DIR = ROOT / "data" / "fantasyid" / "FantasyID"
SPLITS_DIR = ROOT / "splits"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SIZE = (256, 384)  # height, width: ID cards are about 1.5 times wider than tall


def default_transform(size: tuple[int, int] = DEFAULT_SIZE) -> Callable:
    """Resize keeping the card aspect ratio, convert to float and normalize like ImageNet."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class IDDocumentDataset(Dataset):
    """Loads document images listed in a DataFrame with `image_path` and `label` columns."""

    def __init__(self, rows: pd.DataFrame, root: Path, transform: Callable | None = None):
        self.rows = rows.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows.iloc[i]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        return {"image": self.transform(image), "label": int(row["label"]), "path": row["image_path"]}


def sidtd_rows(split: str, doc_types: Sequence[str] | None = None,
               forgery_types: Sequence[str] | None = None, genuine_only: bool = False) -> pd.DataFrame:
    """Select rows of the grouped SIDTD split.

    `forgery_types` filters forged rows only; genuine rows are always kept.
    """
    df = pd.read_csv(SPLITS_DIR / "sidtd_grouped.csv")
    df = df[df["split"] == split]
    if doc_types is not None:
        df = df[df["doc_type"].isin(doc_types)]
    if forgery_types is not None:
        df = df[(df["label"] == 0) | df["forgery_type"].isin(forgery_types)]
    if genuine_only:
        df = df[df["label"] == 0]
    return df


def fantasyid_rows(template: str | None = None, folds: Sequence[int] | None = None,
                   attacks: Sequence[str] | None = None, official_split: str | None = None,
                   genuine_only: bool = False) -> pd.DataFrame:
    """Select rows of the FantasyID index.

    `attacks` filters forged rows only; genuine rows are always kept.
    """
    df = pd.read_csv(SPLITS_DIR / "fantasyid_index.csv")
    if template is not None:
        df = df[df["template"] == template]
    if folds is not None:
        df = df[df["fold"].isin(folds)]
    if official_split is not None:
        df = df[df["official_split"] == official_split]
    if attacks is not None:
        df = df[(df["label"] == 0) | df["attack"].isin(attacks)]
    if genuine_only:
        df = df[df["label"] == 0]
    return df


def sidtd_dataset(split: str, transform: Callable | None = None, jpeg_q75: bool = True,
                  **filters) -> IDDocumentDataset:
    """jpeg_q75=True reads the re-encoded copy that removes the JPEG-quality shortcut."""
    root = SIDTD_Q75_DIR if jpeg_q75 else SIDTD_DIR
    return IDDocumentDataset(sidtd_rows(split, **filters), root, transform)


def fantasyid_dataset(transform: Callable | None = None, **filters) -> IDDocumentDataset:
    return IDDocumentDataset(fantasyid_rows(**filters), FANTASY_DIR, transform)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    checks = {
        "SIDTD train, all": sidtd_dataset("train"),
        "SIDTD test, Crop & Replace": sidtd_dataset("test", forgery_types=["Crop_and_Replace"]),
        "SIDTD train, esp_id genuine": sidtd_dataset("train", doc_types=["esp_id"], genuine_only=True),
        "FantasyID turkiye, fold 0 + attack-only cards": fantasyid_dataset(template="turkiye", folds=[0, -1]),
    }
    for name, ds in checks.items():
        print(f"{name}: {len(ds)} images, labels {ds.rows['label'].value_counts().to_dict()}")

    batch = next(iter(DataLoader(checks["SIDTD train, all"], batch_size=8, shuffle=True)))
    print("batch:", tuple(batch["image"].shape), batch["image"].dtype, batch["label"].tolist())
