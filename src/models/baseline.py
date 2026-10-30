"""Supervised baseline: EfficientNet-B0 fine-tuned to classify genuine vs forged SIDTD templates.

E1 (--train-forgeries all): trained on both forgery types.
E2 (--train-forgeries inpaint): trained on inpainting only, tested on unseen Crop & Replace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2

from src.data.datasets import DEFAULT_SIZE, IMAGENET_MEAN, IMAGENET_STD, sidtd_dataset
from src.evaluation.metrics import auroc, pad_metrics, threshold_for_bpcer

ROOT = Path(__file__).resolve().parents[2]
FORGERY_SETS = {"all": None, "inpaint": ["Inpaint_and_Rewrite"]}
TEST_TYPES = ["Inpaint_and_Rewrite", "Crop_and_Replace"]


def train_transform(size: tuple[int, int] = DEFAULT_SIZE):
    """Mild photometric augmentation only: geometric changes could mimic forgery artifacts."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size, antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model() -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and forgery probabilities."""
    model.eval()
    labels, scores = [], []
    for batch in loader:
        logits = model(batch["image"].to(device)).squeeze(1)
        scores.append(torch.sigmoid(logits).cpu())
        labels.append(batch["label"])
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-forgeries", choices=FORGERY_SETS, default="inpaint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    forgeries = FORGERY_SETS[args.train_forgeries]

    train_ds = sidtd_dataset("train", transform=train_transform(), forgery_types=forgeries)
    val_ds = sidtd_dataset("val", forgery_types=forgeries)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, **loader_args)
    print(f"device {device} | train {len(train_ds)} | val {len(val_ds)}")

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    out_dir = ROOT / "results" / "baseline" / args.train_forgeries
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].float().to(device)
            loss = criterion(model(x).squeeze(1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        y_val, s_val = predict(model, val_loader, device)
        val_auroc = auroc(y_val, s_val)
        print(f"epoch {epoch}/{args.epochs} | train loss {total_loss / len(train_ds):.4f} | val AUROC {val_auroc:.4f}")
        if val_auroc > best_val:
            best_val = val_auroc
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    y_val, s_val = predict(model, val_loader, device)
    threshold = threshold_for_bpcer(s_val[y_val == 0], bpcer=0.05)

    results = {"train_forgeries": args.train_forgeries, "best_val_auroc": best_val,
               "threshold": threshold, "test": {}}
    for forgery_type in TEST_TYPES:
        test_ds = sidtd_dataset("test", forgery_types=[forgery_type])
        y, s = predict(model, DataLoader(test_ds, **loader_args), device)
        results["test"][forgery_type] = {
            "auroc": auroc(y, s), **pad_metrics(y, s, threshold),
            "n_genuine": int((y == 0).sum()), "n_forged": int((y == 1).sum()),
        }

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
