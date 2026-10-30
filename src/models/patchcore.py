"""PatchCore anomaly detection on SIDTD, fitted on genuine documents only.

Follows Roth et al. (2022), "Towards Total Recall in Industrial Anomaly Detection":
WideResNet-50 features from layer2 and layer3 with 3x3 local averaging, a memory bank of
genuine patch features reduced by greedy coreset subsampling, and the nearest-neighbour
distance as the anomaly score. One memory bank is built per document type. Scores are
z-normalized with the validation genuine documents of the same type so that they can be
pooled across types.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

from src.data.datasets import sidtd_dataset, sidtd_rows
from src.evaluation.metrics import auroc, pad_metrics, threshold_for_bpcer

ROOT = Path(__file__).resolve().parents[2]
TEST_TYPES = ["Inpaint_and_Rewrite", "Crop_and_Replace"]


class PatchFeatures(nn.Module):
    """Locally aware patch features from WideResNet-50 layer2 and layer3."""

    def __init__(self) -> None:
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.extractor = create_feature_extractor(backbone, {"layer2": "layer2", "layer3": "layer3"})
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x)
        layer2 = self.pool(feats["layer2"])
        layer3 = F.interpolate(self.pool(feats["layer3"]), size=layer2.shape[-2:],
                               mode="bilinear", align_corners=False)
        return torch.cat([layer2, layer3], dim=1)  # (B, 1536, H/8, W/8)


@torch.no_grad()
def extract(model: nn.Module, dataset, device: torch.device,
            batch_size: int = 8, workers: int = 4) -> torch.Tensor:
    """Feature maps for every image of a dataset, kept on the CPU: (N, C, h, w)."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    return torch.cat([model(batch["image"].to(device)).cpu() for batch in loader])


@torch.no_grad()
def greedy_coreset(features: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center subsampling computed on a random projection of the features."""
    n = len(features)
    k = max(1, int(n * ratio))
    gen = torch.Generator().manual_seed(seed)
    proj = (torch.randn(features.shape[1], proj_dim, generator=gen) / proj_dim ** 0.5).to(features.device)
    z = features @ proj
    first = int(torch.randint(n, (1,), generator=gen))
    selected = [first]
    min_dist = torch.cdist(z, z[first:first + 1]).squeeze(1)
    for _ in range(k - 1):
        idx = int(torch.argmax(min_dist))
        selected.append(idx)
        min_dist = torch.minimum(min_dist, torch.cdist(z, z[idx:idx + 1]).squeeze(1))
    return features[selected]


@torch.no_grad()
def anomaly_maps(feature_maps: torch.Tensor, memory: torch.Tensor, chunk: int = 2) -> torch.Tensor:
    """Distance of every patch to its nearest genuine patch: (N, h, w)."""
    maps = []
    for i in range(0, len(feature_maps), chunk):
        f = feature_maps[i:i + chunk].to(memory.device)
        n, c, h, w = f.shape
        patches = f.permute(0, 2, 3, 1).reshape(-1, c)
        maps.append(torch.cdist(patches, memory).min(dim=1).values.reshape(n, h, w).cpu())
    return torch.cat(maps)


def image_scores(maps: torch.Tensor) -> np.ndarray:
    """Image score = the most anomalous patch."""
    return maps.flatten(1).max(dim=1).values.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-types", nargs="+", default=None, help="default: all document types")
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="default", help="results subfolder")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    doc_types = args.doc_types or sorted(sidtd_rows("train")["doc_type"].unique())
    model = PatchFeatures().to(device).eval()
    out_dir = ROOT / "results" / "patchcore" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_parts, test_parts = [], []
    for doc_type in doc_types:
        train_ds = sidtd_dataset("train", doc_types=[doc_type], genuine_only=True)
        val_ds = sidtd_dataset("val", doc_types=[doc_type], genuine_only=True)
        test_ds = sidtd_dataset("test", doc_types=[doc_type])

        train_maps = extract(model, train_ds, device, args.batch_size, args.workers)
        patches = train_maps.permute(0, 2, 3, 1).reshape(-1, train_maps.shape[1])
        memory = greedy_coreset(patches.to(device), args.coreset_ratio, seed=args.seed)

        val_scores = image_scores(anomaly_maps(extract(model, val_ds, device, args.batch_size, args.workers), memory))
        test_scores = image_scores(anomaly_maps(extract(model, test_ds, device, args.batch_size, args.workers), memory))
        mu, sd = val_scores.mean(), val_scores.std() + 1e-8

        val_parts.append(pd.DataFrame({"doc_type": doc_type, "score": val_scores, "z": (val_scores - mu) / sd}))
        test = test_ds.rows[["image_path", "label", "doc_type", "forgery_type"]].copy()
        test["score"] = test_scores
        test["z"] = (test_scores - mu) / sd
        test_parts.append(test)
        print(f"{doc_type}: memory {len(memory)} patches | test AUROC (all forgeries) "
              f"{auroc(test['label'], test['score']):.3f}")

    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    test.to_csv(out_dir / "test_scores.csv", index=False)

    threshold = threshold_for_bpcer(val["z"], bpcer=0.05)
    results = {"doc_types": doc_types, "coreset_ratio": args.coreset_ratio, "threshold_z": threshold, "test": {}}
    for forgery_type in TEST_TYPES:
        subset = test[(test["label"] == 0) | (test["forgery_type"] == forgery_type)]
        if (subset["label"] == 1).sum() == 0:
            continue
        results["test"][forgery_type] = {
            "auroc": auroc(subset["label"], subset["z"]),
            **pad_metrics(subset["label"], subset["z"], threshold),
            "n_genuine": int((subset["label"] == 0).sum()),
            "n_forged": int((subset["label"] == 1).sum()),
        }

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
