"""Paired, stratified bootstrap of AUROC for two models scored on the same test images.

Genuine and forged images are resampled separately, so every replicate keeps the original
class counts. Each replicate scores the same images with both models, which gives a
confidence interval for the AUROC difference (model A minus model B).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import auroc

TEST_TYPES = ["Inpaint_and_Rewrite", "Crop_and_Replace"]


def load_scores(path: Path, forgery_type: str) -> pd.DataFrame:
    """Genuine images plus the forgeries of one type, indexed by image path."""
    df = pd.read_csv(path)
    if "test_set" in df.columns:  # supervised baseline output
        df = df[df["test_set"] == forgery_type]
    else:  # anomaly model output
        df = df[(df["label"] == 0) | (df["forgery_type"] == forgery_type)]
    return df.set_index("image_path")[["label", "score"]]


def interval(values: np.ndarray) -> list[float]:
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    joined = a.merge(b, left_index=True, right_index=True, suffixes=("_a", "_b"))
    assert (joined["label_a"] == joined["label_b"]).all(), "labels differ between the two score files"
    labels = joined["label_a"].to_numpy()
    sa, sb = joined["score_a"].to_numpy(), joined["score_b"].to_numpy()
    forged, genuine = np.where(labels == 1)[0], np.where(labels == 0)[0]

    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(forged, len(forged)), rng.choice(genuine, len(genuine))])
        reps.append((auroc(labels[idx], sa[idx]), auroc(labels[idx], sb[idx])))
    reps = np.array(reps)
    diff = reps[:, 0] - reps[:, 1]

    return {
        "n_genuine": int(len(genuine)), "n_forged": int(len(forged)),
        "auroc_a": round(auroc(labels, sa), 3), "ci_a": interval(reps[:, 0]),
        "auroc_b": round(auroc(labels, sb), 3), "ci_b": interval(reps[:, 1]),
        "diff_a_minus_b": round(auroc(labels, sa) - auroc(labels, sb), 3), "ci_diff": interval(diff),
        "share_of_replicates_with_diff_le_0": round(float((diff <= 0).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="test_scores.csv of model A")
    parser.add_argument("--b", type=Path, required=True, help="test_scores.csv of model B")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    results = {"a": str(args.a), "b": str(args.b)}
    for forgery_type in TEST_TYPES:
        results[forgery_type] = paired_bootstrap(load_scores(args.a, forgery_type),
                                                 load_scores(args.b, forgery_type), args.n_boot)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
