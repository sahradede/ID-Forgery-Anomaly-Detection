"""Evaluation metrics for forgery detection.

Scores follow one convention everywhere: higher score = more likely forged.
APCER: share of forged documents accepted as genuine.
BPCER: share of genuine documents rejected as forged.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(labels, scores) -> float:
    """Threshold-free separability of genuine (0) and forged (1) documents."""
    return float(roc_auc_score(labels, scores))


def threshold_for_bpcer(genuine_scores, bpcer: float = 0.05) -> float:
    """Threshold that rejects about `bpcer` of the given genuine documents."""
    return float(np.quantile(np.asarray(genuine_scores), 1 - bpcer))


def pad_metrics(labels, scores, threshold: float) -> dict:
    """APCER, BPCER and ACER at a fixed threshold."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    forged = labels == 1
    apcer = float((scores[forged] < threshold).mean()) if forged.any() else float("nan")
    bpcer = float((scores[~forged] >= threshold).mean()) if (~forged).any() else float("nan")
    return {"apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2}
