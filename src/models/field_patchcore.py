"""Field-aware PatchCore for identity documents, fitted on genuine documents only.

Plain PatchCore scores a document by its most anomalous patch. On ID documents that patch
almost always lies in a field that varies between genuine documents (face, signature), so
the score ignores the forged field. For each document type this model uses:
1. a memory bank of all genuine training patches (no coreset),
2. a fixed box per field: the union of the annotated boxes of the training documents,
3. per-field score statistics from the training documents, computed leave-one-document-out
   so that a document is never matched with its own patches.
A field score is the largest patch distance inside the field box, z-normalized with that
field's statistics. The document score is the largest field z-score, and that field is the
predicted forged field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.datasets import SIDTD_DIR, SIDTD_Q75_DIR, sidtd_dataset, sidtd_rows
from src.evaluation.metrics import auroc, pad_metrics, threshold_for_bpcer
from src.models.patchcore import PatchFeatures, extract

ROOT = Path(__file__).resolve().parents[2]
TEST_TYPES = ["Inpaint_and_Rewrite", "Crop_and_Replace"]


def field_layout(doc_type: str, sources) -> dict[str, tuple[float, float, float, float]]:
    """Union of the annotated boxes of each field over the given genuine documents, in pixels."""
    path = SIDTD_DIR / "templates/Annotations/reals" / f"{doc_type}.json"
    annotations = json.loads(path.read_text(encoding="utf-8"))
    wanted = {s.rsplit("_", 1)[1] for s in sources}
    layout: dict[str, list[float]] = {}
    for entry in annotations["_via_img_metadata"].values():
        if Path(entry["filename"]).stem not in wanted:
            continue
        for region in entry["regions"]:
            b = region["shape_attributes"]
            if b.get("name") != "rect":
                continue
            field = region["region_attributes"]["field_name"]
            box = [b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]
            if field in layout:
                old = layout[field]
                box = [min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]), max(old[3], box[3])]
            layout[field] = box
    return {field: tuple(box) for field, box in layout.items()}


def to_map_boxes(layout: dict, image_size: tuple[int, int], map_size: tuple[int, int]) -> dict:
    """Convert pixel boxes to feature-map cells: field -> (x0, y0, x1, y1)."""
    W0, H0 = image_size
    h, w = map_size
    boxes = {}
    for field, (x0, y0, x1, y1) in layout.items():
        mx0, my0 = min(int(x0 * w / W0), w - 1), min(int(y0 * h / H0), h - 1)
        mx1 = min(max(mx0 + 1, int(np.ceil(x1 * w / W0))), w)
        my1 = min(max(my0 + 1, int(np.ceil(y1 * h / H0))), h)
        boxes[field] = (mx0, my0, mx1, my1)
    return boxes


@torch.no_grad()
def nn_maps(feature_maps: torch.Tensor, memory: torch.Tensor, memory_owner: torch.Tensor | None = None,
            query_ids: list[int] | None = None, chunk: int = 512) -> torch.Tensor:
    """Nearest-neighbour distance of every patch to the memory bank: (N, h, w).

    With `memory_owner`, patches of the same document (`query_ids[i]`) are excluded.
    """
    maps = []
    for i in range(len(feature_maps)):
        f = feature_maps[i].to(memory.device)
        c, h, w = f.shape
        patches = f.permute(1, 2, 0).reshape(-1, c)
        exclude = (memory_owner == query_ids[i]) if memory_owner is not None else None
        mins = []
        for j in range(0, len(patches), chunk):
            d = torch.cdist(patches[j:j + chunk], memory)
            if exclude is not None:
                d[:, exclude] = float("inf")
            mins.append(d.min(dim=1).values)
        maps.append(torch.cat(mins).reshape(h, w).cpu())
    return torch.stack(maps)


def field_scores(maps: torch.Tensor, boxes: dict) -> np.ndarray:
    """Largest patch distance inside each field box: (N, n_fields)."""
    return np.array([[m[y0:y1, x0:x1].max().item() for (x0, y0, x1, y1) in boxes.values()] for m in maps])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-types", nargs="+", default=None, help="default: all document types")
    parser.add_argument("--exclude-fields", nargs="*", default=[], help="fields left out of the score")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="default", help="results subfolder")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    doc_types = args.doc_types or sorted(sidtd_rows("train")["doc_type"].unique())
    model = PatchFeatures().to(device).eval()
    out_dir = ROOT / "results" / "field_patchcore" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_parts, test_parts = [], []
    for doc_type in doc_types:
        train_ds = sidtd_dataset("train", doc_types=[doc_type], genuine_only=True)
        val_ds = sidtd_dataset("val", doc_types=[doc_type], genuine_only=True)
        test_ds = sidtd_dataset("test", doc_types=[doc_type])

        train_feat = extract(model, train_ds, device, args.batch_size, args.workers)
        n, c, h, w = train_feat.shape
        memory = train_feat.permute(0, 2, 3, 1).reshape(-1, c).to(device)
        owner = torch.arange(n, device=device).repeat_interleave(h * w)

        layout = field_layout(doc_type, train_ds.rows["source"])
        for field in args.exclude_fields:
            layout.pop(field, None)
        image_size = Image.open(SIDTD_Q75_DIR / train_ds.rows["image_path"].iloc[0]).size
        boxes = to_map_boxes(layout, image_size, (h, w))
        fields = list(boxes)

        train_fs = field_scores(nn_maps(train_feat, memory, owner, list(range(n))), boxes)
        mu, sd = train_fs.mean(axis=0), train_fs.std(axis=0) + 1e-8

        def score(ds):
            fs = field_scores(nn_maps(extract(model, ds, device, args.batch_size, args.workers), memory), boxes)
            z = (fs - mu) / sd
            return z.max(axis=1), [fields[k] for k in z.argmax(axis=1)]

        val_scores, _ = score(val_ds)
        test_scores, top_fields = score(test_ds)
        val_parts.append(pd.DataFrame({"doc_type": doc_type, "score": val_scores}))
        test = test_ds.rows[["image_path", "label", "doc_type", "forgery_type", "field"]].copy()
        test["score"] = test_scores
        test["top_field"] = top_fields
        test_parts.append(test)
        print(f"{doc_type}: {len(fields)} fields, memory {len(memory)} patches | "
              f"test AUROC (all forgeries) {auroc(test['label'], test['score']):.3f}")

    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    test.to_csv(out_dir / "test_scores.csv", index=False)

    threshold = threshold_for_bpcer(val["score"], bpcer=0.05)
    results = {"doc_types": doc_types, "excluded_fields": args.exclude_fields,
               "threshold": threshold, "test": {}}
    for forgery_type in TEST_TYPES:
        subset = test[(test["label"] == 0) | (test["forgery_type"] == forgery_type)]
        forged = subset[subset["label"] == 1]
        if forged.empty:
            continue
        results["test"][forgery_type] = {
            "auroc": auroc(subset["label"], subset["score"]),
            **pad_metrics(subset["label"], subset["score"], threshold),
            "field_hit_rate": float((forged["top_field"] == forged["field"]).mean()),
            "n_genuine": int((subset["label"] == 0).sum()),
            "n_forged": int(len(forged)),
        }

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
