# Results

All SIDTD results use the source-grouped split (`splits/sidtd_grouped.csv`). Test sets: 198 genuine documents plus the forgeries of one type (214 inpainting, 30 Crop & Replace). The threshold is chosen on validation genuine documents at BPCER 5%.

## Supervised baseline (EfficientNet-B0, 256x384)

| Experiment | Train forgeries | JPEG | Epochs | AUROC inpainting | AUROC Crop & Replace |
| --- | --- | --- | --- | --- | --- |
| E2 | inpainting | raw | 10 | 1.000 | 1.000 |
| E1 | all | raw | 10 | 0.998 | 0.992 |
| E2 | inpainting | q75 | 10 | 0.849 | 0.677 |
| E1 | all | q75 | 10 | 0.835 | 0.591 |
| **E2** | **inpainting** | **q75** | **30** | **0.966** | **0.613** |
| **E1** | **all** | **q75** | **30** | **0.978** | **0.697** |

"raw" uses the original files, which contain the JPEG-quality shortcut; "q75" re-encodes all images at quality 75 (see `docs/decisions.md`). The 30-epoch q75 rows are the reference baseline.

### Error rates of the reference baseline (threshold at validation BPCER 5%)

| Experiment | Test forgeries | APCER | BPCER | ACER |
| --- | --- | --- | --- | --- |
| E2 | inpainting | 0.154 | 0.025 | 0.090 |
| E2 | Crop & Replace | 0.833 | 0.025 | 0.429 |
| E1 | inpainting | 0.089 | 0.051 | 0.070 |
| E1 | Crop & Replace | 0.733 | 0.051 | 0.392 |

### Observations
- Removing the JPEG-quality shortcut drops test AUROC from about 1.00 to 0.97-0.98 on inpainting and to 0.61-0.70 on Crop & Replace.
- Trained on inpainting only (E2), the model misses 83% of the unseen Crop & Replace forgeries at the chosen threshold.
- Even when Crop & Replace forgeries are in the training set (E1), AUROC on them is only 0.70: they are hard for this model at 256x384, possibly because the pasted field is small.
- Only 30 Crop & Replace test forgeries: these numbers have a wide uncertainty.

## Field-aware PatchCore (one-class, genuine documents only)

| Model | AUROC inpainting [95% CI] | AUROC Crop & Replace [95% CI] | Field hit rate (inpainting / C&R) |
| --- | --- | --- | --- |
| Plain PatchCore (esp_id only) | 0.55 | - | - |
| **Field-aware PatchCore, all fields** | **0.853 [0.815, 0.887]** | **0.743 [0.655, 0.828]** | 0.65 / 0.37 |
| Field-aware PatchCore, without face, photo, signature | 0.858 | 0.728 | 0.65 / 0.37 |

Per document type (all fields, all forgeries): alb_id 0.867, aze_passport 0.855, esp_id 0.924, est_id 0.871, fin_id 0.845, grc_passport 0.763, lva_passport 0.840, rus_internalpassport 0.883, srb_passport 0.650, svk_id 0.801.

### Paired bootstrap: field-aware PatchCore minus supervised baseline (2,000 replicates)

| Supervised model | Test forgeries | AUROC difference | 95% CI |
| --- | --- | --- | --- |
| E2 (inpainting only) | Inpainting (seen) | -0.113 | [-0.147, -0.080] |
| E2 (inpainting only) | Crop & Replace (unseen) | **+0.130** | **[+0.005, +0.261]** |
| E1 (all forgeries) | Inpainting (seen) | -0.125 | [-0.161, -0.092] |
| E1 (all forgeries) | Crop & Replace (seen) | +0.046 | [-0.088, +0.177] |

### Observations
- On the forgery type the supervised model never saw, the one-class model is significantly better; the lower bound of the interval is close to zero (30 forged test documents), so the result is borderline.
- Without seeing any forgery, the one-class model matches the supervised model trained with Crop & Replace (no significant difference).
- On forgery types seen during training the supervised model is clearly better, which motivates a hybrid model.
