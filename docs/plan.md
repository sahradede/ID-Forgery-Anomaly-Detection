# Project plan

**Goal:** detect forged identity documents captured with a smartphone, using anomaly detection models fitted only on genuine documents, and compare them with supervised models on forgery types they have never seen.

**Deliverables:** reproducible experiments, a working product (iPhone app with a live on-device check and a server-side heatmap analysis), and a final report.

## Timeline

| Week | Work | Gate / output | Status |
| --- | --- | --- | --- |
| 1 | Environment, SIDTD and FantasyID download, data exploration | Both datasets readable; data findings logged | Done |
| 2 | Grouped splits (SIDTD by linked source, FantasyID by card), shared dataset classes, metrics, supervised baseline, JPEG shortcut removed | Reference baseline (E1, E2) | Done |
| 3 | PatchCore and PaDiM, one memory bank per document type | **Gate:** one-class model clearly above chance on pooled SIDTD; compared with baseline Crop & Replace AUROC (0.61-0.70) | In progress |
| 4 | Product v0: FastAPI `/predict` and web UI with the best available model | **Gate:** end-to-end demo works | Planned |
| 5 | Unseen-attack experiments (E1-E3) with bootstrap confidence intervals and several seeds | Comparison table | Planned |
| 6 | Forged-region boxes for SIDTD from annotations; localization metric for heatmaps | **Gate:** is the one-class model adequate? If not, focus on the hybrid model | Planned |
| 7 | Cross-dataset test (SIDTD to FantasyID), Turkish template 5-fold experiment, hybrid model | Generalization results | Planned |
| 8 | Improvements and ablations (resolution, backbone layers, coreset ratio); best model into the product | Final model | Planned |
| 9 | iOS app: camera, document detection, Core ML live score | Live check on the phone | Planned |
| 10 | iOS to server, heatmap view, Docker packaging, demo video | Working product | Planned |
| 11 | Error analysis, report writing | Report draft | Planned |
| 12 | Report, presentation, buffer | Final submission | Planned |

## Experiments

| ID | Model | Trained on | Tested on | Question |
| --- | --- | --- | --- | --- |
| E1 | Supervised (EfficientNet-B0) | Genuine + all forgeries | SIDTD test, per forgery type | Upper bound of the supervised model |
| E2 | Supervised | Genuine + inpainting only | Inpainting and unseen Crop & Replace | Does a supervised model generalize to an unseen forgery type? |
| E3 | One-class (PatchCore, PaDiM) | Genuine only | Both forgery types | Does a model that never saw forgeries do better on unseen types? |
| E4 | One-class | Genuine templates | SIDTD smartphone video frames | Does it hold under real capture conditions? |
| E5 | Best supervised and one-class models | SIDTD | FantasyID official test | Cross-dataset generalization |
| E6 | One-class | Turkish template genuine cards (5-fold) | Turkish template attacks | Results on a Turkish ID design |
| H | Hybrid (supervised + anomaly score) | SIDTD | All of the above | Best of both for the product |

All SIDTD experiments use the JPEG-normalized images (`data/sidtd_q75`). Metrics: AUROC (main), APCER, BPCER and ACER at a threshold set on validation genuine documents (BPCER 5%).

## Findings so far
- The official SIDTD split leaks source documents between train and test; splits were rebuilt by linked source document.
- Genuine and forged SIDTD images are saved with different JPEG settings (quality 92 vs 75). With the original files a supervised model reaches AUROC 1.00 on an unseen forgery type; after re-encoding all images the same way it drops to 0.61.
- The official FantasyID split leaks at card level; FantasyID experiments use card-grouped folds.

Details: `docs/decisions.md` (decisions) and `docs/results.md` (numbers).

## Risks and fallbacks
| Risk | Fallback |
| --- | --- |
| One-class model does not beat the supervised baseline | Report it as a generalization analysis; use the hybrid model in the product |
| Few Crop & Replace test forgeries (30) | Pool across document types; report confidence intervals |
| Training too slow on the laptop | Smaller models or Google Colab |
| iOS work does not fit in time | Web demo as the product; iOS as an extension |
