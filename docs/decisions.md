# Decision log

## 2026-10-28: Decisions after data exploration
- The official SIDTD split will not be used:
  - train and test share 252 source documents (leakage),
  - most CSV rows point to files missing from templates.zip (1,700 of 2,511 train rows).
  The split will be rebuilt grouped by source document.
- SIDTD forgery types: Inpaint_and_Rewrite 1,077, Crop_and_Replace 145. Unseen-attack experiment: the supervised model is trained on inpainting and tested on Crop & Replace.
- Crop & Replace results will be pooled across templates because each template has few samples.
- SIDTD forged-region boxes will be built by combining field boxes from the genuine-document JSON with the field name in the forgery JSON.
- FantasyID provides altered-field boxes (region_provenance = altered). It will be used for the cross-dataset test and the Turkish-template experiment; the Turkish split will be grouped by card.

## 2026-10-28: Source-grouped split
- SIDTD split rebuilt with `src/data/splits.py` (seed 42): 70/10/20 source documents per document type, stratified by document type. A source document and all forgeries derived from it are always in the same split.
- 76 Crop & Replace forgeries take a field from a second source document that lies in another split. They were kept, with a planned with/without sensitivity check.

## 2026-10-28: Linked-source grouping (supersedes the previous entry)
- 30 of the 32 Crop & Replace forgeries in the test split had their second source in another split, so a with/without sensitivity check would leave almost no test samples.
- The split now groups documents linked by a forgery (source and second source, union-find) and assigns whole groups to splits. No forgery has a source in another split; `check_split` enforces this. Result: 701/101/198 genuine documents; 107/8/30 Crop & Replace forgeries in train/val/test.
- `.gitignore` now ignores only the root `/data/` and `/results/` folders; the previous `data/` pattern also ignored `src/data/`.

## 2026-10-28: FantasyID card-grouped folds
- The file stem identifies a physical card: the same stem is used for the three capture devices and for every manipulation of that card.
- The official FantasyID split leaks at card level: e.g. turkiye-008_03 is a genuine training card and also a digital_3 test attack.
- `src/data/splits_fantasyid.py` assigns cards with genuine images to 5 folds per template (seed 42). Cards that only appear as attacks get fold -1 and are added to every test fold.
- The Turkish-template experiment uses 5-fold cross-validation (25 genuine cards) and reports mean and standard deviation.

## 2026-10-28: JPEG-quality shortcut in SIDTD
- Genuine SIDTD templates are stored at JPEG quality 92 (4:4:4 chroma), forged ones at quality 75 (4:2:0, the Pillow default). Image sizes are identical.
- A supervised EfficientNet-B0 reached test AUROC 1.000 on unseen Crop & Replace forgeries when trained on inpainting only, which this compression difference can explain.
- `src/data/normalize_jpeg.py` re-encodes every template at quality 75, 4:2:0 into `data/sidtd_q75`. Forged images barely change; genuine images get the same compression. All SIDTD experiments use this copy by default (`jpeg_q75=True`); `--raw-jpeg` reproduces the original setting.
- Without the shortcut, test AUROC on unseen Crop & Replace drops to 0.677 after 10 epochs.

## 2026-10-29: Baseline training length
- After 10 epochs the supervised baseline had not converged (validation AUROC still rising). The reference baseline is trained for 30 epochs, keeping the checkpoint with the best validation AUROC.
- `--resume` continues an interrupted run from `last.pt` (model, optimizer, epoch, best validation AUROC).

## 2026-10-29: PatchCore implementation
- PatchCore is implemented in `src/models/patchcore.py` following Roth et al. (2022): WideResNet-50 layer2 + layer3 features with 3x3 averaging, greedy coreset (10%), nearest-neighbour distance, image score = maximum patch score. These match the anomalib 2.7 defaults.
- The anomalib Folder/Engine pipeline is not used: it reads folders while our splits are CSV files, and its default validation split is taken from the test set (`val_split_mode=FROM_TEST`). The in-house version uses the same splits, transforms and metrics as the supervised baseline.
- One memory bank per document type. Scores are z-normalized with the validation genuine documents of the same type and pooled across types, because each type has only about three Crop & Replace test forgeries.

## 2026-10-30: Field-aware PatchCore
- Plain PatchCore on esp_id: test AUROC 0.54. Diagnostics: the anomaly-map peak lies inside the forged field for only 7% of forgeries; peaks sit on the signature and the face, which vary between genuine documents.
- Diagnostic only (normalized with test genuine documents, forged field known): comparing each forged field with the same field of genuine documents gives AUROC 0.97, against 0.56 for the plain image score and 0.64 with face, photo and signature masked.
- New model `src/models/field_patchcore.py`: one memory bank of all genuine training patches per document type, fixed field boxes (union of training annotations), per-field z-normalization with leave-one-document-out training statistics, document score = maximum field z-score. The arg-max field is the predicted forged field (field hit rate).
