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
- 76 Crop & Replace forgeries take a field from a second source document that lies in another split. They are kept: this cannot inflate results (the pasted content comes from a genuine document and makes the forgery look more normal), and removing them would halve the Crop & Replace test set. Crop & Replace metrics will be reported with and without them as a sensitivity check.

## 2026-10-28: Linked-source grouping (supersedes the previous entry)
- 30 of the 32 Crop & Replace forgeries in the test split had their second source in another split, so a with/without sensitivity check would leave almost no test samples.
- The split now groups documents linked by a forgery (source and second source, union-find) and assigns whole groups to splits. No forgery has a source in another split; `check_split` enforces this.
- `.gitignore` now ignores only the root `/data/` and `/results/` folders; the previous `data/` pattern also ignored `src/data/`.
