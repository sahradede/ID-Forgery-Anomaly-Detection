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
