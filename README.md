# ID Forgery Anomaly Detection

Forgery detection on smartphone-captured identity documents using anomaly detection models trained only on genuine documents.

## Datasets
- SIDTD: https://tc11.cvc.uab.es/datasets/SIDTD_1
- FantasyID: https://zenodo.org/records/17063366

See `docs/data.md` for download details.

## Environment
- Python 3.11 (uv), PyTorch, anomalib
- Developed on Apple Silicon (MPS)

## Setup
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```
