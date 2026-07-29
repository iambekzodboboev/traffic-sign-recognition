# Project Status

## Project

Traffic Sign Recognition

## Current stage

Stage 3 — Data audit and EDA (see `ROADMAP.md` for full step-by-step plan)

## Completed

- Local Git repository initialized
- GitHub repository created and connected (`traffic-sign-recognition`, public)
- Baseline `.gitignore`, `README.md`, `AGENTS.md` in place
- Python 3.12 environment set up in `.venv`, with PyTorch (CPU), torchvision,
  pandas, matplotlib, scikit-learn, OpenCV, Jupyter/JupyterLab, MLflow, and
  the Kaggle CLI installed; pinned in `requirements.txt`
- Dataset (~14.9GB, 200 classes, 116,642 images) obtained from Kaggle
  ([mikhailkosov/traffic-signs-in-post-soviet-states-200-classes](https://www.kaggle.com/datasets/mikhailkosov/traffic-signs-in-post-soviet-states-200-classes)),
  uploaded to the user's Google Drive
- Stage 0 (workflow setup) complete: Colab ↔ GitHub round trip verified
  (open from GitHub, save back to GitHub), GPU (T4) confirmed working,
  dataset download in Colab via authenticated Drive API (`notebooks/01_dataset_download_colab.ipynb`)
- Stage 3.1 done: full integrity check in Colab, all 116,642 images verified
  openable, 0 corrupt, all 200 class folders present
- Stage 3.2 done: class distribution confirmed (248-1,845 images/class,
  mean 583) — real but moderate imbalance, no classes with only a
  handful of images
- Work for stage 3 is in `notebooks/02_data_audit_eda.ipynb`

## Current task

Stage 3.3 — image property audit (resolution range, aspect ratios, color
mode, file size range)

## Next

- 3.3 Image property audit
- 3.4 Visual sample audit
- 3.5 Class ID -> human-readable name mapping
- 3.6 Investigate track/frame-numbered filenames for leakage risk
- 3.7 Write up data audit findings

## Known problems / blockers

- None
