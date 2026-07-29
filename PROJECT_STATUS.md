# Project Status

## Project

Traffic Sign Recognition

## Current stage

Stage 4 — Preprocessing and leakage prevention (see `ROADMAP.md` for full
step-by-step plan)

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
- Stage 3.3 done: image sizes range 25-2,749px wide, 25-1,496px tall
  (median only 191x157 — long tail of larger images); aspect ratio mostly
  near 1.0 with a real tail to 5.88 (wide/thin plates); color mode is
  majority RGBA (96,444) vs RGB (20,198), no grayscale
- Stage 3.4 done: visual spot-check of 8 random classes x 4 images each,
  all internally consistent, no labeling issues found
- Stage 3.5 done: class ID -> name mapping built from the dataset's own
  `Classes/` folder structure, committed at `metadata/class_names.csv`;
  one data quirk flagged (class 81 has two conflicting names in the
  source taxonomy)
- Stage 3.6 done: confirmed near-duplicate/leakage risk is real for the
  ~10.3% of files with cleanly parseable track filenames; the rest use
  other naming conventions (dataset looks merged from multiple sources)
  and need content-based deduplication instead of filename parsing
- Stage 3.7 done: findings written up below
- Work for stage 3 is in `notebooks/02_data_audit_eda.ipynb`
- Stage 4.1 done: train/val/test split strategy decided and implemented —
  64-bit average-hash near-duplicate grouping within each class (Hamming
  distance <= 6), computed once deterministically (not re-randomized each
  session) via `scripts/build_split_manifest.py`
- Stage 4.2 done: split saved as `metadata/split_manifest.csv` (116,642
  rows), 70/15/15 by group (31,820 / 6,822 / 6,822), ~70.9/15.5/13.5% by
  image; validated in Colab (all paths exist, row count matches, no group
  spans multiple splits)
- Stage 4.3 done: preprocessing pipeline decided — 64x64 target, letterbox
  resize (pad to square, not a plain stretch, given the aspect-ratio tail
  found in 3.3), RGBA -> RGB, ImageNet normalization, light augmentation
  (rotation + color jitter) on training data only
- Stage 4.4 done: PyTorch `TrafficSignDataset` + DataLoaders built from the
  manifest, with a `WeightedRandomSampler` on training to oversample
  minority classes (addresses 3.2's imbalance); sanity-check batch
  visualization added, pending the user's visual confirmation in Colab
- Work for stage 4 is in `notebooks/03_preprocessing.ipynb`

## Stage 3 data audit summary

**Class balance**: 200 classes, 116,642 images, 248-1,845 images/class
(mean 583). Real but moderate imbalance — no classes with only a handful
of images, so class weighting plus modest augmentation for smaller
classes should be enough; no drastic measures needed.

**Image properties**: sizes range from 25px to 2,749px wide (median only
191x157 — a long tail of larger images pulls the mean up), so a modest
target resize (e.g. 64x64 or 128x128) makes sense rather than trying to
preserve the outliers' resolution. Aspect ratio is mostly near-square
(median 1.065) with a real tail up to 5.88 for wide/thin signs (e.g. text
plates) — resizing to a square target won't distort most images much.
Color mode is majority RGBA (96,444) vs RGB (20,198), no grayscale — the
alpha channel needs flattening to RGB before training.

**Visual/label quality**: spot-checked 8 random classes x 4 images each,
all internally consistent, no mislabeled folders found.

**Class names**: mapped from the dataset's own `Classes/` folder, saved
at `metadata/class_names.csv`. One data quirk: class 81 has two
conflicting names in the source taxonomy ("Height limit - 3.5" vs "-
4.5") — flagged, doesn't affect training (numeric ID is what's used),
only the human-readable label.

**Data leakage risk (the important one)**: filenames suggest multiple
photos of the same physical sign exist as near-duplicate frames (e.g.
`00000_00000_00017.png` = class/track/frame). Confirmed real for the
~10.3% of files that cleanly match this pattern — ~900 (class, track)
groups, averaging ~13 images each, ~82% with 5+ images, visually
confirmed as the same sign photographed moments apart. However, filename
parsing only reliably covers about 10% of the dataset; the rest use other
naming conventions, consistent with this dataset being merged from
multiple original sources. **Decision for stage 4.1**: build near-
duplicate groups using content-based detection (e.g. perceptual image
hashing) across the whole dataset, not filename parsing alone, then split
train/val/test by group — never by individual image — to avoid leaking
near-identical frames across the split.

## Stage 4 known limitation (stated honestly)

The near-duplicate grouping is fairly strict (~2.6 images/group on
average, vs. ~13/group for the filename-confirmed tracks in 3.6), so it
reliably catches exact/near-exact duplicates but may miss some more
visually-different frames of the same physical sign (e.g. a longer zoom
sequence). It meaningfully reduces leakage risk versus naive per-image
splitting, but is not a perfect guarantee — worth mentioning as a known
limitation if asked during defense.

## Current task

Waiting for the user to visually confirm the stage 4.4 sanity-check batch
in Colab, then move to stage 5 (baseline model and experiments)

## Next

- Confirm stage 4 sanity check
- 5.1 Set up MLflow tracking
- 5.2 Baseline model, prove the training loop works end-to-end
- 5.3 Train baseline on full dataset

## Known problems / blockers

- None
