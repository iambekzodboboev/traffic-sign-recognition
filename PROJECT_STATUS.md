# Project Status

## Project

Traffic Sign Recognition

## Current stage

Stage 8 — Reproducibility, documentation, defense prep (stages 1-7
complete; see `ROADMAP.md` for full step-by-step plan)

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
  verified in Colab — batch shape `[64, 3, 64, 64]` correct, letterbox
  padding visible, augmentation visible, class names matched their images
- Stage 4 fully complete and verified
- Work for stage 4 is in `notebooks/03_preprocessing.ipynb`
- Stage 5.1-5.4 done and verified in Colab: MLflow tracking (persisted to
  Drive), baseline 3-conv-block CNN sanity-checked on a 3-class subset,
  full 15-epoch training run (final val_acc=0.878), and full evaluation
  (confusion matrix, per-class accuracy, correlation, confused pairs)
- Trained model confirmed saved permanently in MLflow (Drive-backed),
  independent of notebook/session state
- Work for stage 5 is in `notebooks/04_baseline_model.ipynb`
- Stage 6.0 done: cheap visual spot-check (no Colab/GPU) on class 184
  ("Coverage area") against its two biggest baseline confusions, 183
  ("Distance to the object") and 193 ("Limitation of parking duration") —
  sampled directly from the local dataset zip via
  `scripts/spot_check_184.py`. **Not a labeling bug**: all three are
  genuinely distinct, correctly-labeled sign types. 184 and 183 are
  near-visual-twins (same white plate/font/"distance in meters" text,
  differing mainly by an arrow); 184 and 193 differ mainly by "M" vs.
  "мин" text. Confirms stage 5's "genuine visual similarity, not
  mislabeling" hypothesis, and suggests the 64x64 target resolution may
  be part of the problem (small text/arrows may not survive that
  downsampling).
- Stage 6.1 done: ResNet18 (ImageNet-pretrained, fully fine-tuned)
  transfer-learning experiment, chosen over a from-scratch deeper CNN
  because stage 5.4's finding was about confusability between specific
  sign pairs, which pretrained features address directly. Same 64x64
  pipeline, split, batch size, and 15 epochs as the baseline (LR lowered
  to 1e-4) — only the model changed, for a clean comparison. Iterated
  twice before the real run (added mixed precision after an early attempt
  took 2-3 hours in Colab; added a "skip training if an MLflow run
  already finished" guard after reopening the notebook kept forcing full
  retrains), then moved to Kaggle Notebooks after Colab's free-tier GPU
  quota ran out. Run on Kaggle, trained and evaluated in
  `notebooks/05_transfer_learning.ipynb` (built from
  `scripts/build_notebook_05.py`, which matches exactly what was
  executed). **Result: validation accuracy 0.9903** (vs. baseline's
  0.8784) — every confusion flagged in stage 5 dropped to near-zero.
- Stage 6.2 done: compared the two models on validation metrics beyond
  raw accuracy (macro-F1, per-class accuracy std dev, worst-class
  accuracy, accuracy-vs-training-size correlation). **ResNet18 wins on
  every axis** — worst-class accuracy 0.8033 vs. baseline's 0.179;
  macro-F1 0.9877 nearly matches overall accuracy (consistent strength
  across all 200 classes); training-size correlation stayed near zero
  (0.0381). **ResNet18 selected as the final model.**
- Stage 6.3 done: final, one-time evaluation of the chosen model on the
  held-out test set (untouched until the model was already selected).
  **Final test accuracy: 0.9937** (99.37%), slightly above validation
  accuracy — no overfitting from the iterative comparison process. Test
  macro-F1 0.9937; worst test class 170 ("Phone") at 0.8846.
- **Stage 6 (model selection) fully complete.**
- Stage 7.1 done: trained weights downloaded from the live Kaggle session
  (`resnet18_transfer_15ep.pt`, ~45MB state dict, via
  `IPython.display.FileLink`) and placed at
  `models/resnet18_transfer_15ep.pt` locally. Gitignored like the dataset
  zip (build artifact, not source) — if ever missing, redownload from
  that Kaggle notebook's Output or MLflow run
  `21082892cf2d4295851e8c1a96580863`. Class-name mapping reused as-is
  from `metadata/class_names.csv` (stage 3.5), no re-export needed.
- Stage 7.2 done: `scripts/predict_sign.py`, a local CPU-only inference
  script (image in, predicted sign name + confidence out) — no GPU,
  dataset, or network access needed. Applies the exact same letterbox +
  ImageNet-normalize preprocessing as training/eval. **Verified working**
  on 6 sample photos pulled from the local dataset zip (classes 0, 14,
  44, 45, 184, 190, including the previously-confused 44/45 mirror pair
  and 184's old confusions) — all 6 predicted correctly at 99.99-100%
  confidence.
- Stage 7.3 done: decided on a **Telegram bot** for the demo — user sends
  a photo, bot replies with sign name, category (mandatory/warning/etc.
  — already in `metadata/class_names.csv`'s `category` column, no new
  work), confidence, and a clean reference icon. Extracted one canonical
  reference icon per class from the dataset zip's `Classes/` folder via
  `scripts/extract_reference_icons.py` into `assets/class_icons/` (all
  200 resolved unambiguously; 1.4MB total, committed directly).
- Stage 7.4 done: `bot.py`, built with `pyTelegramBotAPI`, reuses
  `scripts/predict_sign.py`'s model-loading/prediction logic directly —
  runs entirely on CPU. Bot token from @BotFather stored in a local
  gitignored `.env` (loaded via `python-dotenv`), never committed.
  **Verified working end-to-end**: tested live on Telegram with a real
  photo, got back the correct sign name, category, confidence, and
  reference icon.
- **Stage 7 (inference/demo workflow) fully complete.**
- Stage 8.2/8.3 (partial) done: `Traffic_Sign_Recognition_Project_Report.docx`
  — a comprehensive, stage-by-stage narrative report covering the whole
  project (stages 0-7), organized around what was measured at each step,
  what it showed, and what decision followed from it, readable without
  needing to look at the code. Includes three charts generated from the
  real data/results and the full results tables. `README.md` itself
  hasn't been updated yet, and 8.1 (notebook/script cleanup) hasn't been
  started.

## Stage 6 model selection results

**Chosen model**: ResNet18, ImageNet-pretrained, fully fine-tuned, same
64x64 letterbox input/split/batch size as the baseline, 15 epochs, mixed
precision. Trained and evaluated on Kaggle Notebooks (Colab's free-tier
GPU quota ran out from earlier attempts before mixed precision was
added). Code in `notebooks/05_transfer_learning.ipynb`.

**Validation accuracy: 0.9903** vs. baseline's 0.8784. **Final held-out
test accuracy: 0.9937** (99.37%) — slightly *above* validation, a good
sign of no overfitting across the iterative comparison process.

**The specific stage-5 confusions were essentially eliminated** (baseline
count -> this model's count, same validation set):
- `44 (Minor road, right)` <-> `45 (Minor road, left)`: 92x -> **0x**
- `184 (Coverage area)` `->183 (Distance to the object)`: 52x -> **1x**
- `184 (Coverage area)` `->193 (Limitation of parking duration)`: 46x -> **0x**
- `189 (Validity period)` `->192 (Paid services)`: 51x -> **2x**
- `190 (Method of parking)` `->196 (Dangerous roadside)` (the single
  biggest baseline confusion): 122x -> **0x**
- `42 (End of overtaking-by-lorries restriction)` `->32 (End of all
  restrictions)`: 85x -> **0x**

This directly confirms stage 5's hypothesis: the baseline's weakness was
insufficient fine-grained visual discrimination (not class imbalance),
and a pretrained backbone fixes it — even at the same 64x64 input size
that stage 6.0's spot-check had flagged as a possible limiting factor for
exactly these plate-style confusions.

**Per-class balance (stage 6.2 criteria, not just overall accuracy)**:
macro-averaged F1 0.9877 on validation (0.9937 on test) sits almost
exactly at overall accuracy, meaning strength is consistent across all
200 classes rather than concentrated in frequent ones. Worst-class
accuracy rose from baseline's 0.179 (class 184) to 0.8033 on validation
/ 0.8846 on test (a different class is now the worst in each split, as
expected once the previously-worst classes were fixed). Accuracy-vs-
training-size correlation stayed near zero (0.0381), confirming this
model is, if anything, even less sensitive to the stage 3.2 class
imbalance than the baseline was.

**Honest caveat for defense**: stage 4 documented that near-duplicate
deduplication isn't perfect (a known, stated limitation). 99%+ accuracy
is a high number, but the improvement is targeted evidence, not generic
inflation — it specifically resolved the exact confusions predicted by
the stage 5 analysis, which is a harder pattern to explain away as
leakage than a uniform accuracy bump would be.

## Stage 5 baseline results

**Overall**: final validation accuracy **87.84%** (200 classes, random
baseline 0.5%). Climbed steadily every single epoch (40.5% -> 87.8% over
15 epochs), no plateau, no overfitting signs (val accuracy stayed at or
above train accuracy throughout, expected given train-only augmentation
and dropout being active only during training).

**Surprising finding — imbalance is not the bottleneck**: correlation
between per-class validation accuracy and training-set size was only
**-0.124** (essentially none). This contradicts the expectation from
stage 3 that smaller classes would be the weakest. The `WeightedRandomSampler`
already in place is handling the imbalance adequately; it is not what's
limiting this baseline.

**What actually explains the weak classes — specific, systematic
confusions between visually/semantically similar sign pairs**, not
scattered random error:
- `44 (Minor road, right)` <-> `45 (Minor road, left)` — 92x. Mirror-image
  confusion.
- `184 (Coverage area)` is the single worst class (17.9% accuracy) — split
  almost entirely between two confusions: `->183 (Distance to the object)`
  (52x) and `->193 (Limitation of parking duration)` (46x), accounting for
  98 of its 134 validation images.
- `189 (Validity period)` `->192 (Paid services)` (51x) — another
  informational/text-plate sign confused with a similar one (recall `189`
  is literally a time-range text plate, per the 3.4 visual audit).
- `190 (Method of parking)` `->196 (Dangerous roadside)` — **122x, the
  single biggest confusion in the whole matrix.** Also explains an
  otherwise-puzzling bright spot near the diagonal in the confusion matrix
  heatmap: class 196 is only 6 columns from 190, so this large off-diagonal
  error visually blended with the diagonal at 200x200 resolution — not a
  data leak or artifact.
- `42 (End of overtaking-by-lorries restriction)` `->32 (End of all
  restrictions)` (85x) — semantically related "end of restriction" signs.

**Takeaway for stage 6**: prioritize things that improve the model's
ability to discriminate fine visual detail (more capacity, or a pretrained
backbone) over further imbalance-focused changes. Worth a quick visual
sanity-check on the `184` confusions specifically, to rule out an actual
labeling issue rather than genuine visual similarity, before assuming it's
purely a model-capacity problem.

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

Stage 8.1 — clean up the final notebooks/scripts and confirm
`requirements.txt` is accurate, so the whole pipeline can be rerun from
scratch (not started yet; discuss scope with the user first). Stage 8.2's
comprehensive narrative report is done
(`Traffic_Sign_Recognition_Project_Report.docx`); README.md itself is
still unchanged.

## Next

- 8.1 Clean up final notebooks/scripts, confirm `requirements.txt` is accurate
- 8.2 Update `README.md` with how to reproduce and the main results
- 8.3 Update `PROJECT_STATUS.md` to reflect completion, prepare defense summary

## Known problems / blockers

- Colab's free-tier GPU quota was exhausted during stage 6.1 (from
  earlier long pre-mixed-precision training attempts); worked around by
  moving that notebook to Kaggle rather than waiting it out. Quota status
  wasn't rechecked since, so assume it may still be limited if a future
  stage wants to use Colab again.
- **Real-world limitation found via live bot testing**: a wide street-scene
  photo (sign small within a lot of background — trees, road, crosswalk)
  was misclassified, but with notably low confidence (42.7%, vs. 99%+ on
  every other tested photo) rather than confidently wrong. Likely cause:
  the training dataset's photos are tighter, sign-focused shots, so a
  whole-scene photo shrinks the actual sign to a small, cluttered patch
  after the 64x64 resize — a framing/domain-gap issue, not a sign of poor
  model accuracy. This is direct, real-world evidence for why the future
  plan (Section 10 of the project report / the presentation's future-vision
  slide) includes a detection step to locate and crop the sign *before*
  classifying it, rather than classifying a whole photo directly.
