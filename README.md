# Traffic Sign Recognition — Signal

An AI/ML Fundamentals capstone project: a traffic sign classifier trained
from scratch to **99.37% test accuracy** across 200 sign classes, served
three ways — a Telegram bot for single photos, a video pipeline that finds
and tracks signs in real driving footage, and **Signal**, a public web app
that ties it all together.

**Live demo**: https://traffic-sign-recognition-95orsuspxcqeuq6yqz8ahk.streamlit.app/

**Student**: Bekzod Boboev
**Project track**: Track 1 — Individual Project Track (original idea, own approved Project Brief)
**ML task type**: multi-class image classification (200 classes) + object localization/tracking for the video extension

## What it does

Upload a driving video (or paste a public link) into Signal, and it plays
the video back while scanning: every sign it spots gets boxed, classified,
and logged live. When the scan finishes you get a trip report — stat
cards, category breakdown, a full list of signs with timestamps, CSV
export — and clicking any sign in the list jumps the video to that exact
moment and marks it.

## Results

| Metric | Baseline (from-scratch CNN) | Final model (ResNet18, fine-tuned) |
|---|---|---|
| Test accuracy | 87.84% (val) | **99.37%** |
| Test macro-F1 | — | 0.9937 |
| Worst-of-200-classes accuracy | 17.9% | 88.46% |

Random-guess baseline across 200 classes is 0.5%. The jump from baseline
to final model isn't a generic accuracy bump — confusion-matrix analysis
on the baseline identified specific, visually-similar sign pairs it kept
confusing (e.g. "Method of parking" vs. "Dangerous roadside"), and
swapping in a pretrained backbone eliminated exactly those confusions.
Full methodology and honest limitations are in
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Models and approaches tested

Two meaningfully different approaches were trained and compared under a
controlled, single-variable change (same 64x64 pipeline, split, batch
size, and epoch count for both):

- **Baseline — 3-conv-block CNN, trained from scratch.** 87.84%
  validation accuracy. Confusion-matrix analysis found the real
  weakness wasn't class imbalance (correlation with training-set size
  only -0.124) but specific, visually-similar sign pairs being confused
  with each other.
- **Final — ResNet18, ImageNet-pretrained, fully fine-tuned.** 99.03%
  validation / 99.37% test accuracy. Won on every comparison axis
  checked (macro-F1, worst-class accuracy, training-size correlation),
  and eliminated the exact confusions the baseline's analysis predicted
  — targeted evidence, not a generic accuracy bump.

Full comparison numbers are in `PROJECT_STATUS.md`'s "Stage 6 model
selection results".

## How it's built, stage by stage

1. **Data audit** — 116,642 images, 200 classes, real class imbalance
   (248–1,845 images/class), and a data-leakage risk from near-duplicate
   video-burst frames in the source dataset.
2. **Preprocessing** — perceptual-hash grouping of near-duplicates, then
   split by group (not by individual image) so near-identical frames
   never leak across train/val/test.
3. **Baseline model** — a 3-conv-block CNN trained from scratch: 87.84%
   val accuracy, with confusion analysis pinpointing the real weakness.
4. **Model selection** — pretrained ResNet18, fine-tuned on the same
   pipeline: 99.37% held-out test accuracy.
5. **Telegram bot** (`bot.py`) — CPU-only inference on single sign
   photos, with a 60%-confidence floor that asks for a clearer photo
   instead of guessing.
6. **Video detection** (`video_detection/`) — extends the classifier to
   real driving footage: classical CV localization (color + shape;
   out-performed a pretrained YOLOv8 baseline in direct testing) feeding
   the same classifier, plus cross-frame tracking so one physical sign
   isn't reported repeatedly. Tested against two real videos with found
   failure modes documented, not hidden.
7. **Signal** (`streamlit_app/`) — the public web app packaging the whole
   video pipeline behind a live-scanning, click-to-seek interface.

## Repo layout

```
notebooks/          Training notebooks (data audit -> preprocessing -> baseline -> ResNet18)
scripts/             Reusable pipeline code: split manifest, inference (predict_sign.py), etc.
models/              Trained weights (resnet18_transfer_15ep.pt, ~45MB, committed for deployment)
metadata/            class_names.csv, split_manifest.csv (fixed, deterministic artifacts)
assets/class_icons/  One clean reference icon per class
bot.py               Telegram bot — single-photo classification
video_detection/     Detector + tracker + CLI for running the pipeline on a video file
streamlit_app/       Signal — the public web demo (Streamlit)
docs/                Course deliverables (showcase page, portfolio page, submission form)
PROJECT_STATUS.md    Current stage, defense summary, full results
ROADMAP.md           Full step-by-step build log with reasoning for every decision
```

## Running it yourself

```bash
pip install -r requirements.txt
```

**Telegram bot** — needs a `TELEGRAM_BOT_TOKEN` in a local `.env` file:

```bash
python bot.py
```

**Video detection CLI** — runs the pipeline on any video file and writes
an annotated video + text report to `video_detection/output/`:

```bash
python video_detection/process_video.py path/to/your_video.mp4
```

See [`video_detection/README.md`](video_detection/README.md) for options
(e.g. `--seconds` to test on just a clip).

**Signal web app** — local copy of the live demo:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

See [`streamlit_app/README.md`](streamlit_app/README.md) for deployment
notes.

## Training or fine-tuning the model

The committed model (`models/resnet18_transfer_15ep.pt`) is ready to use
as-is — you don't need to retrain it to run any of the above. To
reproduce or rebuild it:

1. **`notebooks/00_colab_github_connection_check.ipynb`** through
   **`03_preprocessing.ipynb`** — dataset audit and the leakage-safe
   preprocessing pipeline. Run in Google Colab (open directly from
   GitHub via File → Open notebook → GitHub tab); each notebook
   re-downloads the dataset from its own cell, no local setup needed.
2. **`04_baseline_model.ipynb`** — trains the from-scratch baseline CNN
   (also in Colab).
3. **`05_transfer_learning.ipynb`** — trains the final ResNet18 model.
   This one runs on **Kaggle Notebooks**, not Colab (switched after
   Colab's free-tier GPU quota was exhausted by earlier attempts;
   Kaggle also natively hosts this dataset, so no download step is
   needed there). Add the dataset via **+ Add Input**, turn on
   **Settings → Internet**, and select a GPU accelerator before running.

All training runs log to MLflow (Drive-backed for the Colab notebooks),
so a finished run is never silently retrained — see `ROADMAP.md` section
6 for the exact reasoning and results at each step.

## Example input and output

**Photo classification** (`scripts/predict_sign.py`, or the Telegram
bot): given a photo of a single sign — e.g. class 44, "Minor road,
right" — the output is a predicted sign name, category, and confidence,
such as:

```
Sign: Minor road, right
Category: Priority
Confidence: 99.9%
```

**Video scanning** (`video_detection/process_video.py`, or Signal):
given a driving video, the output is a deduplicated, timestamped list of
every sign found — e.g. a real 172-second test video produced 76
tracked signs (STOP, pedestrian crossing, warning triangles, mandatory
circles, lane-direction signs) from 738 sampled frames in 39 seconds.

## Dataset

[Traffic Signs in Post-Soviet States (200 classes)](https://www.kaggle.com/datasets/mikhailkosov/traffic-signs-in-post-soviet-states-200-classes)
via Kaggle — 116,642 images, 200 classes. Not included in this repo
(14.9GB); see `ROADMAP.md` section 0 for how each training notebook
fetches it.

## Known limitations

- **Near-duplicate deduplication isn't perfect.** The train/val/test
  split groups near-identical images (perceptual hashing) before
  splitting, to avoid leaking the same physical sign across splits —
  but the grouping is fairly strict and may miss some more
  visually-different frames of the same sign. A stated, known gap, not
  a hidden one.
- **The video pipeline has real, found failure modes**: a watermark or
  on-screen caption sitting directly on top of a sign can cause
  confident misclassification; two signs stacked on one post can merge
  into a single box and be confidently mislabeled as something
  unrelated; a box that captures only a sign's icon (not its full
  multi-plate context) can confuse near-visual-twin classes. All
  documented in detail in `video_detection/README.md`, found by testing
  against two different real videos rather than assumed.
- **Confidence is a weaker trust signal on small, blurry video crops**
  than on the large, clean photos the classifier was validated against.
- **The dataset covers only post-Soviet-states sign design standards.**
  The model has no basis for other countries' sign conventions and
  would need retraining (not just fine-tuning) to generalize elsewhere.

## Responsible AI considerations

- **Bias and fairness**: class imbalance (248-1,845 images/class) was
  measured and compensated for with a `WeightedRandomSampler`; the
  measured correlation between per-class training size and accuracy was
  near-zero in both models (-0.124 baseline, 0.0381 final), so imbalance
  was checked as a possible source of unfairness across classes and
  found not to be one. Not checked: bias across image *capture
  conditions* (lighting, weather, source camera) — the dataset is known
  to be merged from multiple original sources, so this dimension likely
  exists but hasn't been specifically audited.
- **Privacy and safety**: the Telegram bot writes each uploaded photo to
  a temp file, uses it once, and deletes it in a `finally` block
  regardless of outcome — nothing is retained. The Signal web app clears
  its per-session temp video on reset. Open gap: unlike a cropped sign
  photo, driving video can incidentally contain other people, faces, or
  license plates in frame, and the pipeline currently does nothing to
  detect or blur incidental PII in uploaded footage.
- **Limitations and proper use**: this system is a decision-support /
  informational tool for driving students, instructors, and anyone
  reviewing driving footage. It is **not** a certified safety system and
  must not be relied on as the sole input for autonomous driving or
  real-time safety-critical decisions.

## Status and full history

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — current stage, defense
  summary, headline numbers, honest limitations.
- [`ROADMAP.md`](ROADMAP.md) — the complete step-by-step build log:
  every stage, what was measured, what it showed, what decision followed.
