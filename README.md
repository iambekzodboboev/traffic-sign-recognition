# Traffic Sign Recognition — Signal

An AI/ML Fundamentals capstone project: a traffic sign classifier trained
from scratch to **99.37% test accuracy** across 200 sign classes, served
three ways — a Telegram bot for single photos, a video pipeline that finds
and tracks signs in real driving footage, and **Signal**, a public web app
that ties it all together.

**Live demo**: https://traffic-sign-recognition-95orsuspxcqeuq6yqz8ahk.streamlit.app/

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

## Dataset

[Traffic Signs in Post-Soviet States (200 classes)](https://www.kaggle.com/datasets/mikhailkosov/traffic-signs-in-post-soviet-states-200-classes)
via Kaggle — 116,642 images, 200 classes. Not included in this repo
(14.9GB); see `ROADMAP.md` section 0 for how each training notebook
fetches it.

## Status and full history

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — current stage, defense
  summary, headline numbers, honest limitations.
- [`ROADMAP.md`](ROADMAP.md) — the complete step-by-step build log:
  every stage, what was measured, what it showed, what decision followed.
