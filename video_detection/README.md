# Video Detection (Stage 9 — proposed future phase)

Extends the finished classifier to real driving video (dashcam / "blogger
driver" style footage), where a sign is small within a busy scene rather
than already cropped to fill the frame.

**Isolation from the Telegram bot**: this folder is a separate component.
It reads the existing trained classifier (`../models/resnet18_transfer_15ep.pt`,
`../metadata/class_names.csv`) and reuses `../scripts/predict_sign.py`'s
functions read-only — it does not modify `../bot.py` or anything the bot
depends on. The bot keeps working exactly as it does today, unaffected by
anything built here.

See `ROADMAP.md` section 9 in the repo root for the full staged plan and
reasoning.

## Contents

- `detector.py` — classical CV (color + shape) sign localization. No
  external model download; see the module docstring for why.
- `classifier.py` — wires a detected crop into the existing trained
  classifier (reused read-only from `scripts/predict_sign.py`).
- `process_video.py` — samples a video at ~4 fps, runs detect+classify on
  each sampled frame, feeds detections through the tracker, and writes an
  annotated output video plus a deduplicated text report of tracked signs.
- `tracker.py` — simple IoU-based tracking across sampled frames, with a
  center-distance fallback for fast-moving signs (see below), so the same
  physical sign is recognized as "still the same sign" rather than
  reported fresh every sampled frame. A track is only ever reported if a
  *majority* of its confident observations agree on the same class --
  not just one lucky frame.
- `test_videos/` — real driving clips used for development/testing
  (gitignored — large binary files, not source, same treatment as the
  dataset zip and model weights).
- `output/` — annotated output videos + tracked-sign `.txt` reports
  (gitignored).

## Status

Steps 9.1-9.4 done (detector, classifier wiring, full video loop with
frame sampling and tracking). Verified on a real ~3-minute driving-test
video: 738 sampled frames at 4 fps, 76 distinct tracked signs reported
after deduplication. Known findings, all confirmed by direct inspection,
not assumption:

- Correctly detects and classifies signs never specifically hand-tested
  before (e.g. "Steep ascent 12%", "Curve right" held correctly for 85
  consecutive sampled frames as the car approached it).
- **Tracking continuity needed two fixes, found by testing, not
  guessed up front**: (1) a sign moving fast across the frame between
  samples can have *zero* box overlap between consecutive samples --
  no amount of IoU-threshold tuning fixes a true zero-overlap gap, so a
  center-distance fallback match was added for when there's no overlap
  at all; (2) sampling at only 2 fps left gaps large enough to trigger
  this regularly -- raised to 4 fps, which this pipeline's speed
  (~0.05s/frame) comfortably affords. Together these turned what were 5
  fragmented "Stop" tracks and a 3-way-split fast-moving sign into 2 and
  1 clean continuous tracks respectively.
- A permanent channel watermark baked into this video sometimes sits
  directly on top of a sign, causing confident misclassification no
  box-boundary fix can remove (the contamination is *inside* the sign's
  own pixel region, not beside it) — specific to "blogger"/social-media
  style footage, not plain dashcam video.
- Two signs close together can merge into a single detection box; the
  resulting crop confuses the classifier, but this reliably shows up as
  low confidence rather than a wrong confident answer.
- Feeding a non-sign red/round object (a traffic light lens) to the
  classifier can produce a moderately confident wrong answer (~65-77%)
  — confidence thresholding alone doesn't catch every such case.
- **A new, subtler finding**: many short tracks all report "Lane
  directions" (a common blue guidance sign). Spot-checked directly rather
  than assumed to be noise -- these are real small signs (visually
  confirmed one mounted on a post next to a painted lane arrow on the
  pavement), not false detections. But the crops are tiny and blurry, and
  a "confident" score on a crop that small carries less real information
  than the same confidence on a large, clean crop -- unlike the bot's
  photo tests, where confidence tracked correctness closely, confidence
  alone is a weaker trust signal once the source crop itself is this
  degraded.

Next: 9.5 (a fuller honest test pass, e.g. against a second video) and
9.6 (package a clean, runnable demo).
