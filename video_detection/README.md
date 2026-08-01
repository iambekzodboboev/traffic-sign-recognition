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
- `process_video.py` — samples a video at ~2 fps, runs detect+classify on
  each sampled frame, and writes an annotated output video.
- `test_videos/` — real driving clips used for development/testing
  (gitignored — large binary files, not source, same treatment as the
  dataset zip and model weights).
- `output/` — annotated output videos produced by the pipeline (gitignored).

## Status

Steps 9.1-9.4 done (detector, classifier wiring, full video loop with
frame sampling). Verified on a real ~3-minute driving-test video: 345
sampled frames, 186 confident + 140 low-confidence ("unsure",
auto-flagged rather than shown as a guess) detections. Known findings:

- Correctly detects and classifies signs the model has never been shown
  in this exact framing before (e.g. a "Steep ascent 12%" warning sign).
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

Next: finish step 9.4 (tracking across frames, so the same physical sign
isn't reported repeatedly and the clearest frame gets picked), then 9.5
(a fuller honest test pass) and 9.6 (package a demo).
