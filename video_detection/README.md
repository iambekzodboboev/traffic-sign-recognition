# Video Detection (Stage 9)

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

## Quickstart

Requirements: the project's usual environment (`pip install -r
requirements.txt` from the repo root) plus the trained model at
`models/resnet18_transfer_15ep.pt` (see the repo root `PROJECT_STATUS.md`
if that file is missing — it's gitignored, not lost).

Run on any driving video, from the repo root:

    python video_detection/process_video.py path/to/your_video.mp4

To try it quickly on just the first N seconds instead of the whole video:

    python video_detection/process_video.py path/to/your_video.mp4 --seconds 20

Add `--help` to see both options again at any time. Output goes to
`video_detection/output/`:

- `<name>_annotated.mp4` — the input video with detection boxes and
  labels drawn on each sampled frame (green = confident; orange = below
  the 60% confidence threshold, shown but flagged "unsure" rather than
  hidden).
- `<name>_annotated.txt` — a deduplicated list of tracked signs (one
  line per physical sign seen across the video, not per frame).

This repo doesn't ship a sample video (large binary files are gitignored
here, same treatment as the dataset zip and model weights) — supply your
own driving/dashcam video to try it. See "Status" below for what to
expect and known limitations, so results can be read with the right
expectations.

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

**Stage 9 complete (9.1-9.6).** Detector, classifier wiring, full video
loop with frame sampling and tracking, tested against two different real
videos, and packaged as the runnable demo described in "Quickstart"
above (a clearer `--seconds` flag, `--help`, and a friendly error for a
missing file, instead of the original raw positional-argument CLI).

Deliberately **not** built, as an honest scope call rather than an
oversight: a live-camera-feed variant. The roadmap flagged this as
optional future work, not a course-project requirement. Not attempted
or tested here, but the core `process_video()` function underneath is
already agnostic to the source (it just calls `cv2.VideoCapture(...)`,
which OpenCV can also point at a live camera index) — extending to live
video later would mean loosening `main()`'s current file-must-exist CLI
check for a camera-index case and testing that path for real, not
redesigning the detect/classify/track pipeline itself.

**Clip 1** — a real ~3-minute driving-test video: 738 sampled frames at
4 fps, 76 distinct tracked signs reported after deduplication. Known
findings, all confirmed by direct inspection, not assumption:

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

**Clip 2** — a public Instagram reel from a driving-school account
(`driving_clip_02.mp4`, portrait 720x1280, 25fps, 123.5s), a different
video with different lighting, route, and camera mount, filmed at a
different closed driving-test facility than clip 1 (confirmed by the
video's own on-screen station captions). 515 sampled frames at 4 fps,
116 tracked signs (higher density than clip 1, consistent with a
facility that has many numbered practice stations each carrying real
signage, not noise). Findings, each confirmed by pulling the actual
crop/frame:

- STOP signs and plain "P" parking signs replicated clip 1's good
  behavior: tracked correctly and consistently across the approach,
  confidence tracking correctness closely on large, clean crops.
- **New failure mode: a confidently-wrong merge, not a low-confidence
  one.** This facility mounts two different sign faces stacked on one
  post (e.g. pedestrian-crossing icon above a left-turn arrow icon).
  The detector boxes both as one region, and the classifier confidently
  (97-98%) mislabels the combined shape as an unrelated class ("Car
  washing"), reproducibly at multiple stations. This means clip 1's
  "merged boxes show up as low confidence" finding is **not a general
  guarantee** — it held for that specific merge shape, not this one.
- **New, subtler failure mode:** a box captured only the icon of a
  multi-element parking sign, excluding the supplementary plate that
  the class taxonomy actually uses to tell "Parking (parking space)"
  apart from "Regulated parking zone" (both share the same blue-square
  "P" icon alone). Classified confidently (100%) as one of the two —
  the icon-only crop genuinely doesn't contain enough information to
  verify which is correct, which is itself the finding.
- Clip 1's watermark-on-sign occlusion did **not** recur — this video's
  captions sit low in the frame (near the road), never overlapping the
  elevated sign poles, confirming that finding was specific to clip 1's
  channel watermark placement, not an inherent pipeline flaw.
- A small, distant red sliver (likely a marker, not a sign) was picked
  up at a moderate 65% confidence — same non-sign-object pattern as
  clip 1.

Full detail (including the "how it was downloaded" note) in
`ROADMAP.md` section 9.5 and `PROJECT_STATUS.md`.

Next: 9.6 (package a clean, runnable demo).
