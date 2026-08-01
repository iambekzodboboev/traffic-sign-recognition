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

- `test_videos/` — real driving clips used for development/testing
  (gitignored — large binary files, not source, same treatment as the
  dataset zip and model weights).
- `output/` — annotated output videos produced by the pipeline (gitignored).

## Status

Not started yet — waiting on a test video before step 1 (choosing and
trying a pretrained sign detector) begins.
