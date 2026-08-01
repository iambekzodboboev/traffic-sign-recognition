# Signal — road sign scanner (public web demo)

A public-facing web app for the video sign-detection pipeline built in
`video_detection/` (stage 9). Upload a driving video or paste a link,
watch signs get logged live as it scans, then get a full trip report
when it's done.

**Isolation, same rule as `video_detection/` itself**: this is a
separate component. It reuses `video_detection/detector.py`,
`classifier.py`, `tracker.py`, and constants/helpers from
`process_video.py` read-only — no detection/classification/tracking
logic is duplicated here, and it never touches `bot.py` or anything it
depends on.

## Run it locally

From the repo root, with the project's usual environment plus this
folder's extra packages:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Needs `models/resnet18_transfer_15ep.pt` and `metadata/class_names.csv`
present (both already in the repo — see note below on the model file).

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (already done — `iambekzodboboev/traffic-sign-recognition`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
3. Pick this repo and branch (`master`), and set the main file path to
   `streamlit_app/app.py`.
4. Streamlit Cloud auto-detects `streamlit_app/requirements.txt` (it
   looks in the same folder as the entrypoint first) and
   `.streamlit/config.toml` at the repo root for theming. No extra
   config needed.
5. Deploy. First build takes a few minutes (installing PyTorch CPU);
   after that, the app sleeps after inactivity and wakes on the next
   visit (normal free-tier behavior).

## Design decisions worth knowing about

- **The trained model (`models/resnet18_transfer_15ep.pt`, ~45MB) is
  committed to the repo**, unlike everywhere else in this project where
  `*.pt` files are gitignored build artifacts. Streamlit Cloud builds
  straight from GitHub with no separate artifact-download step, so the
  model has to actually be in the repo for the public deploy to work.
  Deliberate, one-time exception — see the note in `.gitignore`.
- **Video length is capped at 90 seconds of processing**
  (`MAX_PROCESS_SECONDS` in `app.py`), regardless of how the video
  arrives. Free-tier hosting is a single shared CPU with tight memory —
  this keeps the live-scan experience responsive for everyone using the
  app at once, not just whoever's video happens to be running.
- **Link downloads are capped at 10 minutes** (`MAX_LINK_DOWNLOAD_SECONDS`)
  before any download starts, checked via a metadata-only probe (no
  download yet) — rejects a clearly-too-long link fast instead of
  spending bandwidth on a video that would only ever get partially
  processed anyway.
- **Sampling stays at 4 fps**, matching `process_video.py` exactly —
  stage 9.4 found that a lower sample rate breaks tracking continuity
  for fast-moving signs (see `ROADMAP.md` section 9.4). The response to
  "keep this fast on weaker hardware" is the duration cap above, not a
  lower sample rate, so tracking quality doesn't regress.
