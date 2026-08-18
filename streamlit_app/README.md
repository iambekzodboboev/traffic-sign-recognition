# Signal — combined project showcase + live scanner (public web app)

A single Streamlit app with two pages, switched via a top nav (no page
reload, state preserved):

- **Overview** — the project showcase: the problem, the four-stage
  pipeline, headline results (99.37% test accuracy) with a
  baseline-vs-final comparison, and an honest "known limitations"
  section. Content is the same substance as the graded showcase page
  (`docs/index.html`), restyled to the same dark theme as the scanner so
  the two pages read as one product rather than two different designs.
- **Live Scanner** — upload a driving video or paste a link, watch the
  video actually play while signs get logged live as it scans, then get
  a full trip report when it's done — click any sign in the report and
  the video jumps to that exact moment, paused, with a box and
  name/confidence label marking it.

**Isolation, same rule as `video_detection/` itself**: this is a
separate component. It reuses `video_detection/detector.py`,
`classifier.py`, `tracker.py`, and constants/helpers from
`process_video.py` read-only — no detection/classification/tracking
logic is duplicated here, and it never touches `bot.py` or anything it
depends on.

**Note on `docs/index.html`**: that file is the separately graded
course showcase page and is deliberately left untouched — it has its
own "do not redesign" rule from the assignment. This app's Overview
page is a new, independent presentation of the same true facts, not a
replacement of that file.

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

- **Page switching is a plain `st.session_state["page"]` flag, not
  `st.tabs` or a multi-file `st.navigation` setup.** `st.tabs` was
  tried conceptually and rejected: Streamlit can't be told which tab is
  selected, so any `st.rerun()` during the scan (there are several)
  would silently bounce the user back to the first tab. A session-state
  flag set by plain nav buttons persists correctly across every rerun.
- **The app theme moved from light to dark** (`.streamlit/config.toml`:
  `base = "dark"`, cyan `primaryColor`) to match the showcase content's
  own palette, and the scanner's own CSS (category pills, result cards,
  the embedded video-player component) was recolored to match — the
  goal was one visual system across both pages, not two apps stapled
  together.
- **`stat_card_html()` and `bar_row_html()` are small shared helpers**
  used by both pages — the Overview page's results metrics/comparison
  bars and the Scanner's trip-report stat cards/category breakdown
  render through the same two functions, so they're guaranteed to look
  identical rather than visually drifting apart over time.
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
- **The uploaded/downloaded video is trimmed and transcoded before
  anything else happens** (`prepare_canonical_clip` in `app.py`, via
  `imageio-ffmpeg` — a static ffmpeg binary bundled in the pip package,
  no system `ffmpeg` install needed, works identically locally and on
  Streamlit Cloud). Two reasons at once: most source videos (arbitrary
  phone recordings, Instagram/YouTube downloads) aren't guaranteed to
  use a browser-playable codec, and this also produces the single
  trimmed clip that both the detection pass *and* on-page playback use
  — so a report entry's timestamp always matches the same clip the
  player is scrubbing, with no separate re-encode step to drift out of
  sync.
- **The results-view video player is a genuinely interactive HTML/JS
  component** (`render_interactive_player` in `app.py`, via
  `st.components.v1.html`), not a plain Streamlit video widget — clicking
  a sign in the list calls `video.currentTime = ...` and `video.pause()`,
  then draws a positioned marker box using that sign's detection box
  (captured as *fractions* of frame width/height during the scan, so it
  stays correctly placed regardless of how large the player renders).
  **The same marker is also driven automatically off the video's own
  `timeupdate` event** — as the video plays, each sign gets boxed and
  labeled right as the playhead reaches it, no click needed. This is
  deliberately *not* the same as live server-side recognition keeping
  pace with live playback (there's no reliable way to pace a Python
  detection loop to a browser's real-time video clock in Streamlit) —
  detection already finished during the scan, and replaying it against
  the video's real clock is both simpler and exactly accurate, unlike
  trying to race the two live.
  The clip is embedded directly as a base64 data URI rather than served
  from a file path, since Streamlit doesn't have a built-in static file
  server for arbitrary per-session temp files — kept small enough for
  this (a few MB after the trim+transcode above) by capping both
  duration and output resolution.
