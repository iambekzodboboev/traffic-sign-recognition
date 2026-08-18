"""Signal -- combined project showcase + live video sign-scanner (public web app).

One Streamlit app, two pages selected via st.session_state["page"]:
  - "home": project overview (problem, pipeline, results, honest limitations),
    content sourced from docs/index.html's showcase copy, restyled to match.
  - "scan": the interactive scanner (upload/link -> live scan -> trip report).

Isolated component, same rule as video_detection/ itself: never touches
bot.py or anything it depends on. Reuses detector.py, classifier.py,
tracker.py, and process_video.py's constants/helpers read-only -- no
detection/classification/tracking logic is duplicated here, only the
Streamlit UI around it.

Run locally:
    streamlit run streamlit_app/app.py
"""
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "video_detection"))
sys.path.insert(0, str(PROJECT_ROOT))

from detector import detect_candidate_boxes  # noqa: E402
from classifier import classify_crop, load_classifier  # noqa: E402
from tracker import SignTracker  # noqa: E402
from process_video import SAMPLE_FPS, CONFIDENCE_THRESHOLD, _clip_box  # noqa: E402

GITHUB_URL = "https://github.com/iambekzodboboev/traffic-sign-recognition"

MAX_PROCESS_SECONDS = 90  # cap live scanning time so it stays responsive on free hosting
MAX_LINK_DOWNLOAD_SECONDS = 600  # reject links longer than this before even downloading
CLIP_MAX_WIDTH = 640  # keep the browser-playable clip small enough to embed
MARKER_COLOR = "#6CE5FF"

# (background, foreground) pairs -- tuned for readability on the app's dark surface
CATEGORY_COLORS = {
    "Priority": ("rgba(100,240,181,.16)", "#7CF0C1"),
    "Warning": ("rgba(255,200,107,.16)", "#FFC86B"),
    "Prohibitory": ("rgba(255,107,139,.16)", "#FF93AB"),
    "Mandatory": ("rgba(139,124,255,.20)", "#C0B3FF"),
    "Special regulations": ("rgba(108,229,255,.16)", "#6CE5FF"),
    "Information": ("rgba(108,229,255,.10)", "#9FE0F2"),
    "Service": ("rgba(180,205,255,.14)", "#C3D3FF"),
    "Additional": ("rgba(255,255,255,.09)", "#D6DEEB"),
}
UNSURE_COLOR = ("rgba(255,200,107,.12)", "#FFC86B")

st.set_page_config(page_title="Signal -- Traffic Sign Recognition", page_icon="\U0001F6A6", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --bg: #07101f; --bg-2: #0b1830; --surface: rgba(15,28,52,.72);
      --line: rgba(180,205,255,.16); --line-strong: rgba(180,205,255,.28);
      --text: #f5f8ff; --muted: #9caac1;
      --accent: #6ce5ff; --accent-2: #8b7cff; --accent-3: #64f0b5; --warning: #ffc86b;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
      background:
        radial-gradient(900px 600px at 15% -10%, color-mix(in srgb, var(--accent) 10%, transparent), transparent 70%),
        radial-gradient(800px 560px at 100% 0%, color-mix(in srgb, var(--accent-2) 14%, transparent), transparent 70%),
        linear-gradient(160deg, var(--bg), var(--bg-2) 55%, var(--bg));
    }
    .block-container {max-width: 1120px; padding-top: 1.6rem; padding-bottom: 3rem; margin: 0 auto;}
    html, body, [class*="css"] {font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}

    /* -- nav / brand -- */
    .brand-title {font-weight: 800; font-size: 1.7rem; letter-spacing: -.02em;
      background: linear-gradient(105deg, var(--text) 10%, var(--accent) 55%, var(--accent-2) 90%);
      -webkit-background-clip: text; background-clip: text; color: transparent; line-height: 1.1;}
    .brand-tag {color: var(--muted); font-size: .78rem; font-weight: 600; letter-spacing: .02em; margin-top: 2px;}
    hr.nav-rule {border: none; border-top: 1px solid var(--line); margin: .9rem 0 1.6rem;}

    /* -- hero -- */
    .eyebrow {display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
      border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent); background: color-mix(in srgb, var(--accent) 9%, transparent);
      color: var(--accent); font-size: .72rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 18px;}
    .hero-title {font-size: clamp(38px, 5vw, 62px); line-height: 1.05; letter-spacing: -.04em; margin: 0 0 18px; max-width: 780px; color: var(--text);}
    .hero-title .accent {background: linear-gradient(105deg, var(--accent) 10%, var(--accent-2) 80%); -webkit-background-clip: text; background-clip: text; color: transparent;}
    .hero-lead {font-size: 1.05rem; line-height: 1.7; color: var(--muted); max-width: 640px; margin: 0 0 22px;}
    .chip-row {display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 26px;}
    .chip {padding: 7px 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,.03); color: var(--muted); font-size: .74rem; font-weight: 700;}
    .chip b {color: var(--text); margin-right: 4px;}

    /* -- sections -- */
    .section-gap {height: 46px;}
    .section-kicker {color: var(--accent); font-size: .72rem; font-weight: 850; text-transform: uppercase; letter-spacing: .14em; margin-bottom: 10px;}
    .section-title {font-size: clamp(24px, 3vw, 34px); letter-spacing: -.03em; color: var(--text); max-width: 700px; margin-bottom: 6px;}
    .section-sub {color: var(--muted); font-size: .92rem; max-width: 560px; margin-bottom: 24px;}

    /* -- glass cards -- */
    .glass-card {border: 1px solid var(--line); background: linear-gradient(150deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
      border-radius: 18px; padding: 22px 24px; height: 100%;}
    .glass-card h3 {font-size: 1.05rem; margin: 0 0 12px; color: var(--text); letter-spacing: -.01em;}
    .glass-card p {color: var(--muted); line-height: 1.7; font-size: .88rem; margin: 0;}
    .card-icon {font-size: 1.4rem; margin-bottom: 10px;}

    /* -- stat cards (shared by Overview results + Scanner trip report) -- */
    .stat-card {background: rgba(255,255,255,.045); border: 1px solid var(--line); border-radius: 14px; padding: 16px; text-align: center; height: 100%;}
    .stat-card b {display: block; font-size: 1.55rem; font-variant-numeric: tabular-nums; color: var(--text); letter-spacing: -.02em;}
    .stat-label {display: block; font-size: .72rem; color: var(--muted); margin-top: 4px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;}
    .stat-note {display: block; font-size: .74rem; color: var(--muted); margin-top: 6px; line-height: 1.4;}

    /* -- bar rows (shared by Overview comparison + Scanner category breakdown) -- */
    .bar-row {display: grid; grid-template-columns: 190px 1fr 46px; align-items: center; gap: 10px; font-size: .84rem; color: var(--muted); margin-bottom: 6px;}
    .bar-row span:last-child {text-align: right; color: var(--text); font-weight: 700;}
    .bar-track {height: 8px; background: rgba(255,255,255,.06); border-radius: 4px; overflow: hidden;}
    .bar-fill {height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--accent), var(--accent-2));}

    /* -- pipeline steps -- */
    .step-card {border: 1px solid var(--line); background: rgba(255,255,255,.03); border-radius: 16px; padding: 20px; height: 100%; position: relative;}
    .step-num {width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; font-size: .78rem; font-weight: 800;
      background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); margin-bottom: 14px;}
    .step-card h4 {font-size: .98rem; margin: 0 0 6px; color: var(--text);}
    .step-card p {color: var(--muted); font-size: .82rem; line-height: 1.6; margin: 0;}

    /* -- result bullets -- */
    .result-line {display: flex; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--line); font-size: .85rem;}
    .result-line:last-child {border-bottom: none;}
    .result-line i {width: 8px; height: 8px; min-width: 8px; margin-top: 6px; border-radius: 50%; background: var(--accent-3); box-shadow: 0 0 10px var(--accent-3);}
    .result-line b {display: block; color: var(--text); font-size: .87rem;}
    .result-line span {display: block; color: var(--muted); margin-top: 3px; line-height: 1.55;}

    /* -- limitations -- */
    .limit-box {border-radius: 18px; padding: 24px; background: linear-gradient(120deg, color-mix(in srgb, var(--warning) 8%, var(--surface)), var(--surface)); border: 1px solid color-mix(in srgb, var(--warning) 30%, transparent);}
    .limit-box h3 {margin: 0 0 12px; color: var(--warning); font-size: 1rem;}
    .limit-box ul {margin: 0; padding-left: 18px; color: var(--muted); font-size: .87rem; line-height: 1.75;}
    .limit-box li strong {color: var(--text);}

    /* -- CTA -- */
    .cta-box {text-align: center; border-radius: 24px; padding: 40px 24px;
      background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 12%, var(--bg-2)), color-mix(in srgb, var(--accent-2) 16%, var(--bg-2)));
      border: 1px solid color-mix(in srgb, var(--accent) 32%, transparent); margin: 8px 0 4px;}
    .cta-box h2 {font-size: clamp(22px, 3vw, 32px); letter-spacing: -.03em; color: var(--text); margin: 0 0 10px;}
    .cta-box p {color: var(--muted); max-width: 480px; margin: 0 auto 6px;}

    /* -- scanner: live log feed -- */
    .scanner-tag {color: var(--accent); font-size: .72rem; font-weight: 850; text-transform: uppercase; letter-spacing: .14em; margin-bottom: 6px;}
    .pill {display: inline-block; font-size: .68rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; white-space: nowrap;}
    .feed-row {display: flex; align-items: center; gap: 10px; padding: 9px 12px; border: 1px solid var(--line);
      border-radius: 10px; margin-bottom: 6px; font-size: .85rem; background: rgba(255,255,255,.03);}
    .feed-row .name {flex: 1; font-weight: 700; color: var(--text);}
    .feed-row .ts {color: var(--muted); font-variant-numeric: tabular-nums; font-size: .78rem;}
    .callout {background: rgba(255,255,255,.045); border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px; font-size: .88rem; margin: .8rem 0; color: var(--text);}
    .callout b {color: var(--accent);}

    /* -- footer -- */
    .signal-footer {margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--line);
      display: flex; justify-content: space-between; gap: 16px; color: var(--muted); font-size: .78rem; flex-wrap: wrap;}
    .signal-footer strong {color: var(--text);}
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_card_html(value, label, note=""):
    note_html = f'<span class="stat-note">{note}</span>' if note else ""
    return f'<div class="stat-card"><b>{value}</b><span class="stat-label">{label}</span>{note_html}</div>'


def bar_row_html(label, value_display, width_pct):
    return (
        f'<div class="bar-row"><span>{label}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{width_pct}%"></div></div>'
        f'<span>{value_display}</span></div>'
    )


@st.cache_resource(show_spinner="Loading the sign classifier (one-time)...")
def get_classifier():
    return load_classifier()


def get_video_meta(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0
    return fps, frame_count, width, height, duration


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def workdir():
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="signal_")
    return Path(st.session_state.workdir)


def reset_app():
    """Clears the scan session but keeps the user on whichever top-level
    page (Overview/Scanner) they were on -- "New video" should return to
    the upload screen, not bounce back to the Overview page."""
    wd = st.session_state.get("workdir")
    if wd and Path(wd).exists():
        shutil.rmtree(wd, ignore_errors=True)
    keep_page = st.session_state.get("page", "home")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.page = keep_page
    st.rerun()


def download_from_link(url, dest_dir):
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration")
    if duration and duration > MAX_LINK_DOWNLOAD_SECONDS:
        raise ValueError(
            f"That video is {fmt_time(duration)} long. This demo only accepts links up to "
            f"{fmt_time(MAX_LINK_DOWNLOAD_SECONDS)} -- try a shorter clip."
        )

    out_template = str(dest_dir / "source.%(ext)s")
    ydl_opts = {"quiet": True, "no_warnings": True, "format": "b", "outtmpl": out_template, "noplaylist": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def prepare_canonical_clip(source_path, dest_dir):
    """Trims to MAX_PROCESS_SECONDS and transcodes to browser-playable H.264,
    via the static ffmpeg binary bundled by imageio-ffmpeg (no system ffmpeg
    needed, works the same locally and on Streamlit Cloud). This clip is
    used both for the actual detection pass and for on-page playback, so
    the two are always talking about the exact same frames/timestamps."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out_path = Path(dest_dir) / "clip.mp4"
    cmd = [
        ffmpeg_exe, "-y", "-i", str(source_path),
        "-t", str(MAX_PROCESS_SECONDS),
        "-an",
        "-vf", f"scale='min({CLIP_MAX_WIDTH},iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"couldn't prepare the video for playback ({result.stderr[-300:]})")
    return out_path


def run_scan(video_path):
    if st.session_state.get("cap_message"):
        st.info(st.session_state.cap_message)

    model, class_names_df = get_classifier()
    fps, frame_count, width, height, duration = get_video_meta(video_path)
    frame_stride = max(1, round(fps / SAMPLE_FPS))

    st.video(str(video_path), autoplay=True, muted=True)

    cap = cv2.VideoCapture(str(video_path))
    tracker = SignTracker()
    first_seen_sample = {}
    first_seen_box = {}

    progress_ph = st.progress(0.0)
    status_ph = st.empty()
    feed_header_ph = st.empty()
    feed_ph = st.empty()

    feed_header_ph.markdown("**Live log**")
    history = []
    frame_idx = 0
    sample_idx = 0
    start = time.time()

    while frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            boxes = detect_candidate_boxes(frame)
            detections = []
            for (x, y, w, h) in boxes:
                x, y, w, h = _clip_box(x, y, w, h, width, height)
                if w <= 0 or h <= 0:
                    continue
                crop = frame[y:y + h, x:x + w]
                results = classify_crop(crop, model, class_names_df, top_k=1)
                class_id, name, category, confidence = results[0]
                detections.append(((x, y, w, h), class_id, name, category, confidence))

            track_ids = tracker.update(sample_idx, detections)

            for (box, class_id, name, category, confidence), track_id in zip(detections, track_ids):
                x, y, w, h = box
                if track_id not in first_seen_sample:
                    first_seen_sample[track_id] = sample_idx
                    first_seen_box[track_id] = {
                        "x": x / width, "y": y / height, "w": w / width, "h": h / height,
                    }
                    entry_category = category if confidence >= CONFIDENCE_THRESHOLD else "unsure"
                    history.insert(0, {"time": fmt_time(sample_idx / SAMPLE_FPS), "name": name, "category": entry_category})

            progress_ph.progress(min(1.0, frame_idx / max(frame_count, 1)))
            confident_count = sum(1 for h in history if h["category"] != "unsure")
            status_ph.markdown(
                f"Scanning -- {fmt_time(frame_idx / fps)} / {fmt_time(duration)}  ·  "
                f"**{confident_count} signs found so far**"
            )

            rows_html = ""
            for h in history[:8]:
                cat = h["category"]
                label = "Unsure" if cat == "unsure" else cat
                bg, fg = UNSURE_COLOR if cat == "unsure" else CATEGORY_COLORS.get(cat, ("rgba(255,255,255,.06)", "var(--muted)"))
                rows_html += (
                    f'<div class="feed-row"><span class="name">{h["name"]}</span>'
                    f'<span class="pill" style="background:{bg};color:{fg};">{label}</span>'
                    f'<span class="ts">{h["time"]}</span></div>'
                )
            feed_ph.markdown(rows_html or "_nothing confident yet_", unsafe_allow_html=True)

            sample_idx += 1

        frame_idx += 1

    cap.release()
    reports = tracker.finish()
    for r in reports:
        sample = first_seen_sample.get(r["track_id"], 0)
        r["first_seen_s"] = sample / SAMPLE_FPS
        r["box"] = first_seen_box.get(r["track_id"], {"x": 0, "y": 0, "w": 0, "h": 0})

    elapsed = time.time() - start
    st.session_state.reports = reports
    st.session_state.video_meta = {
        "duration": duration, "fps": fps, "frames_scanned": sample_idx,
        "elapsed": elapsed,
    }
    st.session_state.stage = "done"
    st.rerun()


def render_interactive_player(clip_path, reports):
    """A single self-contained HTML component: the clip, plus a clickable
    list of every sign found. Clicking a row seeks the video to that
    moment, pauses it, and draws a box + name/confidence label over the
    sign, using the (x, y, w, h) fractions captured during the scan.

    The same marker is also driven automatically off the video's own
    playback clock (its `timeupdate` event) -- as the video plays, each
    sign gets boxed and labeled right as the playhead reaches it, no
    click needed. This is what actually gives a "recognized live"
    experience: true frame-accurate sync isn't achievable between a
    live server-side detection loop and live browser playback (nothing
    paces one to the other), but here the detection has already
    happened and its timestamps are exact, so replaying against the
    video's real clock is both simpler and more accurate.

    Rendered inside its own iframe (components.html), so it carries its
    own <style> block matching the outer app's dark theme -- it does not
    inherit the page-level CSS injected above."""
    clip_b64 = base64.b64encode(Path(clip_path).read_bytes()).decode("ascii")

    signs = []
    for r in sorted(reports, key=lambda r: r["first_seen_s"]):
        bg, fg = CATEGORY_COLORS.get(r["category"], ("rgba(255,255,255,.06)", "#D6DEEB"))
        signs.append({
            "time": r["first_seen_s"], "time_label": fmt_time(r["first_seen_s"]),
            "name": r["name"], "category": r["category"],
            "confidence_pct": round(r["confidence"] * 100),
            "x": r["box"]["x"], "y": r["box"]["y"], "w": r["box"]["w"], "h": r["box"]["h"],
            "bg": bg, "fg": fg,
        })

    rows_html = "".join(
        f'<div class="prow" id="prow-{i}" onclick="seekTo({i})">'
        f'<span class="pname">{s["name"]}</span>'
        f'<span class="ppill" style="background:{s["bg"]};color:{s["fg"]};">{s["category"]}</span>'
        f'<span class="pts">{s["confidence_pct"]}%</span>'
        f'<span class="pts">{s["time_label"]}</span>'
        f'</div>'
        for i, s in enumerate(signs)
    )

    html = f"""
    <div class="player-wrap">
      <video id="rv" controls playsinline autoplay muted>
        <source src="data:video/mp4;base64,{clip_b64}" type="video/mp4">
      </video>
      <div id="marker" class="marker"><span id="marker-label" class="marker-label"></span></div>
    </div>
    <div class="plist">{rows_html}</div>
    <style>
      * {{ box-sizing: border-box; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
      html, body {{ margin: 0; background: #07101f; }}
      .player-wrap {{ position: relative; margin-bottom: 12px; line-height: 0; }}
      video {{ width: 100%; max-height: 420px; display: block; background: #000; margin: 0 auto; border-radius: 12px 12px 0 0; }}
      .marker {{ position: absolute; border: 3px solid {MARKER_COLOR}; display: none; pointer-events: none; border-radius: 2px; }}
      .marker-label {{ position: absolute; bottom: 100%; left: -3px; background: {MARKER_COLOR}; color: #07101f;
                        font-size: 12px; font-weight: 700; padding: 2px 6px; white-space: nowrap; border-radius: 3px 3px 0 0; }}
      .plist {{ max-height: 320px; overflow-y: auto; border: 1px solid rgba(180,205,255,.16); border-top: none; border-radius: 0 0 12px 12px; background: rgba(255,255,255,.03); }}
      .prow {{ display: flex; align-items: center; gap: 10px; padding: 9px 14px; font-size: 14px; color: #f5f8ff;
               border-bottom: 1px solid rgba(180,205,255,.14); cursor: pointer; }}
      .prow:last-child {{ border-bottom: none; }}
      .prow:hover {{ background: rgba(255,255,255,.05); }}
      .prow.active {{ background: rgba(108,229,255,.12); }}
      .pname {{ flex: 1; font-weight: 700; color: #f5f8ff; }}
      .ppill {{ font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 999px; white-space: nowrap; }}
      .pts {{ color: #9caac1; font-variant-numeric: tabular-nums; font-size: 12px; min-width: 34px; text-align: right; }}
    </style>
    <script>
      const SIGNS = {json.dumps(signs)};
      const MARKER_WINDOW = 2.0;
      const video = document.getElementById('rv');
      const marker = document.getElementById('marker');
      const label = document.getElementById('marker-label');
      let manualIndex = null;

      function showMarker(s, i) {{
        marker.style.left = (s.x * 100) + '%';
        marker.style.top = (s.y * 100) + '%';
        marker.style.width = (s.w * 100) + '%';
        marker.style.height = (s.h * 100) + '%';
        label.textContent = s.name + ' ' + s.confidence_pct + '%';
        marker.style.display = 'block';
        document.querySelectorAll('.prow').forEach(function(el) {{ el.classList.remove('active'); }});
        if (i !== null) document.getElementById('prow-' + i).classList.add('active');
      }}

      function hideMarker() {{
        marker.style.display = 'none';
        document.querySelectorAll('.prow').forEach(function(el) {{ el.classList.remove('active'); }});
      }}

      function seekTo(i) {{
        manualIndex = i;
        video.currentTime = SIGNS[i].time;
        video.pause();
        showMarker(SIGNS[i], i);
      }}

      video.addEventListener('timeupdate', function() {{
        if (video.paused) return;
        manualIndex = null;
        const t = video.currentTime;
        let bestI = null, bestS = null;
        for (let i = 0; i < SIGNS.length; i++) {{
          const s = SIGNS[i];
          if (s.time <= t && t - s.time < MARKER_WINDOW) {{ bestI = i; bestS = s; }}
        }}
        if (bestS) {{ showMarker(bestS, bestI); }} else {{ hideMarker(); }}
      }});

      video.addEventListener('seeked', function() {{
        if (manualIndex !== null) return;
        hideMarker();
      }});
    </script>
    """
    components.html(html, height=780, scrolling=True)


def render_navbar():
    cols = st.columns([3, 1, 1.3, 1.3])
    with cols[0]:
        st.markdown(
            '<div class="brand-title">Signal</div>'
            '<div class="brand-tag">Traffic Sign Recognition &middot; AI/ML Capstone</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button("Overview", use_container_width=True,
                      type="primary" if st.session_state.page == "home" else "secondary"):
            st.session_state.page = "home"
            st.rerun()
    with cols[2]:
        if st.button("Live Scanner", use_container_width=True,
                      type="primary" if st.session_state.page == "scan" else "secondary"):
            st.session_state.page = "scan"
            st.rerun()
    with cols[3]:
        st.link_button("GitHub ↗", GITHUB_URL, use_container_width=True)
    st.markdown('<hr class="nav-rule">', unsafe_allow_html=True)


def render_home():
    st.markdown('<div class="eyebrow">AI/ML Fundamentals Capstone</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Signal finds every <span class="accent">road sign</span> in your drive.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-lead">A computer-vision pipeline that reads Post-Soviet-states traffic '
        'signs from a single photo or a full driving video, and reports every sign it found with '
        'a timestamp and confidence score.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="chip-row">
          <span class="chip"><b>Task</b>Image classification + video detection</span>
          <span class="chip"><b>Model</b>ResNet18 (transfer learning)</span>
          <span class="chip"><b>Author</b>Bekzod Boboev</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1.1, 1.1, 2])
    with c1:
        if st.button("Try the Live Scanner →", type="primary", use_container_width=True, key="hero_scan"):
            st.session_state.page = "scan"
            st.rerun()
    with c2:
        st.link_button("View Repository", GITHUB_URL, use_container_width=True)

    # -- Context --
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">01 · Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The project in one clear story.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Recognizing an isolated, cropped sign photo is the easy version '
        'of this problem &mdash; real driving footage buries each sign in a moving, cluttered scene.</div>',
        unsafe_allow_html=True,
    )
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(
            '<div class="glass-card"><div class="card-icon">\U0001F3AF</div><h3>The problem</h3>'
            '<p>A classifier trained on clean, sign-only photos performs very differently on a real '
            'driving scene, where the sign is a small, moving detail in a busy frame &mdash; confirmed '
            'directly in this project when a wide street photo was misread by the first deployed version.</p></div>',
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            '<div class="glass-card"><div class="card-icon">\U0001F464</div><h3>Who uses it</h3>'
            '<p>Driving students and instructors reviewing a practice drive, or anyone curious what '
            'signs actually appear in a piece of driving footage &mdash; a quick photo lookup, or a full '
            'drive sign report.</p></div>',
            unsafe_allow_html=True,
        )

    # -- System / pipeline --
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">02 · System</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How it works, without the notebook noise.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Four stages, from a raw photo or video to a named, timestamped result.</div>',
        unsafe_allow_html=True,
    )
    steps = [
        ("01", "Photo or video", "A single sign photo, or an uploaded/linked driving video &mdash; 200 real sign classes, 116,642 training images."),
        ("02", "Detect &amp; crop", "Classical color + shape detection (OpenCV) finds candidate sign regions in each sampled video frame &mdash; no external detector model needed."),
        ("03", "ResNet18 classifier", "A pretrained ResNet18, fine-tuned on the full dataset, predicts the sign type and category for each crop."),
        ("04", "Tracked trip report", "Detections are matched across frames so the same physical sign isn't reported twice &mdash; the result is a named, timestamped, confidence-scored list."),
    ]
    step_cols = st.columns(4)
    for col, (num, title, body) in zip(step_cols, steps):
        with col:
            st.markdown(
                f'<div class="step-card"><div class="step-num">{num}</div><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    # -- Evidence / results --
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">03 · Evidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results, measured on data the model never trained on.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">The held-out test set was untouched until the final model was '
        "already chosen, so these numbers aren't tuned against it.</div>",
        unsafe_allow_html=True,
    )
    stat_cols = st.columns(4)
    stats = [
        ("99.37%", "Test accuracy", "200 classes, random-guess baseline 0.5%"),
        ("0.9937", "Macro F1 (test)", "Consistent across all 200 classes"),
        ("88.46%", "Worst-class accuracy", "The hardest class still beats chance easily"),
        ("87.84%", "Baseline (from scratch)", "Before switching to a pretrained backbone"),
    ]
    for col, (value, label, note) in zip(stat_cols, stats):
        with col:
            st.markdown(stat_card_html(value, label, note), unsafe_allow_html=True)

    st.markdown('<div class="section-gap" style="height:22px;"></div>', unsafe_allow_html=True)
    e1, e2 = st.columns([1.05, 0.95])
    with e1:
        bars = "".join([
            bar_row_html("Baseline (from-scratch CNN)", "87.84%", 87.84),
            bar_row_html("Final model (ResNet18)", "99.37%", 99.37),
            bar_row_html('"Unsure" confidence cutoff', "60%", 60),
        ])
        st.markdown(f'<div class="glass-card"><h3>Comparison</h3>{bars}</div>', unsafe_allow_html=True)
    with e2:
        st.markdown(
            '<div class="glass-card"><h3>What this proves</h3>'
            '<div class="result-line"><i></i><div><b>Targeted improvement, not generic inflation</b>'
            '<span>Specific baseline confusions (visually similar sign pairs) dropped to near-zero '
            'after switching backbones &mdash; harder to explain away as overfitting than a uniform bump.</span></div></div>'
            '<div class="result-line"><i></i><div><b>Consistent across all 200 classes</b>'
            '<span>Macro-F1 sits almost exactly at overall accuracy, not propped up by a few frequent classes.</span></div></div>'
            '<div class="result-line"><i></i><div><b>Tested on real driving video, not just clean photos</b>'
            '<span>Ran against two different real driving videos, with failure modes documented honestly, not just successes.</span></div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # -- Limitations --
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="limit-box"><h3>⚠ Known limitations</h3><ul>'
        '<li><strong>Fixed camera-mount assumption.</strong> The video detector\'s search region was built '
        'and tested against dashboard-mounted camera clips; a very different angle or mount (e.g. a much '
        'higher windshield mount, or landscape video) hasn\'t yet been verified to work as well.</li>'
        '<li><strong>Overlays on top of a sign.</strong> A watermark or on-screen caption drawn directly over '
        'a sign can cause a confident misclassification &mdash; no box-boundary fix can remove pixels that '
        'sit inside the sign\'s own region.</li>'
        '<li><strong>Two signs sharing one post.</strong> Adjacent or stacked signs can merge into a single '
        'detection box; this usually shows up as low confidence, but isn\'t a guarantee &mdash; a merged '
        'silhouette can occasionally resemble another class strongly enough to score high.</li>'
        '<li><strong>Small, blurry crops.</strong> Confidence is a weaker trust signal on tiny video crops '
        'than on the large, clean photos the classifier was validated against.</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )

    # -- CTA --
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cta-box"><h2>Ready to see it find real signs?</h2>'
        '<p>Upload your own driving video or paste a public link, and get a full trip report in seconds.</p></div>',
        unsafe_allow_html=True,
    )
    cta1, cta2, _ = st.columns([1.1, 1.1, 2])
    with cta1:
        if st.button("Try the Live Scanner →", type="primary", use_container_width=True, key="cta_scan"):
            st.session_state.page = "scan"
            st.rerun()
    with cta2:
        st.link_button("View Repository", GITHUB_URL, use_container_width=True)

    render_footer()


def render_footer():
    st.markdown(
        '<div class="signal-footer"><div><strong>Bekzod Boboev</strong> · AI/ML Fundamentals Capstone</div>'
        '<div>Traffic Sign Recognition showcase · ResNet18, 99.37% test accuracy</div></div>',
        unsafe_allow_html=True,
    )


def render_scanner():
    left, mid, right = st.columns([1, 5, 1])
    with mid:
        st.markdown('<div class="scanner-tag">Live Scanner</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-lead" style="margin-bottom:1.2rem;">Upload a driving video, or paste a '
            "link. We'll find every road sign in it.</p>",
            unsafe_allow_html=True,
        )

        if "stage" not in st.session_state:
            st.session_state.stage = "upload"

        if st.session_state.stage == "upload":
            uploaded = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
            st.caption("or paste a link (e.g. a public Instagram/YouTube video)")
            link = st.text_input("Video link", label_visibility="collapsed", placeholder="https://...")

            if st.button("Scan video", type="primary", disabled=not (uploaded or link)):
                wd = workdir()
                try:
                    if uploaded is not None:
                        raw_path = wd / uploaded.name
                        raw_path.write_bytes(uploaded.getbuffer())
                    else:
                        with st.spinner("Fetching the video..."):
                            raw_path = download_from_link(link, wd)

                    _, _, _, _, raw_duration = get_video_meta(raw_path)
                    cap_message = None
                    if raw_duration > MAX_PROCESS_SECONDS:
                        cap_message = (
                            f"This video is {fmt_time(raw_duration)} long -- scanning the first "
                            f"{fmt_time(MAX_PROCESS_SECONDS)} to keep the demo responsive."
                        )

                    with st.spinner("Preparing the video..."):
                        clip_path = prepare_canonical_clip(raw_path, wd)
                except Exception as e:
                    st.error(f"Couldn't load that video: {e}")
                else:
                    st.session_state.video_path = str(clip_path)
                    st.session_state.cap_message = cap_message
                    st.session_state.stage = "scanning"
                    st.rerun()

            st.caption("ResNet18 classifier · 200 sign classes, 99.37% test accuracy · classical CV detector")

        elif st.session_state.stage == "scanning":
            run_scan(Path(st.session_state.video_path))

        elif st.session_state.stage == "done":
            reports = st.session_state.reports
            meta = st.session_state.video_meta

            if not reports:
                st.warning("No signs found with enough confidence in this video. Try a clearer driving clip.")
            else:
                counts = Counter(r["category"] for r in reports)
                most_common_name, most_common_n = Counter(r["name"] for r in reports).most_common(1)[0]

                trip_stats = [
                    (len(reports), "signs found", ""),
                    (fmt_time(meta["duration"]), "drive time", ""),
                    (meta["frames_scanned"], "frames scanned", ""),
                ]
                trip_cols = st.columns(3)
                for col, (value, label, note) in zip(trip_cols, trip_stats):
                    with col:
                        st.markdown(stat_card_html(value, label, note), unsafe_allow_html=True)

                st.markdown("**By category**")
                max_count = max(counts.values())
                bars_html = "".join(
                    bar_row_html(category, str(n), int(n / max_count * 100))
                    for category, n in counts.most_common()
                )
                st.markdown(bars_html, unsafe_allow_html=True)

                st.markdown(
                    f'<div class="callout">Most common: <b>{most_common_name}</b> – spotted {most_common_n} time'
                    f'{"s" if most_common_n != 1 else ""}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**All signs found** — click a sign to jump the video to that moment")
                render_interactive_player(Path(st.session_state.video_path), reports)

                report_df = pd.DataFrame([
                    {
                        "time": fmt_time(r["first_seen_s"]), "name": r["name"], "category": r["category"],
                        "confidence_pct": round(r["confidence"] * 100, 1), "frames_seen": r["frames_seen"],
                    }
                    for r in sorted(reports, key=lambda r: r["first_seen_s"])
                ])
                st.download_button(
                    "Download report (CSV)", report_df.to_csv(index=False), file_name="signal_trip_report.csv",
                    mime="text/csv",
                )

            if st.button("New video", type="primary"):
                reset_app()

        render_footer()


if "page" not in st.session_state:
    st.session_state.page = "home"

render_navbar()

if st.session_state.page == "home":
    render_home()
else:
    render_scanner()
