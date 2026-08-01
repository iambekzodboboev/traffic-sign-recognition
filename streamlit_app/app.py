"""Public web demo for the traffic sign video pipeline (Signal UI direction).

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

MAX_PROCESS_SECONDS = 90  # cap live scanning time so it stays responsive on free hosting
MAX_LINK_DOWNLOAD_SECONDS = 600  # reject links longer than this before even downloading
CLIP_MAX_WIDTH = 640  # keep the browser-playable clip small enough to embed
MARKER_COLOR = "#0E7C86"

CATEGORY_COLORS = {
    "Priority": ("#EAF3DE", "#3B6D11"),
    "Warning": ("#FAEEDA", "#854F0B"),
    "Prohibitory": ("#FCEBEB", "#A32D2D"),
    "Mandatory": ("#E8ECFB", "#29349B"),
    "Special regulations": ("#E1F5EE", "#0F6E56"),
    "Information": ("#E6F1FB", "#0C447C"),
    "Service": ("#EEF1F3", "#445164"),
    "Additional": ("#F5EEF7", "#6B3F82"),
}
UNSURE_COLOR = ("#FEF3E2", "#B45309")

st.set_page_config(page_title="Signal - road sign scanner", page_icon="\U0001F6A6", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 720px;}
    .signal-logo {font-weight: 700; font-size: 1.6rem; color: #0E7C86; margin-bottom: .1rem;}
    .signal-tagline {color: #64748B; font-size: .95rem; margin-bottom: 1.4rem;}
    .stat-row {display: flex; gap: 10px; margin: .5rem 0 1.2rem;}
    .stat-card {flex: 1; background: #F4F6F8; border-radius: 10px; padding: 14px; text-align: center;}
    .stat-card b {display: block; font-size: 1.5rem; font-variant-numeric: tabular-nums; color: #16202A;}
    .stat-card span {font-size: .72rem; color: #64748B;}
    .pill {display: inline-block; font-size: .68rem; font-weight: 600; padding: 3px 9px;
           border-radius: 999px; white-space: nowrap;}
    .feed-row {display: flex; align-items: center; gap: 10px; padding: 8px 10px; border: 1px solid #E2E8EE;
               border-radius: 8px; margin-bottom: 6px; font-size: .85rem;}
    .feed-row .name {flex: 1; font-weight: 600;}
    .feed-row .ts {color: #94A3B0; font-variant-numeric: tabular-nums; font-size: .78rem;}
    .bar-row {display: grid; grid-template-columns: 130px 1fr 24px; align-items: center; gap: 8px;
              font-size: .82rem; color: #475569; margin-bottom: 6px;}
    .bar-track {height: 8px; background: #F4F6F8; border-radius: 4px; overflow: hidden;}
    .bar-fill {height: 100%; background: #0E7C86;}
    .callout {background: #F4F6F8; border-radius: 8px; padding: 10px 14px; font-size: .88rem; margin: .8rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
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
    wd = st.session_state.get("workdir")
    if wd and Path(wd).exists():
        shutil.rmtree(wd, ignore_errors=True)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
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
                bg, fg = UNSURE_COLOR if cat == "unsure" else CATEGORY_COLORS.get(cat, ("#F4F6F8", "#475569"))
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
    video's real clock is both simpler and more accurate."""
    clip_b64 = base64.b64encode(Path(clip_path).read_bytes()).decode("ascii")

    signs = []
    for r in sorted(reports, key=lambda r: r["first_seen_s"]):
        bg, fg = CATEGORY_COLORS.get(r["category"], ("#F4F6F8", "#475569"))
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
      .player-wrap {{ position: relative; margin-bottom: 12px; line-height: 0; }}
      video {{ width: 100%; max-height: 420px; display: block; background: #000; margin: 0 auto; }}
      .marker {{ position: absolute; border: 3px solid {MARKER_COLOR}; display: none; pointer-events: none; }}
      .marker-label {{ position: absolute; bottom: 100%; left: -3px; background: {MARKER_COLOR}; color: #fff;
                        font-size: 12px; font-weight: 600; padding: 2px 6px; white-space: nowrap; }}
      .plist {{ max-height: 320px; overflow-y: auto; border: 1px solid #E2E8EE; border-radius: 8px; }}
      .prow {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; font-size: 14px;
               border-bottom: 1px solid #E2E8EE; cursor: pointer; }}
      .prow:last-child {{ border-bottom: none; }}
      .prow:hover {{ background: #F4F6F8; }}
      .prow.active {{ background: #E1F5EE; }}
      .pname {{ flex: 1; font-weight: 600; color: #16202A; }}
      .ppill {{ font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 999px; white-space: nowrap; }}
      .pts {{ color: #94A3B0; font-variant-numeric: tabular-nums; font-size: 12px; min-width: 34px; text-align: right; }}
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

      // Drives the marker off the video's own playback clock: whichever
      // sign's timestamp the playhead most recently passed (within
      // MARKER_WINDOW seconds) gets marked automatically, so signs get
      // boxed and labeled in real time as the video plays -- no click
      // needed. Recomputed from currentTime on every tick, so it's
      // correct whether playing forward, paused, or manually scrubbed.
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


st.markdown('<div class="signal-logo">Signal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="signal-tagline">Upload a driving video, or paste a link. '
    "We'll find every road sign in it.</div>",
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

    st.caption("ResNet18 classifier - 200 sign classes, 99.37% test accuracy - classical CV detector")

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

        st.markdown(
            f"""
            <div class="stat-row">
              <div class="stat-card"><b>{len(reports)}</b><span>signs found</span></div>
              <div class="stat-card"><b>{fmt_time(meta['duration'])}</b><span>drive time</span></div>
              <div class="stat-card"><b>{meta['frames_scanned']}</b><span>frames scanned</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**By category**")
        max_count = max(counts.values())
        bars_html = ""
        for category, n in counts.most_common():
            width_pct = int(n / max_count * 100)
            bars_html += (
                f'<div class="bar-row"><span>{category}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{width_pct}%"></div></div>'
                f'<span>{n}</span></div>'
            )
        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown(
            f'<div class="callout">Most common: <b>{most_common_name}</b> - spotted {most_common_n} time'
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
