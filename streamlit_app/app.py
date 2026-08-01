"""Public web demo for the traffic sign video pipeline (Signal UI direction).

Isolated component, same rule as video_detection/ itself: never touches
bot.py or anything it depends on. Reuses detector.py, classifier.py,
tracker.py, and process_video.py's constants/helpers read-only -- no
detection/classification/tracking logic is duplicated here, only the
Streamlit UI around it.

Run locally:
    streamlit run streamlit_app/app.py
"""
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "video_detection"))
sys.path.insert(0, str(PROJECT_ROOT))

from detector import detect_candidate_boxes  # noqa: E402
from classifier import classify_crop, load_classifier  # noqa: E402
from tracker import SignTracker  # noqa: E402
from process_video import SAMPLE_FPS, CONFIDENCE_THRESHOLD, _clip_box  # noqa: E402

MAX_PROCESS_SECONDS = 90  # cap live scanning time so it stays responsive on free hosting
MAX_LINK_DOWNLOAD_SECONDS = 600  # reject links longer than this before even downloading

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


def pill_html(label, category):
    bg, fg = CATEGORY_COLORS.get(category, ("#F4F6F8", "#475569"))
    return f'<span class="pill" style="background:{bg};color:{fg};">{label}</span>'


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


def run_scan(video_path):
    model, class_names_df = get_classifier()
    fps, frame_count, width, height, duration = get_video_meta(video_path)
    process_seconds = min(duration, MAX_PROCESS_SECONDS)
    frame_stride = max(1, round(fps / SAMPLE_FPS))
    max_frames = int(process_seconds * fps)

    if duration > MAX_PROCESS_SECONDS:
        st.info(
            f"This video is {fmt_time(duration)} long -- scanning the first "
            f"{fmt_time(MAX_PROCESS_SECONDS)} to keep the demo responsive."
        )

    cap = cv2.VideoCapture(str(video_path))
    tracker = SignTracker()
    first_seen_sample = {}

    preview_ph = st.empty()
    progress_ph = st.progress(0.0)
    status_ph = st.empty()
    feed_header_ph = st.empty()
    feed_ph = st.empty()

    feed_header_ph.markdown("**Live log**")
    history = []
    frame_idx = 0
    sample_idx = 0
    start = time.time()

    while frame_idx < max_frames:
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

            annotated = frame.copy()
            for (box, class_id, name, category, confidence), track_id in zip(detections, track_ids):
                x, y, w, h = box
                if track_id not in first_seen_sample:
                    first_seen_sample[track_id] = sample_idx
                    if confidence >= CONFIDENCE_THRESHOLD:
                        history.insert(0, {
                            "time": fmt_time(sample_idx / SAMPLE_FPS),
                            "name": name, "category": category, "confidence": confidence,
                        })
                    else:
                        history.insert(0, {
                            "time": fmt_time(sample_idx / SAMPLE_FPS),
                            "name": name, "category": "unsure", "confidence": confidence,
                        })

                color = (0, 165, 255) if confidence < CONFIDENCE_THRESHOLD else (14, 124, 134)
                label = f"{name} {confidence * 100:.0f}%"
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            preview_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            progress_ph.progress(min(1.0, frame_idx / max(max_frames, 1)))
            confident_count = sum(1 for h in history if h["category"] != "unsure")
            status_ph.markdown(
                f"Scanning -- {fmt_time(frame_idx / fps)} / {fmt_time(process_seconds)}  ·  "
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

    elapsed = time.time() - start
    st.session_state.reports = reports
    st.session_state.video_meta = {
        "duration": process_seconds, "fps": fps, "frames_scanned": sample_idx,
        "elapsed": elapsed,
    }
    st.session_state.stage = "done"
    st.rerun()


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
                video_path = wd / uploaded.name
                video_path.write_bytes(uploaded.getbuffer())
            else:
                with st.spinner("Fetching the video..."):
                    video_path = download_from_link(link, wd)
        except Exception as e:
            st.error(f"Couldn't load that video: {e}")
        else:
            st.session_state.video_path = str(video_path)
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

        st.markdown("**All signs found**")
        for r in sorted(reports, key=lambda r: r["first_seen_s"]):
            st.markdown(
                f'<div class="feed-row"><span class="name">{r["name"]}</span>'
                f'{pill_html(r["category"], r["category"])}'
                f'<span class="ts">{r["confidence"] * 100:.0f}%</span>'
                f'<span class="ts">{fmt_time(r["first_seen_s"])}</span></div>',
                unsafe_allow_html=True,
            )

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
