"""Stage 9.4 -- process a whole video: sample frames (not every one), detect
and classify signs in each sampled frame, and write an annotated output
video. Still fully separate from the Telegram bot -- reuses the trained
classifier read-only via classifier.py, never touches bot.py.

Usage:
    python video_detection/process_video.py path/to/video.mp4 [max_seconds]
"""
import sys
import time
from pathlib import Path

import cv2

from detector import detect_candidate_boxes
from classifier import classify_crop, load_classifier

SAMPLE_FPS = 2
CONFIDENCE_THRESHOLD = 0.60  # same idea as the bot's threshold, applied independently here


def _clip_box(x, y, w, h, frame_w, frame_h):
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    return x, y, w, h


def process_video(input_path, output_path, max_seconds=None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / src_fps if src_fps else 0
    print(f"Input: {width}x{height}, {src_fps:.1f} fps, {frame_count} frames, {duration:.1f}s")

    frame_stride = max(1, round(src_fps / SAMPLE_FPS))
    print(f"Sampling every {frame_stride} frames (~{SAMPLE_FPS} fps)")

    max_frames = int(max_seconds * src_fps) if max_seconds else frame_count

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, SAMPLE_FPS, (width, height))

    model, class_names_df = load_classifier()

    frame_idx = 0
    processed = 0
    confident_detections = 0
    low_conf_detections = 0
    start_time = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            boxes = detect_candidate_boxes(frame)
            for (x, y, w, h) in boxes:
                x, y, w, h = _clip_box(x, y, w, h, width, height)
                if w <= 0 or h <= 0:
                    continue
                crop = frame[y:y + h, x:x + w]

                results = classify_crop(crop, model, class_names_df, top_k=1)
                class_id, name, category, confidence = results[0]

                if confidence < CONFIDENCE_THRESHOLD:
                    color = (0, 165, 255)  # orange: shown, but flagged as unreliable
                    label = f"? unsure ({confidence * 100:.0f}%)"
                    low_conf_detections += 1
                else:
                    color = (0, 255, 0)
                    label = f"{name} [{category}] {confidence * 100:.0f}%"
                    confident_detections += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                label_y = max(0, y - th - 10)
                cv2.rectangle(frame, (x, label_y), (x + tw + 6, label_y + th + 10), color, -1)
                cv2.putText(frame, label, (x + 3, label_y + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            writer.write(frame)
            processed += 1

        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - start_time

    print(f"\nProcessed {processed} sampled frames ({frame_idx} source frames) in {elapsed:.1f}s "
          f"({elapsed / max(processed, 1):.2f}s/sampled frame)")
    print(f"Confident detections drawn: {confident_detections}")
    print(f"Low-confidence ('unsure') detections drawn: {low_conf_detections}")
    print(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python video_detection/process_video.py path/to/video.mp4 [max_seconds]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    max_seconds = float(sys.argv[2]) if len(sys.argv) == 3 else None

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    suffix = f"_first{int(max_seconds)}s" if max_seconds else ""
    output_path = output_dir / f"{input_path.stem}_annotated{suffix}.mp4"

    process_video(input_path, output_path, max_seconds)
