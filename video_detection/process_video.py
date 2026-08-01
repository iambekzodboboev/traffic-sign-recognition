"""Stage 9.4/9.6 -- process a whole video: sample frames (not every one),
detect and classify signs in each sampled frame, and write an annotated
output video. Still fully separate from the Telegram bot -- reuses the
trained classifier read-only via classifier.py, never touches bot.py.

Demo usage (run from anywhere, e.g. the project root):
    python video_detection/process_video.py path/to/video.mp4
    python video_detection/process_video.py path/to/video.mp4 --seconds 20

Output: an annotated .mp4 (boxes + labels drawn on each sampled frame)
and a .txt report of deduplicated tracked signs, both written to
video_detection/output/.
"""
import argparse
import time
from pathlib import Path

import cv2

from detector import detect_candidate_boxes
from classifier import classify_crop, load_classifier
from tracker import SignTracker

SAMPLE_FPS = 4
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
    tracker = SignTracker()

    frame_idx = 0
    sample_idx = 0
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
            detections = []
            for (x, y, w, h) in boxes:
                x, y, w, h = _clip_box(x, y, w, h, width, height)
                if w <= 0 or h <= 0:
                    continue
                crop = frame[y:y + h, x:x + w]
                results = classify_crop(crop, model, class_names_df, top_k=1)
                class_id, name, category, confidence = results[0]
                detections.append(((x, y, w, h), class_id, name, category, confidence))

                if confidence < CONFIDENCE_THRESHOLD:
                    low_conf_detections += 1
                else:
                    confident_detections += 1

            track_ids = tracker.update(sample_idx, detections)

            for (box, class_id, name, category, confidence), track_id in zip(detections, track_ids):
                x, y, w, h = box
                if confidence < CONFIDENCE_THRESHOLD:
                    color = (0, 165, 255)  # orange: shown, but flagged as unreliable
                    label = f"#{track_id} ? unsure ({confidence * 100:.0f}%)"
                else:
                    color = (0, 255, 0)
                    label = f"#{track_id} {name} [{category}] {confidence * 100:.0f}%"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                label_y = max(0, y - th - 10)
                cv2.rectangle(frame, (x, label_y), (x + tw + 6, label_y + th + 10), color, -1)
                cv2.putText(frame, label, (x + 3, label_y + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            writer.write(frame)
            processed += 1
            sample_idx += 1

        frame_idx += 1

    cap.release()
    writer.release()
    reports = tracker.finish()
    elapsed = time.time() - start_time

    print(f"\nProcessed {processed} sampled frames ({frame_idx} source frames) in {elapsed:.1f}s "
          f"({elapsed / max(processed, 1):.2f}s/sampled frame)")
    print(f"Confident detections drawn (per-frame, before tracking): {confident_detections}")
    print(f"Low-confidence ('unsure') detections drawn: {low_conf_detections}")
    print(f"Saved annotated video to {output_path}")

    print(f"\n=== Tracked signs (deduplicated across frames): {len(reports)} ===")
    for r in sorted(reports, key=lambda r: r["track_id"]):
        print(f"  #{r['track_id']}: {r['name']} [{r['category']}] "
              f"best confidence {r['confidence'] * 100:.0f}%, "
              f"seen in {r['frames_seen']} frame(s), "
              f"{r['frames_confident']} of them confident")

    report_path = output_path.with_suffix(".txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Tracked signs for {input_path.name}: {len(reports)}\n\n")
        for r in sorted(reports, key=lambda r: r["track_id"]):
            f.write(f"#{r['track_id']}: {r['name']} [{r['category']}] "
                    f"best confidence {r['confidence'] * 100:.0f}%, "
                    f"seen in {r['frames_seen']} frame(s), "
                    f"{r['frames_confident']} of them confident\n")
    print(f"Saved tracked-sign report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and classify traffic signs in a driving video, "
                     "and write an annotated copy plus a text report.")
    parser.add_argument("video", type=Path, help="path to an input video file (e.g. .mp4)")
    parser.add_argument("--seconds", type=float, default=None,
                         help="only process the first N seconds (default: whole video)")
    args = parser.parse_args()

    if not args.video.exists():
        parser.error(f"video file not found: {args.video}")

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    suffix = f"_first{int(args.seconds)}s" if args.seconds else ""
    output_path = output_dir / f"{args.video.stem}_annotated{suffix}.mp4"

    process_video(args.video, output_path, args.seconds)


if __name__ == "__main__":
    main()
