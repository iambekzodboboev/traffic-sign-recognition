"""Stage 9.1/9.3 -- classical computer-vision sign detector. Finds candidate
sign regions in a frame using color (the standardized red/blue used on real
signs) plus shape (triangle / quad / round), rather than a downloaded
pretrained detector model (see video_detection/README.md for why).

This is intentionally separate from the Telegram bot -- it does not import
or modify bot.py, only the shared classifier in scripts/predict_sign.py.
"""
import cv2
import numpy as np

# This camera is mounted in a fixed position for the whole test video, so
# the dashboard/steering wheel reliably occupies the lower part of every
# frame. Restricting the search to the upper region removes most
# false positives (skin tone, dashboard trim) for free. This fraction is a
# heuristic tied to this specific camera mount -- a different video/mount
# would need it recalibrated.
SEARCH_REGION_TOP_FRACTION = 0.58

MIN_AREA_FRACTION = 0.0007  # of full-frame area; filters out tiny color noise


def _shape_score(contour):
    """Returns True if a contour's shape roughly matches a real sign shape
    (triangle, quad/diamond, or round -- covers circles and octagons).

    Uses the convex hull rather than the raw contour: a sign's color mask
    is often a ring with holes (white text/pictograms inside a red border
    aren't red), which makes the raw contour jagged and its shape unreadable
    -- the hull "fills in" that jaggedness back into the sign's true outline.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter == 0:
        return False
    approx = cv2.approxPolyDP(hull, 0.04 * perimeter, True)
    n_vertices = len(approx)
    if n_vertices in (3, 4):
        return True
    hull_area = cv2.contourArea(hull)
    circularity = 4 * np.pi * hull_area / (perimeter ** 2)
    return circularity > 0.75


def detect_candidate_boxes(frame):
    """Returns a list of (x, y, w, h) candidate sign boxes in full-frame
    pixel coordinates."""
    h, w = frame.shape[:2]
    search_h = int(h * SEARCH_REGION_TOP_FRACTION)
    region = frame[0:search_h, :]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    blue_mask = cv2.inRange(hsv, (95, 80, 50), (130, 255, 255))
    combined = cv2.bitwise_or(red_mask, blue_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # A sign's colored border can be thin, especially when small/distant --
    # dilating thickens it and closes small gaps so its contour area/shape
    # come out reliably instead of underestimating a thin ring as noise.
    combined = cv2.dilate(combined, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * MIN_AREA_FRACTION
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh
        if not (0.4 < aspect < 2.4):
            continue
        if not _shape_score(c):
            continue
        boxes.append((x, y, bw, bh))  # y already in full-frame coords (region starts at row 0)
    return boxes


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python video_detection/detector.py path/to/frame.jpg")
        sys.exit(1)

    frame_path = Path(sys.argv[1])
    frame = cv2.imread(str(frame_path))
    boxes = detect_candidate_boxes(frame)
    print(f"Found {len(boxes)} candidate box(es):")
    for (x, y, bw, bh) in boxes:
        print(f"  x={x} y={y} w={bw} h={bh}")

    out = frame.copy()
    search_h = int(frame.shape[0] * SEARCH_REGION_TOP_FRACTION)
    cv2.line(out, (0, search_h), (frame.shape[1], search_h), (0, 200, 255), 2)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
    out_path = frame_path.parent / f"{frame_path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"Saved annotated frame to {out_path}")
