"""Stage 9.3 -- crop classification. Wraps the existing trained classifier
(scripts/predict_sign.py) for in-memory OpenCV crops, so video frames don't
need to be written to disk per detection.

Read-only reuse: only imports from scripts/predict_sign.py, never edits it
(and never touches bot.py). The model itself is not retrained or changed.
"""
import re
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image

# Make "scripts.predict_sign" importable regardless of the current working
# directory or how this file is invoked (project root isn't automatically
# on sys.path when running a script that lives in a subfolder).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.predict_sign import CLASS_NAMES_PATH, eval_transform, load_model


def load_classifier():
    """Returns (model, class_names_df) -- call once, reuse for every crop."""
    model = load_model()
    class_names_df = pd.read_csv(CLASS_NAMES_PATH)
    return model, class_names_df


def classify_crop(crop_bgr, model, class_names_df, top_k=1):
    """crop_bgr: an OpenCV-style BGR numpy array (e.g. frame[y:y+h, x:x+w]).
    Returns a list of (class_id, name, category, confidence) tuples."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    tensor = eval_transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_ids = torch.topk(probs, top_k)
    results = []
    for prob, class_id in zip(top_probs.tolist(), top_ids.tolist()):
        row = class_names_df.loc[class_names_df["class_id"] == class_id]
        name = row["name"].values[0] if len(row) else f"class {class_id}"
        category = row["category"].values[0] if len(row) else "Unknown"
        category = re.sub(r"^\d+\s*", "", category)
        results.append((class_id, name, category, prob))
    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python video_detection/classifier.py path/to/frame.jpg")
        sys.exit(1)

    frame = cv2.imread(sys.argv[1])
    model, class_names_df = load_classifier()

    from detector import detect_candidate_boxes

    boxes = detect_candidate_boxes(frame)
    print(f"Found {len(boxes)} candidate box(es):")
    for (x, y, w, h) in boxes:
        crop = frame[y:y + h, x:x + w]
        results = classify_crop(crop, model, class_names_df, top_k=1)
        class_id, name, category, confidence = results[0]
        print(f"  box=({x},{y},{w},{h}) -> {name} [{category}] {confidence * 100:.1f}%")
