"""Stage 7.2 -- local inference: image in, predicted sign name + confidence
out. Runs entirely on CPU using the trained stage 6 model (models/
resnet18_transfer_15ep.pt); no GPU, dataset, or network access needed.

Usage:
    python scripts/predict_sign.py path/to/photo.jpg
"""
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_transfer_15ep.pt"
CLASS_NAMES_PATH = PROJECT_ROOT / "metadata" / "class_names.csv"

TARGET_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 200


class LetterboxResize:
    """Same padding+resize used for training/eval -- must match exactly."""

    def __init__(self, size, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        img = img.convert('RGB')
        w, h = img.size
        max_side = max(w, h)
        canvas = Image.new('RGB', (max_side, max_side), self.fill)
        canvas.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return canvas.resize((self.size, self.size), Image.LANCZOS)


eval_transform = T.Compose([
    LetterboxResize(TARGET_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(image_path, model, class_names_df, top_k=3):
    with Image.open(image_path) as img:
        tensor = eval_transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_ids = torch.topk(probs, top_k)
    results = []
    for prob, class_id in zip(top_probs.tolist(), top_ids.tolist()):
        name_row = class_names_df.loc[class_names_df['class_id'] == class_id, 'name']
        name = name_row.values[0] if len(name_row) else f"class {class_id}"
        results.append((class_id, name, prob))
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict_sign.py path/to/photo.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"No such file: {image_path}")
        sys.exit(1)

    class_names_df = pd.read_csv(CLASS_NAMES_PATH)
    model = load_model()
    results = predict(image_path, model, class_names_df)

    print(f"\nPredictions for {image_path.name}:")
    for rank, (class_id, name, prob) in enumerate(results, start=1):
        print(f"  {rank}. {name} (class {class_id}) -- {prob * 100:.2f}% confidence")


if __name__ == "__main__":
    main()
