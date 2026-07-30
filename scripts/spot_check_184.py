"""Stage 6.1 lightweight check (no Colab/GPU needed): visually spot-check
class 184 ("Coverage area") against its two biggest confusions from the
stage 5 baseline (183 "Distance to the object", 193 "Limitation of parking
duration"), by sampling images directly from the local dataset zip. Purpose:
tell genuine visual similarity apart from a data-labeling bug, same method
as stage 3.4's spot-check, before deciding whether to spend a full
retraining run on it."""
import io
import random
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

ZIP_PATH = "Traffic sign dataset.zip"
CLASSES = [(184, "Coverage area"), (183, "Distance to the object"), (193, "Limitation of parking duration")]
N_SAMPLES = 6
SEED = 42
OUT_PATH = r"C:\Users\Lenovo\AppData\Local\Temp\claude\C--Users-Lenovo-Desktop-My-project\f055d64e-f7aa-4988-a5b4-b8aee35f2204\scratchpad\class_184_spot_check.png"

random.seed(SEED)

with zipfile.ZipFile(ZIP_PATH) as z:
    all_names = z.namelist()
    fig, axes = plt.subplots(len(CLASSES), N_SAMPLES, figsize=(N_SAMPLES * 2.2, len(CLASSES) * 2.6))
    for row, (class_id, class_name) in enumerate(CLASSES):
        prefix = f"Data/{class_id}/"
        entries = [n for n in all_names if n.startswith(prefix) and not n.endswith('/')]
        print(f"Class {class_id} ({class_name}): {len(entries)} images total in dataset")
        sample = random.sample(entries, min(N_SAMPLES, len(entries)))
        for col, name in enumerate(sample):
            with z.open(name) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(class_name, fontsize=10)
        axes[row, 0].axis('on')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(f"{class_id}\n{class_name}", fontsize=10)

    plt.suptitle("Stage 6.1 spot-check: class 184 vs. its two biggest baseline confusions", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=120)
    print(f"Saved grid to {OUT_PATH}")
