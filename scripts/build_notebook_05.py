"""One-off builder for notebooks/05_transfer_learning.ipynb. Not part of the
pipeline itself -- run once locally to generate the notebook, then discard
or ignore; the notebook file is the actual committed artifact."""
import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""# Transfer learning experiment (ResNet18 backbone)

Roadmap stage 6.1. Self-contained like the previous notebooks -- this
re-downloads the dataset and rebuilds the *exact same* preprocessing
pipeline as `04_baseline_model.ipynb` (same split, same 64x64 letterbox
resize, same normalization, same augmentation). The **only thing that
changes in this experiment is the model itself**: a pretrained ResNet18
instead of the baseline's 3-conv-block CNN trained from scratch. Keeping
everything else identical is what makes the comparison to the stage 5
baseline (87.84% val accuracy) meaningful.

**Why this experiment**: stage 5.4 found that class imbalance was *not*
the bottleneck (correlation between per-class accuracy and training-set
size was only -0.124). What actually explained the weak classes was
systematic confusion between specific, visually similar sign pairs (e.g.
mirrored left/right signs, similar informational plate signs). A
pretrained backbone brings much richer, more discriminative visual
features (learned from ~1.4M ImageNet photos) than a small CNN trained
from scratch on this dataset alone -- the question this experiment
answers is whether that extra discriminative power actually reduces
those specific confusions.

**Before running**: needs a GPU (Runtime > Change runtime type > T4 GPU
for *this* notebook specifically, even if set elsewhere). Training is
checkpointed to Drive every epoch, same as the baseline notebook, so a
disconnect just means re-running the notebook (Runtime > Run all) to
resume from the last completed epoch."""))

cells.append(md("### Setup: download the dataset (from Google Drive, authenticated)"))

cells.append(code("""from google.colab import auth
auth.authenticate_user()

import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

FILE_ID = "1Mi0IleRucNmwnQ4g_ZEWOyBFFv4mO9ba"
ZIP_PATH = "/content/traffic_sign_dataset.zip"

drive_service = build('drive', 'v3')
request = drive_service.files().get_media(fileId=FILE_ID)

with io.FileIO(ZIP_PATH, 'wb') as fh:
    downloader = MediaIoBaseDownload(fh, request, chunksize=200 * 1024 * 1024)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download progress: {int(status.progress() * 100)}%")

print("Download complete.")"""))

cells.append(code("""import os

EXPECTED_SIZE = 14_945_416_206
assert os.path.exists(ZIP_PATH), "Download failed: no file was written at all."
actual_size = os.path.getsize(ZIP_PATH)
print(f"Downloaded file size: {actual_size:,} bytes ({actual_size / 1e9:.2f} GB)")
assert actual_size > 1_000_000_000, "Downloaded file is far too small to be the real dataset."
print("Size check passed.")"""))

cells.append(code("""!rm -rf /content/dataset
!mkdir -p /content/dataset
!unzip -q "$ZIP_PATH" -d /content/dataset
!rm "$ZIP_PATH"

from pathlib import Path
data_dir = Path('/content/dataset/Data')
assert data_dir.is_dir(), "Unzip did not produce /content/dataset/Data as expected."
class_folders = sorted(data_dir.iterdir(), key=lambda p: int(p.name))
print(f"Unzip done. Found {len(class_folders)} class folders under Data/.")"""))

cells.append(md("""### Setup: load the split manifest, class names, and rebuild the preprocessing pipeline

Identical to `04_baseline_model.ipynb`'s setup -- repeated here so this
notebook stays self-contained and the comparison stays apples-to-apples."""))

cells.append(code("""import urllib.request
import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/iambekzodboboev/traffic-sign-recognition/master"
urllib.request.urlretrieve(f"{REPO_RAW}/metadata/split_manifest.csv", "/content/split_manifest.csv")
urllib.request.urlretrieve(f"{REPO_RAW}/metadata/class_names.csv", "/content/class_names.csv")

manifest_df = pd.read_csv("/content/split_manifest.csv", dtype={'class_id': str})
class_names_df = pd.read_csv("/content/class_names.csv")
print(f"Manifest: {len(manifest_df)} rows, class names: {len(class_names_df)} rows")"""))

cells.append(code("""import torchvision.transforms as T
from PIL import Image

TARGET_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LetterboxResize:
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


train_transform = T.Compose([
    LetterboxResize(TARGET_SIZE),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = T.Compose([
    LetterboxResize(TARGET_SIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])"""))

cells.append(code("""import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class TrafficSignDataset(Dataset):
    def __init__(self, manifest_df, split, data_dir, transform):
        self.df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(self.data_dir / row['relative_path']) as img:
            img = self.transform(img)
        return img, int(row['class_id'])


BATCH_SIZE = 64

train_dataset = TrafficSignDataset(manifest_df, 'train', data_dir, train_transform)
val_dataset = TrafficSignDataset(manifest_df, 'val', data_dir, eval_transform)
test_dataset = TrafficSignDataset(manifest_df, 'test', data_dir, eval_transform)

train_class_counts = train_dataset.df['class_id'].value_counts()
class_weights = {cid: 1.0 / count for cid, count in train_class_counts.items()}
sample_weights = train_dataset.df['class_id'].map(class_weights).values
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cpu':
    print("WARNING: no GPU detected. Set Runtime > Change runtime type > T4 GPU "
          "for this notebook, then re-run from the top.")"""))

cells.append(md("""## Step 6.1a -- MLflow tracking setup

Same Drive-backed tracking URI and experiment name as the baseline run,
so both runs show up side by side in MLflow for direct comparison."""))

cells.append(code("""import os
from google.colab import drive
drive.mount('/content/drive')

!pip install -q mlflow

import mlflow
import mlflow.pytorch

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

MLFLOW_DIR = "/content/drive/MyDrive/traffic_sign_recognition/mlruns"
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("traffic-sign-baseline")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")"""))

cells.append(md("""## Step 6.1b -- ResNet18 pretrained backbone

We load `ResNet18` with ImageNet-pretrained weights and replace its final
classification layer (1000 ImageNet classes) with a new one for our 200
sign classes. All layers are fine-tuned (not frozen) rather than only
training the new final layer: our dataset (82k+ training images, sign
icons rather than natural photos) is large enough and different enough
from ImageNet that adapting the deeper layers too should help.

We use a **10x lower learning rate** (1e-4 vs. the baseline's 1e-3): the
pretrained weights already encode useful visual features, so we adjust
them gently instead of overwriting them the way a from-scratch model's
random weights need to be aggressively updated.

**Known trade-off, stated honestly**: we kept the same 64x64 input size
as the baseline so this stays a clean single-variable comparison
(architecture only). ResNet18's pretrained weights were learned on
224x224 images, so its deeper layers see a much smaller feature map here
(64x64 -> 2x2 after its 32x total downsampling) than they were trained
on. If this experiment doesn't help much, a natural follow-up is
re-running it with a larger input size as its own separate experiment."""))

cells.append(code("""import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def build_resnet18(num_classes=200):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return (total_loss / total_count, total_correct / total_count,
            torch.cat(all_preds), torch.cat(all_labels))


sample_model = build_resnet18(num_classes=200)
total_params = sum(p.numel() for p in sample_model.parameters())
print(f"ResNet18 total parameters: {total_params:,} (all trainable, none frozen)")
del sample_model"""))

cells.append(md("""## Step 6.1c -- Sanity check on a tiny 3-class subset

Same quick check as the baseline notebook's step 5.2 -- confirm the
pretrained model can actually learn on this pipeline before committing
to a full run."""))

cells.append(code("""tiny_classes = ['0', '1', '2']
tiny_manifest = manifest_df[manifest_df['class_id'].isin(tiny_classes)]
tiny_train_loader = DataLoader(
    TrafficSignDataset(tiny_manifest, 'train', data_dir, train_transform),
    batch_size=32, shuffle=True)
tiny_val_loader = DataLoader(
    TrafficSignDataset(tiny_manifest, 'val', data_dir, eval_transform),
    batch_size=32, shuffle=False)

tiny_model = build_resnet18(num_classes=200).to(device)
tiny_optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    train_loss, train_acc = train_one_epoch(tiny_model, tiny_train_loader, tiny_optimizer, criterion, device)
    val_loss, val_acc, _, _ = evaluate(tiny_model, tiny_val_loader, criterion, device)
    print(f"Epoch {epoch + 1}: train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

print("\\nRandom guessing among these 3 classes would be ~0.33 accuracy.")
print("A pretrained model should climb well above that within just 1-2 epochs.")

del tiny_model, tiny_optimizer
if device.type == 'cuda':
    torch.cuda.empty_cache()"""))

cells.append(md("""## Step 6.1d -- Train on the full dataset

Same training loop shape as the baseline: 15 epochs (matching the
baseline's epoch count for a fair comparison), checkpointed to Drive
every epoch, resumable after a disconnect."""))

cells.append(code("""NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
RUN_NAME = "resnet18_transfer_15ep"

CHECKPOINT_DIR = "/content/drive/MyDrive/traffic_sign_recognition/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_checkpoint.pt")

model = build_resnet18(num_classes=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Found checkpoint: resuming from epoch {start_epoch}/{NUM_EPOCHS} "
          f"(last val_acc={checkpoint['val_acc']:.3f})")
else:
    print("No checkpoint found, starting fresh from epoch 0.")

with mlflow.start_run(run_name=RUN_NAME) as run:
    mlflow.log_params({
        "model": "ResNet18 (ImageNet-pretrained, fully fine-tuned)",
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "target_size": TARGET_SIZE,
        "optimizer": "Adam",
    })

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")
        mlflow.log_metrics({
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        }, step=epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, CHECKPOINT_PATH)

    example_input = torch.randn(1, 3, TARGET_SIZE, TARGET_SIZE)
    mlflow.pytorch.log_model(
        model, name="model", input_example=example_input, serialization_format="pickle"
    )
    run_id = run.info.run_id

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

print(f"\\nDone. MLflow run id: {run_id}")
print("Baseline (stage 5) validation accuracy was 0.8784 -- compare against that.")"""))

cells.append(md("""## Step 6.1e -- Evaluate on validation set: confusion matrix and per-class accuracy

Same analysis as stage 5.4, so results compare directly."""))

cells.append(code("""import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
print(f"Final validation accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

cm = confusion_matrix(val_labels.numpy(), val_preds.numpy(), labels=list(range(200)))

plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap='viridis')
plt.title('Confusion matrix (200x200 classes) -- ResNet18 transfer learning')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.colorbar()
plt.tight_layout()
plt.show()"""))

cells.append(code("""per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
class_acc_df = pd.DataFrame({
    'class_id': range(200),
    'accuracy': per_class_acc,
    'n_val_images': cm.sum(axis=1),
})
class_acc_df = class_acc_df.merge(class_names_df, on='class_id', how='left')

print("15 worst-performing classes on validation:")
print(class_acc_df.sort_values('accuracy')[['class_id', 'name', 'accuracy', 'n_val_images']]
      .head(15).to_string(index=False))

print("\\n10 best-performing classes on validation:")
print(class_acc_df.sort_values('accuracy', ascending=False)[['class_id', 'name', 'accuracy', 'n_val_images']]
      .head(10).to_string(index=False))"""))

cells.append(md("""## Step 6.1f -- Did this fix the specific confusions found in stage 5?

Directly re-check the confusions flagged in the baseline's writeup,
rather than relying on eyeballing a new top-30 list -- this is the
actual question this experiment is trying to answer."""))

cells.append(code("""# The confusions flagged in stage 5 (PROJECT_STATUS.md), re-checked
# directly against this model's confusion matrix. Baseline counts are
# only filled in for the exact (true, pred) direction that was reported
# there; "n/a" means the reverse direction wasn't separately reported.
flagged_pairs = [
    (44, 45, 92, "Minor road, right -> Minor road, left (mirror confusion)"),
    (45, 44, None, "Minor road, left -> Minor road, right (reverse direction)"),
    (184, 183, 52, "Coverage area -> Distance to the object"),
    (184, 193, 46, "Coverage area -> Limitation of parking duration"),
    (189, 192, 51, "Validity period -> Paid services"),
    (190, 196, 122, "Method of parking -> Dangerous roadside (biggest baseline confusion)"),
    (42, 32, 85, "End of overtaking-by-lorries restriction -> End of all restrictions"),
]

print(f"{'true->pred':<12}{'baseline':>10}{'this model':>12}  description")
for true_c, pred_c, old_count, desc in flagged_pairs:
    new_count = int(cm[true_c, pred_c])
    old_str = str(old_count) if old_count is not None else "n/a"
    print(f"{true_c:>4}->{pred_c:<4}{old_str:>10}{new_count:>12}  {desc}")

print(f"\\nOverall validation accuracy: this model={val_acc:.4f}  baseline=0.8784")"""))

cells.append(md("""### Result

Not yet run -- waiting on the user to execute this in Colab and share
the resulting numbers before deciding stage 6.2 (compare experiments,
pick the best model)."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/05_transfer_learning.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Wrote notebooks/05_transfer_learning.ipynb with", len(cells), "cells")
