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

Roadmap stage 6.1. Runs on **Kaggle Notebooks** rather than Colab --
Colab's free-tier GPU quota was exhausted (from earlier long training
attempts, before mixed precision was added), and this exact dataset is
natively hosted on Kaggle already, with its own separate GPU quota. Still
self-contained like the previous notebooks: rebuilds the *exact same*
preprocessing pipeline as `04_baseline_model.ipynb` (same split, same
64x64 letterbox resize, same normalization, same augmentation). The
**only thing that changes in this experiment is the model itself**: a
pretrained ResNet18 instead of the baseline's 3-conv-block CNN trained
from scratch. Keeping everything else identical is what makes the
comparison to the stage 5 baseline (87.84% val accuracy) meaningful.

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

**Before running, three one-time settings on Kaggle**:
1. **Add the dataset**: click **+ Add Input** (top right) and search for
   `mikhailkosov/traffic-signs-in-post-soviet-states-200-classes` -- this
   mounts it read-only under `/kaggle/input/`, no download step needed
   (unlike Colab, which had to pull it from Drive every session).
2. **Turn on internet**: right sidebar **Settings > Internet > On**
   (needed to `pip install mlflow` and pull `metadata/*.csv` from
   GitHub).
3. **Turn on a GPU**: right sidebar **Settings > Accelerator > GPU T4 x2**
   (or whichever GPU Kaggle offers).

**For the real training run**, use **Save Version > Save & Run All
(Commit)** rather than running cells interactively. This executes the
whole notebook on Kaggle's servers in the background, so you can close
the tab -- no risk of losing progress to a ~90-minute idle disconnect
like Colab. The trained model and printed results end up attached to
that version once it finishes; check the **Logs** tab to follow
progress while it runs."""))

cells.append(md("### Setup: locate the dataset (mounted read-only via Kaggle's \"+ Add Input\")"))

cells.append(code("""from pathlib import Path

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
candidates = [p for p in KAGGLE_INPUT_ROOT.rglob("Data") if p.is_dir()]
assert candidates, (
    "Could not find a 'Data' folder under /kaggle/input -- make sure you've "
    "added the dataset via '+ Add Input' (top right) and searched for "
    "'mikhailkosov/traffic-signs-in-post-soviet-states-200-classes'."
)
data_dir = candidates[0]
class_folders = sorted(data_dir.iterdir(), key=lambda p: int(p.name))
print(f"Found dataset at {data_dir}. {len(class_folders)} class folders under Data/.")"""))

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
    print("WARNING: no GPU detected. Set right sidebar Settings > Accelerator > "
          "GPU T4 x2 for this notebook, then re-run from the top.")"""))

cells.append(md("""## Step 6.1a -- MLflow tracking setup

Tracking data is stored under `/kaggle/working/mlruns` -- Kaggle's
writable output directory, which persists for the current session and
becomes part of this notebook version's Output once committed. (Colab
used a Drive-backed path for the same purpose; Kaggle's equivalent
persistent, writable location is `/kaggle/working`.) Same experiment
name as the baseline run, for direct comparison."""))

cells.append(code("""import os

!pip install -q mlflow

import mlflow
import mlflow.pytorch

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

MLFLOW_DIR = "/kaggle/working/mlruns"
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
re-running it with a larger input size as its own separate experiment.

**Speed note**: an earlier version of this notebook (same data pipeline,
same 15 epochs) took 2-3 hours in Colab, vs. the baseline's 15-50
minutes. The data pipeline is identical between the two notebooks, so
that slowdown isn't from data loading -- it's ResNet18 (11.7M params, 18
conv layers) doing far more GPU math per image than the baseline's tiny
3-conv-block CNN, running in full FP32 and not using the GPU's tensor
cores. We use **automatic mixed precision (AMP)** below to fix that: it
runs most of the forward pass in float16 (Kaggle's T4s have tensor cores
built for this too) while keeping a float32 copy of weights for stable
updates, via `torch.amp.autocast` + `GradScaler`. This changes only *how
fast* the same computation runs, not what's being compared -- results
should be numerically the same as full precision within normal training
noise."""))

cells.append(code("""import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def build_resnet18(num_classes=200):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    use_amp = device.type == 'cuda'
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_preds, all_labels = [], []
    use_amp = device.type == 'cuda'
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
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
tiny_scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    train_loss, train_acc = train_one_epoch(tiny_model, tiny_train_loader, tiny_optimizer, criterion, device, tiny_scaler)
    val_loss, val_acc, _, _ = evaluate(tiny_model, tiny_val_loader, criterion, device)
    print(f"Epoch {epoch + 1}: train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

print("\\nRandom guessing among these 3 classes would be ~0.33 accuracy.")
print("A pretrained model should climb well above that within just 1-2 epochs.")

del tiny_model, tiny_optimizer, tiny_scaler
if device.type == 'cuda':
    torch.cuda.empty_cache()"""))

cells.append(md("""## Step 6.1d -- Train on the full dataset (only if not already done)

**This is the fix for "why do I have to retrain every time I reopen the
notebook."** Before training anything, this checks MLflow for an
already-**completed** run named `resnet18_transfer_15ep`. If one exists
(you already ran this once, successfully, in an earlier session), it
just loads that trained model directly -- no training, no waiting.
Training (15 epochs, matching the baseline's count for a fair
comparison; checkpointed to `/kaggle/working` every epoch including the
AMP scaler's state, resumable if the session is interrupted mid-run)
only happens the *first* time this succeeds, or if you deliberately
delete the MLflow run to force a fresh one.

Note: the dataset is mounted instantly via Kaggle's Input feature (no
download/unzip step at all, unlike Colab), so the only real time cost
left is the training itself. Using **Save Version > Save & Run All
(Commit)** for this run means it executes server-side and isn't tied to
keeping the browser tab open."""))

cells.append(code("""NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
RUN_NAME = "resnet18_transfer_15ep"

from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("traffic-sign-baseline")
existing_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
    order_by=["attributes.start_time DESC"],
)
finished_run = next((r for r in existing_runs if r.info.status == "FINISHED"), None)

if finished_run is not None:
    run_id = finished_run.info.run_id
    print(f"Found an already-completed run '{RUN_NAME}' (run_id={run_id}) -- loading it instead of training again.")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
    logged_val_acc = finished_run.data.metrics.get("val_acc")
    if logged_val_acc is not None:
        print(f"Its last logged val_acc was {logged_val_acc:.4f}.")
else:
    print(f"No completed '{RUN_NAME}' run found -- training now (this only needs to happen once).")

    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_checkpoint.pt")

    model = build_resnet18(num_classes=200).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            print("Checkpoint predates the AMP scaler (older run) -- continuing with a fresh scaler, safe to do.")
        start_epoch = checkpoint['epoch'] + 1
        print(f"Found a partial checkpoint: resuming from epoch {start_epoch}/{NUM_EPOCHS} "
              f"(last val_acc={checkpoint['val_acc']:.3f})")
    else:
        print("No partial checkpoint either, starting fresh from epoch 0.")

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params({
            "model": "ResNet18 (ImageNet-pretrained, fully fine-tuned)",
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "target_size": TARGET_SIZE,
            "optimizer": "Adam",
            "mixed_precision": device.type == 'cuda',
        })

        for epoch in range(start_epoch, NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
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
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc,
            }, CHECKPOINT_PATH)

        example_input = torch.randn(1, 3, TARGET_SIZE, TARGET_SIZE)
        mlflow.pytorch.log_model(
            model, name="model", input_example=example_input, serialization_format="pickle"
        )
        run_id = run.info.run_id

        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    # Also save the plain state dict directly under /kaggle/working, so the
    # trained weights show up in this notebook version's Output tab and are
    # downloadable/reusable even without going through MLflow.
    torch.save(model.state_dict(), "/kaggle/working/resnet18_transfer_15ep.pt")

print(f"\\nUsing MLflow run id: {run_id}")
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

cells.append(md("""### Step 6.1 result

Confirmed on Kaggle: **validation accuracy 0.9903** (vs. baseline's
0.8784). Every confusion flagged in stage 5 dropped to near-zero: the
mirror-image 44<->45 pair, the 184 "Coverage area" confusions, and the
biggest baseline error (190 "Method of parking" -> 196 "Dangerous
roadside", 122x at baseline) all fell to 0-2 occurrences. This confirms
stage 5's hypothesis directly -- the baseline's weakness really was
insufficient visual discrimination, not class imbalance, and a
pretrained backbone fixes it even at the same 64x64 input size (the
resolution trade-off flagged in step 6.1b turned out not to be a hard
blocker)."""))

cells.append(md("""## Step 6.2 -- Compare experiments and pick the best model

Per-class balance, not just overall accuracy, per the roadmap's 6.2
criteria: macro-averaged F1 (treats all 200 classes equally regardless
of size), per-class accuracy std dev, worst-class accuracy, and the
same accuracy-vs-training-size correlation check as stage 5.4 (to see if
this pretrained model is still insensitive to the stage 3.2 imbalance).
Baseline numbers are from the stage 5 writeup (its raw confusion matrix
isn't available in this Kaggle session, only its documented summary
stats)."""))

cells.append(code("""from sklearn.metrics import precision_recall_fscore_support

# --- This model's (ResNet18) per-class balance metrics ---
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    val_labels.numpy(), val_preds.numpy(), labels=list(range(200)), average='macro', zero_division=0
)

per_class_acc = class_acc_df['accuracy'].values
worst_class_row = class_acc_df.sort_values('accuracy').iloc[0]

train_counts = train_dataset.df['class_id'].astype(int).value_counts()
class_acc_df['n_train_images'] = class_acc_df['class_id'].map(train_counts).fillna(0)
size_correlation = class_acc_df['accuracy'].corr(class_acc_df['n_train_images'])

print("=" * 72)
print("STAGE 6.2 -- MODEL COMPARISON")
print("(baseline numbers are from the stage 5 writeup, not recomputed here --")
print(" baseline's raw confusion matrix isn't available in this Kaggle session)")
print("=" * 72)
print(f"{'metric':<45}{'baseline (CNN)':>13}{'this (ResNet18)':>14}")
print(f"{'Overall validation accuracy':<45}{0.8784:>13.4f}{val_acc:>14.4f}")
print(f"{'Worst single class accuracy':<45}{0.179:>13.4f}{worst_class_row['accuracy']:>14.4f}")
print(f"{'Per-class accuracy std dev':<45}{'n/a':>13}{per_class_acc.std():>14.4f}")
print(f"{'Macro-averaged F1 (all 200 classes equal)':<45}{'n/a':>13}{macro_f1:>14.4f}")
print(f"{'Corr(accuracy, training-set size)':<45}{-0.124:>13.4f}{size_correlation:>14.4f}")
print("=" * 72)

print(f"\\nWorst class for this model: {int(worst_class_row['class_id'])} "
      f"({worst_class_row['name']}) at {worst_class_row['accuracy']:.4f} accuracy "
      f"(baseline's worst was class 184 'Coverage area' at 0.179).")

print("\\nDecision: ResNet18 (this run) wins on every axis measured -- overall "
      "accuracy, worst-class accuracy, and the specific stage-5 confusions "
      "(see step 6.1f) -- so it's the model selected coming out of stage 6.2.")

try:
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("stage6.2_decision", "selected as best model")
        mlflow.log_metrics({
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "worst_class_accuracy": float(worst_class_row['accuracy']),
            "per_class_accuracy_std": float(per_class_acc.std()),
            "train_size_correlation": float(size_correlation),
        })
    print(f"\\nLogged stage 6.2 comparison metrics to MLflow run {run_id}.")
except Exception as e:
    print(f"\\n(Could not log extra metrics to MLflow: {e} -- comparison above is unaffected.)")"""))

cells.append(md("""### Step 6.2 result

Confirmed on Kaggle: ResNet18 wins on every metric. Worst-class accuracy
jumped from 0.179 (baseline's class 184) to 0.8033 (a different class,
54 "Intersection of equivalent roads" -- expected, since fixing the
previously-worst classes surfaces a new relative worst, still far
higher). Macro-F1 (0.9877) sits almost exactly at overall accuracy
(0.9903), meaning the model is strong across all 200 classes equally,
not just the frequent ones. Training-size correlation (0.0381) stayed
near zero, confirming this model is, if anything, even less sensitive to
the stage 3.2 imbalance than the baseline was. **ResNet18 selected as
the final model.**"""))

cells.append(md("""## Step 6.3 -- Final test-set evaluation (run once)

The test set has not been touched until now, specifically so it isn't
implicitly tuned against while comparing experiments in 6.1/6.2. This is
the one number that gets reported as the final result."""))

cells.append(code("""test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
print(f"FINAL TEST SET ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"(For reference: validation accuracy was {val_acc:.4f})")

test_cm = confusion_matrix(test_labels.numpy(), test_preds.numpy(), labels=list(range(200)))
test_per_class_acc = test_cm.diagonal() / test_cm.sum(axis=1).clip(min=1)

test_macro_precision, test_macro_recall, test_macro_f1, _ = precision_recall_fscore_support(
    test_labels.numpy(), test_preds.numpy(), labels=list(range(200)), average='macro', zero_division=0
)

test_class_acc_df = pd.DataFrame({
    'class_id': range(200),
    'accuracy': test_per_class_acc,
    'n_test_images': test_cm.sum(axis=1),
}).merge(class_names_df, on='class_id', how='left')

worst_test_row = test_class_acc_df.sort_values('accuracy').iloc[0]

print(f"\\nTest per-class accuracy std dev: {test_per_class_acc.std():.4f}")
print(f"Test macro-averaged F1: {test_macro_f1:.4f}")
print(f"Worst class on test set: {int(worst_test_row['class_id'])} ({worst_test_row['name']}) "
      f"at {worst_test_row['accuracy']:.4f} accuracy")

print("\\n15 worst-performing classes on TEST set:")
print(test_class_acc_df.sort_values('accuracy')[['class_id', 'name', 'accuracy', 'n_test_images']]
      .head(15).to_string(index=False))

try:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_macro_f1": test_macro_f1,
            "test_per_class_accuracy_std": float(test_per_class_acc.std()),
        })
        mlflow.set_tag("stage6.3_final_test_accuracy", f"{test_acc:.4f}")
    print(f"\\nLogged final test-set metrics to MLflow run {run_id}.")
except Exception as e:
    print(f"\\n(Could not log test metrics to MLflow: {e} -- printed results above are unaffected.)")"""))

cells.append(md("""### Final stage 6 result

Confirmed on Kaggle: **final test-set accuracy 0.9937** (99.37%),
slightly *above* the validation accuracy (0.9903) -- a good sign of no
overfitting to the validation set across the iterative comparison
process. Test macro-F1 0.9937, essentially matching accuracy (consistent
strength across all 200 classes). Worst class on the test set: 170
"Phone" at 0.8846 accuracy -- every one of the 15 worst test classes
still scored above 88%.

**Stage 6 (model selection) is complete.** ResNet18 (ImageNet-pretrained,
fully fine-tuned, 64x64 input, 15 epochs, mixed precision) is the chosen
model, trained and evaluated on Kaggle Notebooks after Colab's free-tier
GPU quota ran out. Next: stage 7 (inference/demo workflow)."""))

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
