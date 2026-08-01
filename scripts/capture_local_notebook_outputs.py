"""Stage 8.1 -- capture real execution outputs for notebooks/02_data_audit_eda.ipynb
and notebooks/03_preprocessing.ipynb by running their analysis logic locally
against an already-extracted copy of the dataset, instead of Colab.

Why this exists: those notebooks are committed with Colab-specific setup
cells (Google Drive auth + download) so they can run standalone in a fresh
Colab session (see the notebooks' own "self-contained" note). That's the
right way to make them reproducible for someone starting from scratch, but
it also means the committed .ipynb files carry no saved outputs -- anyone
browsing them on GitHub sees code only, not results. This script re-runs
the same analysis (same dataset, same logic, only the file-access path
differs -- a local extracted folder instead of a Colab-downloaded one) to
capture real outputs, which attach_local_notebook_outputs.py then writes
back into the actual committed notebooks. The Colab download cells
themselves are left with no output, since this script does not run them.

One-time local use: extract the dataset zip first (e.g.
`unzip -q "Traffic sign dataset.zip" -d data/local_dataset_extract`,
data/ is gitignored), then:
    python scripts/capture_local_notebook_outputs.py
Writes a pickle of captured outputs to data/local_notebook_outputs.pkl
for attach_local_notebook_outputs.py to consume.
"""
import io
import pickle
import random
import re
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "local_dataset_extract" / "Data"
CLASSES_DIR = PROJECT_ROOT / "data" / "local_dataset_extract" / "Classes"
OUT_PICKLE = PROJECT_ROOT / "data" / "local_notebook_outputs.pkl"

captured = {"nb02": {}, "nb03": {}}


def run_cell(notebook, cell_index, fn):
    """Runs fn(), capturing stdout text and any matplotlib figures it
    creates, keyed by (notebook, cell_index) to match the real notebook."""
    buf = io.StringIO()
    images = []
    with redirect_stdout(buf):
        result = fn()
    for num in plt.get_fignums():
        fig = plt.figure(num)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
        images.append(img_buf.getvalue())
        plt.close(fig)
    captured[notebook][cell_index] = {
        "stdout": buf.getvalue(),
        "images": images,
        "repr": None if result is None else repr(result),
    }
    print(f"[{notebook} cell {cell_index}] captured "
          f"({len(buf.getvalue())} chars stdout, {len(images)} image(s))")


# ---------------------------------------------------------------------------
# Notebook 02: data audit / EDA
# ---------------------------------------------------------------------------

def nb02_cell6():
    global class_folders
    class_folders = sorted(DATA_DIR.iterdir(), key=lambda p: int(p.name))
    found = set(p.name for p in class_folders)
    expected = set(str(i) for i in range(200))
    print(f"Found {len(class_folders)} class folders")
    print("Missing classes:", sorted(expected - found, key=int) or "None")
    print("Unexpected extra folders:", sorted(found - expected) or "None")


def nb02_cell8():
    corrupt_files = []
    total_checked = 0
    for class_folder in tqdm(class_folders, desc="Checking classes"):
        for img_path in class_folder.iterdir():
            total_checked += 1
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                corrupt_files.append((str(img_path), str(e)))
    print(f"\nChecked {total_checked} images")
    print(f"Corrupt/unreadable: {len(corrupt_files)}")
    for path, err in corrupt_files[:20]:
        print(" ", path, "->", err)


def nb02_cell11():
    global class_counts, counts
    class_counts = {cf.name: len(list(cf.iterdir())) for cf in class_folders}
    counts = np.array(list(class_counts.values()))
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    print(f"Number of classes: {len(counts)}")
    print(f"Total images: {counts.sum()}")
    print(f"Min images in a class: {counts.min()} (class {min_class})")
    print(f"Max images in a class: {counts.max()} (class {max_class})")
    print(f"Mean: {counts.mean():.1f}, Median: {np.median(counts):.1f}")
    print(f"Imbalance ratio (max/min): {counts.max() / counts.min():.2f}x")


def nb02_cell12():
    sorted_by_id = sorted(class_counts.items(), key=lambda x: int(x[0]))
    plt.figure(figsize=(20, 5))
    plt.bar([x[0] for x in sorted_by_id], [x[1] for x in sorted_by_id])
    plt.xlabel('Class ID')
    plt.ylabel('Number of images')
    plt.title('Images per class, in class-ID order')
    plt.xticks([])
    plt.tight_layout()

    sorted_by_count = sorted(class_counts.items(), key=lambda x: x[1])
    plt.figure(figsize=(20, 5))
    plt.bar(range(len(sorted_by_count)), [x[1] for x in sorted_by_count], color='steelblue')
    plt.axhline(counts.mean(), color='red', linestyle='--', label=f'Mean ({counts.mean():.0f})')
    plt.xlabel('Classes, sorted from fewest to most images')
    plt.ylabel('Number of images')
    plt.title('Class distribution sorted (shows the imbalance shape)')
    plt.legend()
    plt.tight_layout()


def nb02_cell15():
    global props_df
    records = []
    for class_folder in tqdm(class_folders, desc="Reading image properties"):
        class_id = class_folder.name
        for img_path in class_folder.iterdir():
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode
            file_size = img_path.stat().st_size
            records.append({
                'class_id': class_id, 'width': width, 'height': height,
                'mode': mode, 'file_size_bytes': file_size,
            })
    props_df = pd.DataFrame(records)
    props_df['aspect_ratio'] = props_df['width'] / props_df['height']
    print(f"Total images analyzed: {len(props_df)}")
    return props_df.head()


def nb02_cell16():
    print("--- Width (px) ---")
    print(props_df['width'].describe())
    print("\n--- Height (px) ---")
    print(props_df['height'].describe())
    print("\n--- Aspect ratio (width / height) ---")
    print(props_df['aspect_ratio'].describe())
    print("\n--- File size (bytes) ---")
    print(props_df['file_size_bytes'].describe())
    print("\n--- Color mode counts ---")
    print(props_df['mode'].value_counts())


def nb02_cell17():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(props_df['width'], bins=50, color='steelblue')
    axes[0, 0].set_title('Width distribution')
    axes[0, 0].set_xlabel('Width (px)')
    axes[0, 1].hist(props_df['height'], bins=50, color='steelblue')
    axes[0, 1].set_title('Height distribution')
    axes[0, 1].set_xlabel('Height (px)')
    axes[1, 0].hist(props_df['aspect_ratio'], bins=50, color='steelblue')
    axes[1, 0].set_title('Aspect ratio (width / height) distribution')
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
    axes[1, 0].legend()
    mode_counts = props_df['mode'].value_counts()
    axes[1, 1].bar(mode_counts.index.astype(str), mode_counts.values, color='steelblue')
    axes[1, 1].set_title('Color mode counts (log scale)')
    axes[1, 1].set_yscale('log')
    plt.tight_layout()


def nb02_cell20():
    random.seed(42)
    n_sample_classes = 8
    images_per_class = 4
    sample_class_ids = random.sample([cf.name for cf in class_folders], n_sample_classes)
    fig, axes = plt.subplots(n_sample_classes, images_per_class,
                              figsize=(images_per_class * 2.2, n_sample_classes * 2.2))
    for row, class_id in enumerate(sample_class_ids):
        class_path = DATA_DIR / class_id
        img_files = list(class_path.iterdir())
        chosen = random.sample(img_files, min(images_per_class, len(img_files)))
        for col in range(images_per_class):
            ax = axes[row, col]
            if col < len(chosen):
                with Image.open(chosen[col]) as img:
                    ax.imshow(img.convert('RGB'))
                ax.set_title(f"class {class_id}", fontsize=9)
            ax.axis('off')
    plt.suptitle('Random sample: 4 images from each of 8 random classes', y=1.01)
    plt.tight_layout()


def nb02_cell23():
    global class_names_df
    class_entries = {}
    for category_dir in CLASSES_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        for subfolder in category_dir.iterdir():
            if not subfolder.is_dir():
                continue
            m = re.match(r'^(\d+)\s+(.*)$', subfolder.name)
            if not m:
                continue
            cid = int(m.group(1))
            name = m.group(2).strip()
            if cid == 81:
                if cid in class_entries:
                    class_entries[cid] = {
                        'name': 'Height limit (3.5 or 4.5, ambiguous in source data)',
                        'category': category_dir.name,
                    }
                else:
                    class_entries[cid] = {'name': name, 'category': category_dir.name}
            else:
                class_entries[cid] = {'name': name, 'category': category_dir.name}
    print(f"Found {len(class_entries)} numbered class name entries")
    missing = set(range(200)) - set(class_entries.keys())
    print("Missing IDs (0-199) with no name found:", sorted(missing) or "None")
    class_names_df = pd.DataFrame([
        {'class_id': cid, **class_entries[cid]} for cid in range(200)
    ]).sort_values('class_id').reset_index(drop=True)
    return class_names_df.head(10)


def nb02_cell24():
    data_class_ids = set(int(cf.name) for cf in class_folders)
    mapped_ids = set(class_names_df['class_id'])
    print("Classes with images but no name mapping:", sorted(data_class_ids - mapped_ids) or "None")
    print("Classes with a name mapping but no image folder:", sorted(mapped_ids - data_class_ids) or "None")
    for cid in [0, 6, 14, 28, 81, 163, 189]:
        row = class_names_df[class_names_df['class_id'] == cid].iloc[0]
        print(f"class {cid}: {row['name']} ({row['category']})")


def nb02_cell27():
    global groups_a, track_sizes
    pat_a = re.compile(r'^(\d{5})_(\d{5})_(\d{5})\.\w+$', re.IGNORECASE)
    pat_b = re.compile(r'^(\d{3})_(\d{4})\.\w+$', re.IGNORECASE)
    pat_c = re.compile(r'^(\d{5})\.\w+$', re.IGNORECASE)

    pattern_counts = Counter()
    groups_a = defaultdict(list)
    mismatches_a = 0
    other_examples = []

    for class_folder in tqdm(class_folders, desc="Classifying filenames"):
        class_id = class_folder.name
        for img_path in class_folder.iterdir():
            name = img_path.name
            ma = pat_a.match(name)
            if ma:
                pattern_counts['A: prefix_track_frame'] += 1
                prefix, track, frame = ma.groups()
                groups_a[(class_id, track)].append((int(frame), img_path))
                if int(prefix) != int(class_id):
                    mismatches_a += 1
                continue
            mb = pat_b.match(name)
            if mb:
                pattern_counts['B: xxx_xxxx'] += 1
                continue
            mc = pat_c.match(name)
            if mc:
                pattern_counts['C: bare number (no track info)'] += 1
                continue
            pattern_counts['D: other/unrecognized'] += 1
            if len(other_examples) < 8:
                other_examples.append(name)

    total = sum(pattern_counts.values())
    print("Filename pattern breakdown:")
    for k, v in pattern_counts.items():
        print(f"  {k}: {v} ({v / total * 100:.1f}%)")
    print(f"\nPattern A: class-folder mismatches with embedded prefix: {mismatches_a} "
          f"/ {pattern_counts['A: prefix_track_frame']}")
    print("\nExamples of pattern D (other/unrecognized):")
    for e in other_examples:
        print(" ", e)

    track_sizes = {k: len(v) for k, v in groups_a.items()}


def nb02_cell28():
    sizes = sorted(track_sizes.values())
    n = len(sizes)
    print(f"Pattern-A (class, track) groups: {n}")
    print(f"Images per group -- min: {sizes[0]}, max: {sizes[-1]}, "
          f"mean: {sum(sizes)/n:.2f}, median: {sizes[n // 2]}")
    single = sum(1 for s in sizes if s == 1)
    five_plus = sum(1 for s in sizes if s >= 5)
    print(f"Groups with only 1 image: {single} ({single / n * 100:.1f}%)")
    print(f"Groups with 5+ images: {five_plus} ({five_plus / n * 100:.1f}%)")

    plt.figure(figsize=(10, 4))
    plt.hist(sizes, bins=range(1, sizes[-1] + 2), color='steelblue')
    plt.xlabel('Images in a (class, track) group')
    plt.ylabel('Number of groups')
    plt.title('Pattern A: distribution of group sizes')
    plt.tight_layout()


def nb02_cell29():
    candidate_groups = [k for k, v in track_sizes.items() if v >= 6]
    random.seed(0)
    example_groups = random.sample(candidate_groups, min(3, len(candidate_groups)))
    for ex_class, ex_track in example_groups:
        frames = sorted(groups_a[(ex_class, ex_track)], key=lambda x: x[0])
        fig, axes = plt.subplots(1, len(frames), figsize=(len(frames) * 1.8, 2.2))
        if len(frames) == 1:
            axes = [axes]
        for ax, (frame_id, img_path) in zip(axes, frames):
            with Image.open(img_path) as img:
                ax.imshow(img.convert('RGB'))
            ax.set_title(f"frame {frame_id}", fontsize=8)
            ax.axis('off')
        plt.suptitle(f"Class {ex_class}, track {ex_track} -- {len(frames)} frames in sequence")
        plt.tight_layout()


# ---------------------------------------------------------------------------
# Notebook 03: preprocessing
# ---------------------------------------------------------------------------

def nb03_cell7():
    global manifest_df, class_names_df_03
    manifest_df = pd.read_csv(PROJECT_ROOT / "metadata" / "split_manifest.csv", dtype={'class_id': str})
    class_names_df_03 = pd.read_csv(PROJECT_ROOT / "metadata" / "class_names.csv")
    print(f"Manifest: {len(manifest_df)} rows")
    print(f"Class names: {len(class_names_df_03)} rows")
    return manifest_df.head()


def nb03_cell8():
    print("Split counts (by image):")
    print(manifest_df['split'].value_counts())
    print("\nSplit counts (by group):")
    print(manifest_df.groupby('group_id')['split'].first().value_counts())

    total_dataset_images = sum(len(list(cf.iterdir())) for cf in class_folders)
    print(f"\nManifest rows: {len(manifest_df)}, dataset images: {total_dataset_images}, "
          f"match: {len(manifest_df) == total_dataset_images}")

    missing = [p for p in tqdm(manifest_df['relative_path'], desc="Checking manifest paths")
               if not (DATA_DIR / p).exists()]
    print(f"Manifest paths missing from disk: {len(missing)}")

    bad_groups = manifest_df.groupby('group_id')['split'].nunique()
    bad_groups = bad_groups[bad_groups > 1]
    print(f"Groups spanning multiple splits (should be 0): {len(bad_groups)}")


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


def nb03_cell10():
    global train_transform, eval_transform
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
    ])
    print("Transforms defined: train has augmentation, val/test do not.")


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


def nb03_cell12():
    global train_loader, class_names_df_03
    train_dataset = TrafficSignDataset(manifest_df, 'train', DATA_DIR, train_transform)
    val_dataset = TrafficSignDataset(manifest_df, 'val', DATA_DIR, eval_transform)
    test_dataset = TrafficSignDataset(manifest_df, 'test', DATA_DIR, eval_transform)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_class_counts = train_dataset.df['class_id'].value_counts()
    class_weights = {cid: 1.0 / count for cid, count in train_class_counts.items()}
    sample_weights = train_dataset.df['class_id'].map(class_weights).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("DataLoaders ready.")


def nb03_cell13():
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, labels shape: {labels.shape}")

    def denormalize(tensor):
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)

    n_show = 8
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2.5))
    for i in range(n_show):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        label_id = labels[i].item()
        name_row = class_names_df_03[class_names_df_03['class_id'] == label_id]
        name = name_row.iloc[0]['name'] if len(name_row) else str(label_id)
        axes[i].set_title(f"{label_id}: {name}", fontsize=8)
        axes[i].axis('off')
    plt.suptitle('Sanity check: one training batch (denormalized, with augmentation)')
    plt.tight_layout()


if __name__ == "__main__":
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"{DATA_DIR} not found. Extract the dataset zip there first, e.g.\n"
            f'  unzip -q "Traffic sign dataset.zip" -d data/local_dataset_extract'
        )

    run_cell("nb02", 6, nb02_cell6)
    run_cell("nb02", 8, nb02_cell8)
    run_cell("nb02", 11, nb02_cell11)
    run_cell("nb02", 12, nb02_cell12)
    run_cell("nb02", 15, nb02_cell15)
    run_cell("nb02", 16, nb02_cell16)
    run_cell("nb02", 17, nb02_cell17)
    run_cell("nb02", 20, nb02_cell20)
    run_cell("nb02", 23, nb02_cell23)
    run_cell("nb02", 24, nb02_cell24)
    run_cell("nb02", 27, nb02_cell27)
    run_cell("nb02", 28, nb02_cell28)
    run_cell("nb02", 29, nb02_cell29)

    run_cell("nb03", 7, nb03_cell7)
    run_cell("nb03", 8, nb03_cell8)
    run_cell("nb03", 10, nb03_cell10)
    run_cell("nb03", 12, nb03_cell12)
    run_cell("nb03", 13, nb03_cell13)

    OUT_PICKLE.parent.mkdir(exist_ok=True)
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(captured, f)
    print(f"\nSaved captured outputs to {OUT_PICKLE}")
