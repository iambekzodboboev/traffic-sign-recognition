"""
Build a leakage-safe train/val/test split manifest for the traffic sign dataset.

Why this exists: roadmap step 3.6 found that near-duplicate photos of the
same physical sign exist in this dataset (video-burst-like frames), but
only ~10% of files have a filename pattern that identifies which photos
belong together. This script instead groups images by visual similarity
(a simple perceptual average-hash, compared within each class only), then
splits train/val/test by GROUP rather than by individual image, so no
near-duplicate frames of the same sign can leak across the split.

The split is deterministic (fixed random seed) and written once to
metadata/split_manifest.csv, which is committed to the repo. It is not
meant to be regenerated with fresh randomness on every run -- re-running
this script reproduces the same split, it doesn't create a new one.

Usage:
    python scripts/build_split_manifest.py

Reads:  "Traffic sign dataset.zip" (project root)
Writes: metadata/split_manifest.csv
"""
import csv
import io
import os
import random
import zipfile
from collections import defaultdict

from PIL import Image

ZIP_PATH = "Traffic sign dataset.zip"
OUTPUT_PATH = "metadata/split_manifest.csv"
HASH_SIZE = 8  # 8x8 -> 64-bit average hash
HAMMING_THRESHOLD = 6  # bits; images within this distance are grouped together
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def average_hash(img: Image.Image, hash_size: int = HASH_SIZE) -> int:
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return bits


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def build_split(n, ratios, rng):
    """Split n items into train/val/test index lists using the given ratios."""
    order = list(range(n))
    rng.shuffle(order)
    n_val = max(1, round(n * ratios["val"])) if n >= 2 else 0
    n_test = max(1, round(n * ratios["test"])) if n >= 3 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        remaining = n - n_train
        n_val = remaining - remaining // 2
        n_test = remaining // 2
    train = order[:n_train]
    val = order[n_train:n_train + n_val]
    test = order[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def main():
    z = zipfile.ZipFile(ZIP_PATH)
    names = sorted(n for n in z.namelist() if n.startswith("Data/") and not n.endswith("/"))

    by_class = defaultdict(list)
    for n in names:
        class_id = n.split("/")[1]
        by_class[class_id].append(n)

    rows = []  # (relative_path, class_id, group_id)
    group_counter = 0
    warnings = []

    for class_id in sorted(by_class, key=int):
        files = by_class[class_id]
        hashes = []
        for n in files:
            with z.open(n) as f:
                img = Image.open(io.BytesIO(f.read()))
                hashes.append(average_hash(img))

        uf = UnionFind(len(files))
        for i in range(len(files)):
            hi = hashes[i]
            for j in range(i + 1, len(files)):
                if (hi ^ hashes[j]).bit_count() <= HAMMING_THRESHOLD:
                    uf.union(i, j)

        local_group_ids = {}
        for i, n in enumerate(files):
            root = uf.find(i)
            if root not in local_group_ids:
                local_group_ids[root] = group_counter
                group_counter += 1
            rel_path = n[len("Data/"):]
            rows.append((rel_path, class_id, local_group_ids[root]))

        n_groups = len(local_group_ids)
        if n_groups < 3:
            warnings.append(f"class {class_id}: only {n_groups} group(s) for {len(files)} images")
        print(f"class {class_id}: {len(files)} images -> {n_groups} groups")

    print(f"\nTotal images: {len(rows)}, total groups: {group_counter}")
    if warnings:
        print(f"\n{len(warnings)} class(es) with very few groups (can't split cleanly):")
        for w in warnings:
            print(" ", w)

    groups_by_class = defaultdict(set)
    for rel_path, class_id, group_id in rows:
        groups_by_class[class_id].add(group_id)

    rng = random.Random(RANDOM_SEED)
    group_split = {}
    for class_id, group_ids in groups_by_class.items():
        group_ids = sorted(group_ids)
        train_idx, val_idx, test_idx = build_split(len(group_ids), SPLIT_RATIOS, rng)
        for i in train_idx:
            group_split[group_ids[i]] = "train"
        for i in val_idx:
            group_split[group_ids[i]] = "val"
        for i in test_idx:
            group_split[group_ids[i]] = "test"

    os.makedirs("metadata", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "class_id", "group_id", "split"])
        for rel_path, class_id, group_id in rows:
            writer.writerow([rel_path, class_id, group_id, group_split[group_id]])

    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")

    split_counts = defaultdict(int)
    for gid, s in group_split.items():
        split_counts[s] += 1
    print("Split counts (by group):", dict(split_counts))

    image_split_counts = defaultdict(int)
    for _, _, gid in rows:
        image_split_counts[group_split[gid]] += 1
    print("Split counts (by image):", dict(image_split_counts))


if __name__ == "__main__":
    main()
