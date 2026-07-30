# Roadmap

Detailed, small-step breakdown of the project, grouped by the stages in
`README.md` / `AGENTS.md`. We do these one at a time: do the step, check the
result together, commit, push, then move to the next one.

Dataset (already confirmed by listing the zip, no extraction/processing done
yet): `Data/` has 200 classes, 116,642 images total, class sizes range from
248 to 1,845 (real imbalance). `Classes/` gives human-readable names per
class ID. Filenames look track/frame-numbered, which is a likely data-leakage
risk if not handled carefully (see stage 3).

## 0. Workflow setup (Colab + GitHub)

- [x] 0.1 Decide how Colab and GitHub connect. Decided and **verified
      working**: open the notebook directly from GitHub in Colab (File ->
      Open notebook -> GitHub tab), edit there, then File -> Save a copy in
      GitHub to push back — confirmed via the `Created using Colab` commit
      on GitHub. GPU also confirmed working (T4, `CUDA available: True`) —
      note: each open notebook tab in Colab has its own separate runtime,
      so hardware accelerator must be set per notebook, not once globally.
      No git commands needed inside Colab.
- [x] 0.2 Decide how the dataset gets into Colab for each session. Original
      dataset source confirmed:
      [traffic-signs-in-post-soviet-states-200-classes](https://www.kaggle.com/datasets/mikhailkosov/traffic-signs-in-post-soviet-states-200-classes)
      on Kaggle (matches the local dataset). Final decision: the user
      uploaded the dataset zip to their own Google Drive, so each Colab
      session downloads it from there and unzips it to local session disk
      before use — reading the 116,642 individual images directly from a
      mounted Drive would be too slow for training. Originally implemented
      with `gdown` against the public share link; switched to the
      authenticated Drive API (`google.colab.auth` + `googleapiclient`)
      after hitting Google's public-link download quota ("Too many users
      have viewed or downloaded this file recently") from repeated testing
      — the authenticated approach downloads by file ID as the owner and
      isn't subject to that quota. Implemented in
      `notebooks/01_dataset_download_colab.ipynb`. **Verified working**:
      run in Colab, output matched the local audit exactly (200 classes,
      116,642 images).
- [x] 0.3 Add a `notebooks/` folder to the repo as home for the Colab
      notebook(s). Added with a connection-check notebook
      (`00_colab_github_connection_check.ipynb`) to test 0.1.

## 1. Scope and problem definition — done

Input: a photo of a traffic sign. Output: which of the 200 sign types it is.

## 2. Repository and project setup — done

Git/GitHub set up, Python environment installed, dataset present locally.

## 3. Data audit and EDA

- [x] 3.1 Verify dataset integrity: confirm all 200 class folders load, spot
      check for corrupt/unreadable images. **Done, verified in Colab**: all
      200 class folders present, all 116,642 images checked (full check,
      not a sample) — 0 corrupt/unreadable.
- [x] 3.2 Class distribution: bar chart of images per class, confirm/quantify
      the imbalance already spotted (248-1,845 per class). **Done, verified
      in Colab**: confirmed 248-1,845 range, mean 583 images/class
      (116,642 / 200). Imbalance is real but moderate/gradual, not
      extreme — most classes cluster 250-1,000, with a handful spiking up
      toward the max. No classes with only a handful of images, so class
      weighting + moderate augmentation for smaller classes should be
      enough later (no need for drastic measures).
- [x] 3.3 Image property audit: resolution range, aspect ratios, color mode
      (RGB vs grayscale), file size range. **Done, verified in Colab**:
      width 25-2,749px, height 25-1,496px, but median only 191x157 (long
      tail of larger images pulls the mean up) — a modest target resize
      size (e.g. 64x64 or 128x128) makes sense. Aspect ratio mostly near
      1.0 (median 1.065) with a real tail up to 5.88 (wide/thin plates).
      Color mode: majority RGBA (96,444), minority RGB (20,198) — all have
      real color, no grayscale, but alpha channel needs flattening to RGB
      before training.
- [x] 3.4 Visual sample audit: plot a grid of sample images from a handful of
      classes to sanity-check labels are actually correct. **Done, verified
      in Colab**: 8 random classes x 4 images each, all rows internally
      consistent (same sign type per row despite angle/lighting/quality
      differences). No labeling issues found. Confirms non-square classes
      like time-restriction text plates (e.g. class 189) are a real,
      expected sign type, not a data error.
- [x] 3.5 Build a class ID -> human-readable name mapping using the
      `Classes/` folder (needed later so predictions say "Stop sign" instead
      of "class 14"). **Done**: derived directly from the dataset zip's
      folder structure (not hand-typed, to avoid transcription errors
      across 200 rows), committed at `metadata/class_names.csv`, and
      cross-checked in Colab against the 200 classes with images — exact
      match. One data quirk found: class 81 has two conflicting names in
      the source taxonomy ("Height limit - 3.5" vs "Height limit - 4.5"),
      flagged in the CSV rather than silently guessing.
- [x] 3.6 Investigate the track/frame-like filenames (e.g.
      `00000_00000_00017.png`) to confirm whether multiple images are near-
      duplicate frames of the same physical sign. This determines how we
      must split data in stage 4 to avoid leakage. **Done, verified**:
      confirmed real for the ~10.3% of files matching the clean pattern
      (~900 groups, mean ~13 images/group, ~82% with 5+ images, visually
      confirmed as near-duplicate frames). But this dataset mixes several
      naming conventions from what looks like multiple merged sources —
      the other ~90% of files don't cleanly parse into a track ID.
      Filename parsing alone isn't sufficient to build safe splits;
      stage 4.1 needs content-based near-duplicate detection (e.g.
      perceptual hashing) to cover the whole dataset.
- [x] 3.7 Write up findings (imbalance, leakage risk, image properties) as a
      short summary in `PROJECT_STATUS.md`. **Done** — see the "Stage 3
      data audit summary" section there.

## 4. Preprocessing and leakage prevention

- [x] 4.1 Decide the train/validation/test split strategy. Per 3.6: build
      near-duplicate groups via content-based hashing (not filename
      parsing alone, which only covers ~10% of files), then split by
      group, not by individual image, so no near-duplicate leaks across
      splits. **Done**: 64-bit average-hash per image, grouped within each
      class by Hamming distance <= 6, computed once locally (deterministic,
      seed 42) via `scripts/build_split_manifest.py` — not recomputed
      randomly each session, so results stay comparable across the
      project. Known limitation stated honestly: grouping is fairly
      strict (~2.6 images/group avg, vs ~13/group for the filename-
      confirmed tracks in 3.6), so it catches exact/near-exact duplicates
      reliably but may miss some more visually-different frames of the
      same sign. Meaningfully reduces leakage risk vs. naive per-image
      splitting; not a perfect guarantee.
- [x] 4.2 Implement the split and save it as a manifest (a CSV of
      filename -> split), rather than physically copying files. **Done**:
      `metadata/split_manifest.csv` (116,642 rows), split 70/15/15 by
      group (31,820 / 6,822 / 6,822 groups), ~70.9/15.5/13.5% by image.
      Validated in `notebooks/03_preprocessing.ipynb` (4.2): all manifest
      paths exist on disk, row count matches dataset image count, and no
      group spans more than one split.
- [x] 4.3 Decide the preprocessing pipeline: target image size, normalization,
      and an augmentation plan for the smaller classes to help with the
      imbalance found in 3.2. **Done**: 64x64 target (median was 191x157
      per 3.3), letterbox resize (pad to square then resize, not a plain
      stretch, since aspect ratio had a real tail to 5.88), RGBA -> RGB,
      ImageNet normalization (keeps a pretrained-backbone option open for
      6.1), light augmentation (rotation + color jitter) on training data
      only.
- [x] 4.4 Build the PyTorch Dataset/DataLoader from the split manifest; sanity
      check by loading and visualizing one batch. **Done, verified in
      Colab**: custom `TrafficSignDataset` reading the manifest,
      `WeightedRandomSampler` on the training loader (oversamples minority
      classes, addressing 3.2's imbalance without class-specific
      augmentation logic). Batch shape confirmed `[64, 3, 64, 64]`;
      letterboxing visible (black padding on non-square signs, not
      stretched); augmentation visible (rotation/color jitter); class
      names matched their images correctly (e.g. class 97 showed a
      hazmat/explosive-load icon, class 103 showed pedestrian+bicycle
      icons).

## 5. Baseline model and experiments

- [ ] 5.1 Set up MLflow tracking (and decide where the tracking data lives,
      given the Colab/local split).
- [ ] 5.2 Build a simple baseline CNN; first prove the training loop works
      end-to-end on a tiny subset (e.g. 2-3 classes, a couple epochs).
- [ ] 5.3 Train the baseline on the full dataset for a small number of
      epochs; confirm it clearly beats random guessing (~0.5% for 200
      classes) and log it in MLflow.
- [ ] 5.4 Evaluate on the validation set; look at a confusion matrix /
      per-class accuracy to see where it struggles (expect the smaller
      classes to be weakest).

## 6. Model selection

- [ ] 6.1 Try a small number of deliberate improvements one at a time
      (augmentation, class weighting, a deeper network or transfer learning
      backbone), tracking each as its own MLflow run.
- [ ] 6.2 Compare runs and pick the best model using validation metrics
      (overall accuracy plus per-class balance, not accuracy alone).
- [ ] 6.3 Run the chosen model once on the held-out test set for the final,
      reported number.

## 7. Inference / demo workflow

- [ ] 7.1 Export the trained model and the class-name mapping as small
      artifact files.
- [ ] 7.2 Write a small local inference script: image in, predicted sign name
      + confidence out. Test manually on a few sample photos.
- [ ] 7.3 Revisit and decide: Telegram bot or web app for the demo.
- [ ] 7.4 Build and test a minimal version of the chosen interface end-to-end
      with a real photo.

## 8. Reproducibility, documentation, defense prep

- [ ] 8.1 Clean up the final notebook/scripts and confirm `requirements.txt`
      is accurate, so the whole pipeline can be rerun from scratch.
- [ ] 8.2 Update `README.md` with how to reproduce and the main results.
- [ ] 8.3 Update `PROJECT_STATUS.md` to reflect completion and prepare a
      short defense summary of what was done and why.
