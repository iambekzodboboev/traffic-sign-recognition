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

- [x] 5.1 Set up MLflow tracking (and decide where the tracking data lives,
      given the Colab/local split). **Implemented**: mount Google Drive in
      Colab, point MLflow's tracking URI at a Drive folder (not
      `/content`, which is wiped every session) so runs persist across
      sessions.
- [x] 5.2 Build a simple baseline CNN; first prove the training loop works
      end-to-end on a tiny subset (e.g. 2-3 classes, a couple epochs).
      **Implemented**: 3-conv-block CNN with batch norm, sanity-checked on
      a 3-class subset before the full run.
- [x] 5.3 Train the baseline on the full dataset for a small number of
      epochs; confirm it clearly beats random guessing (~0.5% for 200
      classes) and log it in MLflow. **Done, verified in Colab**: 15
      epochs, Adam. Final epoch: train_loss=1.367, train_acc=0.589,
      val_loss=0.484, val_acc=0.878. Val accuracy climbed steadily every
      single epoch (40.5% -> 87.8%), no plateau, clearly and hugely above
      the 0.5% random-guess baseline. Val > train throughout, expected
      given train-only augmentation and dropout (not overfitting — no
      sign of val trailing off). Hit and fixed two real bugs along the
      way: a missing `import os` in the MLflow setup cell, and
      `mlflow.pytorch.log_model` needing an explicit `input_example` for
      this MLflow version (training itself succeeded first try; only the
      final save step failed and was recovered without retraining, since
      per-epoch checkpointing meant the trained model was never lost).
- [x] 5.4 Evaluate on the validation set; look at a confusion matrix /
      per-class accuracy to see where it struggles (expect the smaller
      classes to be weakest). **Done, verified in Colab.** Surprising
      finding: correlation between per-class accuracy and training-set
      size was only -0.124 (essentially none) — contrary to the
      expectation above, **class imbalance is not what's hurting this
      model.** The confusion pairs tell the real story instead: errors
      are concentrated in specific, visually/semantically similar sign
      pairs (mirrored left/right signs, similar "end of restriction"
      signs, similar informational plate signs) rather than being spread
      thin across low-data classes. Full breakdown in `PROJECT_STATUS.md`
      under "Stage 5 baseline results".

All of 5.1-5.4 implemented together in `notebooks/04_baseline_model.ipynb`
(self-contained). Not yet run — waiting on the user to execute it in
Colab and share the resulting numbers/plots to discuss before deciding
what to try in stage 6.

## 6. Model selection

- [x] 6.0 Cheap visual spot-check on class 184's two biggest confusions,
      done locally (no Colab/GPU needed) before spending a full retraining
      run: sampled 6 random images each from classes 184 ("Coverage area"),
      183 ("Distance to the object"), 193 ("Limitation of parking
      duration") directly from the local dataset zip via
      `scripts/spot_check_184.py`. **Finding: not a labeling bug.** All
      three classes are genuinely distinct, correctly-labeled sign types
      (arrow + distance-in-meters plate; plain distance-in-meters plate;
      duration-in-minutes plate). But 184 vs. 183 are near-visual-twins —
      same white rectangular plate, same font, same "distance in meters"
      text, differing only by the presence/direction of an arrow, which is
      a small detail. 184 vs. 193 differ mainly by "M" vs. "мин" text,
      easily lost at low resolution. This confirms stage 5's hypothesis
      (genuine visual similarity, not mislabeling) and suggests the 64x64
      target resolution itself may be part of why: the distinguishing
      detail in these plates is small text/arrows that may not survive
      that much downsampling.
- [x] 6.1 Try a small number of deliberate improvements one at a time
      (augmentation, class weighting, a deeper network or transfer learning
      backbone), tracking each as its own MLflow run. **Experiment: a
      pretrained ResNet18 backbone (transfer learning), fully fine-tuned**,
      chosen over a from-scratch deeper CNN — stage 5.4 found the
      baseline's weakness was specific visual confusability between sign
      pairs, not lack of raw model size, and a pretrained backbone brings
      much richer discriminative features than more from-scratch capacity
      would. Everything else (64x64 letterbox pipeline, split, batch size,
      15 epochs) kept identical to the baseline for a clean single-variable
      comparison; learning rate lowered to 1e-4 for gentler fine-tuning
      updates. Iterated on twice before running for real: added mixed
      precision (`torch.amp.autocast` + `GradScaler`) after an early
      pre-fix attempt took 2-3 hours in Colab vs. the baseline's 15-50 min
      (data pipeline is identical between the two, so the slowdown was
      ResNet18's heavier GPU compute in full FP32, not data loading); then
      added a "check MLflow for an already-finished run before training"
      guard after realizing every notebook reopen was forcing a full
      retrain with no fast path back to an already-trained model. **Moved
      to Kaggle Notebooks** after Colab's free-tier GPU quota was
      exhausted (from the earlier long pre-AMP attempts) — this dataset is
      natively hosted on Kaggle with a separate quota. Adapted: dataset
      mounts via Kaggle's "+ Add Input" (no download/unzip needed at all),
      MLflow tracking/checkpoints moved from Drive to `/kaggle/working`,
      model also saved there directly as a plain file. Implemented (and
      matches exactly what was executed) in
      `notebooks/05_transfer_learning.ipynb`, built from
      `scripts/build_notebook_05.py`.
      **Result, confirmed on Kaggle: validation accuracy 0.9903** (vs.
      baseline's 0.8784). Every confusion flagged in stage 5 dropped to
      near-zero (e.g. the 122x baseline confusion 190->196 fell to 0x) —
      direct confirmation that a pretrained backbone fixes the specific
      visual-similarity weakness identified in stage 5, even at the same
      64x64 input the stage 6.0 spot-check had flagged as a possible
      limiting factor.
- [x] 6.2 Compare runs and pick the best model using validation metrics
      (overall accuracy plus per-class balance, not accuracy alone):
      macro-averaged F1, per-class accuracy std dev, worst-class accuracy,
      and the same accuracy-vs-training-size correlation check as stage
      5.4. **Result**: ResNet18 wins on every axis — worst-class accuracy
      0.8033 vs. baseline's 0.179 (a different class is now the worst,
      class 54 "Intersection of equivalent roads", since the previously-
      worst classes were fixed); macro-F1 0.9877 sits almost exactly at
      overall accuracy, meaning strength is consistent across all 200
      classes, not concentrated in frequent ones; training-size
      correlation stayed near zero (0.0381), confirming this model is, if
      anything, even less sensitive to the stage 3.2 imbalance than the
      baseline. **ResNet18 selected as the final model.**
- [x] 6.3 Run the chosen model once on the held-out test set for the final,
      reported number (test set untouched until the model was already
      chosen in 6.2, so it isn't implicitly tuned against). **Result:
      final test accuracy 0.9937** (99.37%), slightly *above* validation
      accuracy (0.9903) — a good sign of no overfitting from the iterative
      comparison process. Test macro-F1 0.9937 (essentially matching
      accuracy). Worst class on test: 170 "Phone" at 0.8846 — every one of
      the 15 worst test classes still scored above 88%.

**Stage 6 complete.** Final model: ResNet18 (ImageNet-pretrained, fully
fine-tuned, 64x64 input, 15 epochs, mixed precision), trained and
evaluated on Kaggle Notebooks, 99.37% held-out test accuracy.

## 7. Inference / demo workflow

- [x] 7.1 Export the trained model and the class-name mapping as small
      artifact files. **Done**: downloaded the trained weights
      (`resnet18_transfer_15ep.pt`, ~45MB state dict) directly from the
      live Kaggle session's `/kaggle/working` output (via `IPython.display.FileLink`,
      no need to wait for a full "Save Version" commit) and placed it at
      `models/resnet18_transfer_15ep.pt` locally. Gitignored (`*.pt`,
      same pattern as the dataset zip) — it's a build artifact, not
      source, so it isn't committed; if it's ever missing, redownload it
      from that Kaggle notebook's Output, or from MLflow run
      `21082892cf2d4295851e8c1a96580863`. The class-name mapping
      (`metadata/class_names.csv`) already existed in the repo from stage
      3.5, no re-export needed.
- [x] 7.2 Write a small local inference script: image in, predicted sign name
      + confidence out. Test manually on a few sample photos. **Done**:
      `scripts/predict_sign.py`, runs entirely on CPU (no GPU, dataset, or
      network needed) — loads the ResNet18 architecture, the local
      `models/resnet18_transfer_15ep.pt` weights, and
      `metadata/class_names.csv`; applies the exact same letterbox-resize
      + ImageNet-normalize preprocessing used in training/eval; prints
      top-3 predictions with confidence. **Verified**: tested on 6 sample
      photos pulled from the local dataset zip (classes 0, 14, 44, 45,
      184, 190 — including the previously-confused mirror pair 44/45 and
      class 184's old confusions) — all 6 predicted correctly at
      99.99-100% confidence.
- [x] 7.3 Revisit and decide: Telegram bot or web app for the demo.
      **Decided: Telegram bot** — user sends a photo, bot replies with the
      predicted sign name, category (mandatory/warning/etc. — already
      available in `metadata/class_names.csv`'s `category` column, no new
      work needed), confidence, and a clean reference picture of that
      sign type. Reference icons extracted one-time from the dataset
      zip's `Classes/` folder via `scripts/extract_reference_icons.py`
      into `assets/class_icons/<class_id>.png` (200/200 resolved
      unambiguously, 1.4MB total, small enough to commit directly unlike
      the dataset zip or model weights).
- [x] 7.4 Build and test a minimal version of the chosen interface end-to-end
      with a real photo. **Done**: `bot.py`, built with `pyTelegramBotAPI`,
      reuses `scripts/predict_sign.py`'s model-loading/prediction logic
      directly. Runs entirely on CPU, no GPU/Kaggle needed. Bot token
      registered via @BotFather, stored in a local gitignored `.env`
      file (loaded via `python-dotenv`), never committed. **Verified
      working end-to-end**: user messaged the live bot on Telegram,
      sent a real photo, got back the correct sign name + category +
      confidence + reference icon.

**Stage 7 (inference/demo) complete.**

## 8. Reproducibility, documentation, defense prep

- [ ] 8.1 Clean up the final notebook/scripts and confirm `requirements.txt`
      is accurate, so the whole pipeline can be rerun from scratch.
- [ ] 8.2 Update `README.md` with how to reproduce and the main results.
- [ ] 8.3 Update `PROJECT_STATUS.md` to reflect completion and prepare a
      short defense summary of what was done and why.

## 9. Real-World Video Detection (in progress — steps 9.1-9.4 done)

**Why this is a new phase, not a tweak**: the trained classifier (stages
5-6) and the bot (stage 7) both assume the input is already a close, clean
photo of a single sign — matching exactly what the training dataset looks
like. Real-world testing of the deployed bot confirmed this directly: a
wide street-scene photo (sign small within a lot of background) came back
low-confidence and wrong, while every close-up test photo scored 99%+ (see
`PROJECT_STATUS.md`, "Known problems"). To work on real driving video —
dashcam or "blogger driver" style footage, where a sign is one small,
moving detail in a busy scene — the system needs a capability it doesn't
have yet: finding *where* a sign is before deciding *which* sign it is.
That's a different, additional ML task (object detection), layered in
front of the classifier that already works, not a retraining of it.

**The reassuring part, confirmed true in practice**: the trained ResNet18
classifier (99.37% test accuracy) was never retrained or touched for any
of this. It's reused exactly as-is, read-only, from `scripts/predict_sign.py`.

**Isolation, honored throughout**: everything for this phase lives in
`video_detection/`, a separate component. `bot.py` (the Telegram bot) was
never modified and still works exactly as it did at the end of stage 7.

- [x] 9.1 Choose a detection approach. **Decision: classical computer
      vision (color + shape detection in OpenCV), not a downloaded
      pretrained detector.** This deviates from the roadmap's original
      preference, for two concrete reasons found during actual testing,
      not assumed up front: (1) downloading a third-party detector
      checkpoint from an untrusted source is a real security risk (unlike
      the ResNet18 weights, which came from torchvision's own official,
      trusted channel); (2) tested anyway with the official, trusted
      Ultralytics YOLOv8 (COCO-pretrained) as a fair comparison — it
      completely missed a clearly-visible STOP sign in a real test frame
      (COCO's "stop sign" class didn't generalize to this angled, small,
      real-world case), while the classical color+shape approach found
      it correctly on the first try, plus a pedestrian-crossing sign in
      the same frame. Classical CV won the comparison outright, and
      needs no external model trust decision at all. Implemented in
      `video_detection/detector.py`.
- [x] 9.2 Real-world validation — done pragmatically, not via a formal
      labeled dataset. Rather than building a ~50-100 frame
      hand-annotated bounding-box set with a separate tool (the original
      plan), validation was done by direct visual inspection: extracting
      real frames from the user's own driving-test-track video (see
      below) and checking the detector's boxes against them by eye at
      each iteration. A deliberate scope simplification for a course
      project — faster, and caught every real problem found so far
      (see 9.3-9.4 below) just as effectively.
      **Test video**: a real ~3-minute driving-test-track clip
      (portrait 720x1280, ~30fps) provided by the user, showing a
      "driving school test" style route (Uzbek captions, channel
      watermark overlay) with STOP signs, a pedestrian crossing,
      warning triangles, mandatory circles, lane-direction signs, and
      traffic lights. Stored at `video_detection/test_videos/`
      (gitignored, same treatment as the dataset zip).
- [x] 9.3 Build the detect → crop → classify pipeline.
      `video_detection/classifier.py` crops each detected box and runs
      it through the exact same letterbox-resize + normalize
      preprocessing already built in stage 4 (imported directly from
      `scripts/predict_sign.py`, never duplicated or modified), then the
      existing trained classifier. Two real bugs found and fixed via
      direct testing before moving on: (a) a detected box can extend
      slightly into a neighboring object (e.g. a car mirror next to a
      sign) — tried a 12% inward shrink margin, which *broke* an
      already-correct STOP sign result (99.4% → 22%, wrong) by cutting
      into the sign's own text; settled on a gentler 5% margin, which
      helps the mirror case without breaking the working one; (b) found,
      and explicitly did **not** try to fix, a deeper issue: a channel
      watermark rendered semi-transparently *on top of* a sign (not
      beside it) causes confident misclassification that no box-boundary
      adjustment can remove, since those pixels are inside the sign's own
      region — a real limitation specific to "blogger"/social-media
      style footage, stated honestly rather than patched over.
- [x] 9.4 Adapt for continuous video. `video_detection/process_video.py`
      samples the video (not every frame) and runs detect+classify per
      sampled frame; `video_detection/tracker.py` matches detections
      across samples by box overlap (IoU) so the same physical sign
      isn't reported fresh every time, and only reports a track once a
      *majority* of its confident observations agree on the same class
      (the "not just one lucky frame" rule). Two real problems found by
      testing, not anticipated up front: sampling at 2 fps left gaps
      large enough that a sign moving fast across the frame (not just
      growing, e.g. one off to the side of the road) could have *zero*
      box overlap between consecutive samples — no IoU threshold fixes
      a true zero-overlap gap. Fixed with (a) a center-to-center distance
      fallback match for when IoU finds no overlap at all, and (b)
      raising the sample rate to 4 fps (cheap: this pipeline runs at
      ~0.05s/sampled frame, far from any real-time limit). Together,
      these turned what were 5 fragmented "Stop" tracks and a 3-way-split
      fast-moving sign into 2 and 1 clean continuous tracks.
      **Full-video result**: 738 sampled frames (~4 fps, 172s video) in
      39 seconds, 76 deduplicated tracked signs reported.
- [~] 9.5 Test honestly on real footage and document what breaks —
      substantial honest testing already done throughout 9.1-9.4 (not a
      separate pass yet). Confirmed, specific findings, each verified by
      direct inspection rather than assumed:
      1. Correctly detects and classifies sign types never
         individually hand-tested before (e.g. "Steep ascent 12%";
         "Curve right" held correctly across 85 consecutive sampled
         frames as the car approached it).
      2. A channel watermark drawn on top of a sign causes confident
         misclassification (see 9.3) — footage-specific, not fixable
         by box adjustment.
      3. Two signs close together can merge into one detection box; the
         resulting crop confuses the classifier, but this reliably
         shows up as *low* confidence rather than a wrong confident
         answer — the safety net works as intended here.
      4. Feeding a non-sign red/round object (a traffic light lens) to
         the classifier produced a moderately confident wrong answer
         (~65-77%) — confidence thresholding alone doesn't catch every
         such case.
      5. Many short tracks reported "Lane directions" repeatedly.
         Spot-checked directly (not assumed to be noise): these are
         real small signs, visually confirmed (one on a post next to a
         painted lane arrow on the pavement) — but the crops are tiny
         and blurry, and a "confident" score means less on a crop that
         degraded than on the large, clean crops tested earlier.
      Not yet done: a dedicated pass against a *second*, different
      video (different lighting/route/camera mount) to check how much
      of the above is specific to this one test clip.
- [ ] 9.6 Package a demo: a script that takes a video file (or a live
      camera feed) and produces an annotated output — boxes and labels
      drawn on the video, or the live "which rule is active right now"
      style already sketched conceptually in the project presentation's
      future-vision slide. `process_video.py` already does the video-in,
      annotated-video-out part; what's left is packaging it as a clean,
      documented, easy-to-run demo (plus optionally the "live camera
      feed" variant). This is the concrete bridge from "a model that
      classifies photos" to the dash-cam driving-assistant concept
      already pitched as future work.

**Code map for this phase** (all in `video_detection/`, isolated from
the bot): `detector.py` (localization) → `classifier.py` (wires a crop
into the existing model) → `tracker.py` (deduplicates across frames) →
`process_video.py` (ties it all together: sample → detect → classify →
track → annotate + report). Full narrative and findings in
`video_detection/README.md`.

**Honest scope note**: this is a meaningfully bigger undertaking than
anything built so far — it introduces a whole additional ML sub-field
(object detection), a new small labeled validation set, and real
video-processing engineering. It's appropriately a future/stretch phase,
not a quick follow-on task.
