# Project Status

## Project

Traffic Sign Recognition

## Current stage

Stages 0-7, 9 (9.1-9.6), and 10 (public Streamlit web demo) **complete**.
Stage 8 nearly complete: 8.1 and 8.3 done, only 8.2 (`README.md` update)
remains. See `ROADMAP.md` for the full step-by-step plan.

## Defense summary

**Goal**: recognize traffic signs (200 classes, post-Soviet-states sign
set) from a photo, served through a Telegram bot, then extended toward
recognizing signs in real driving video.

**Approach, stage by stage, and why**:
- **Data audit (3)** surfaced two real risks before any model was
  touched: moderate class imbalance (248-1,845 images/class) and a
  data-leakage risk from near-duplicate video-burst frames.
- **Preprocessing (4)** built a leakage-safe split by grouping
  near-duplicate images (perceptual hashing), since filenames alone only
  reliably identified ~10% of the dataset's groupings — then split by
  group, never by individual image.
- **Baseline (5)**: a from-scratch 3-conv-block CNN reached 87.84% val
  accuracy. Confusion-matrix analysis found the real bottleneck wasn't
  imbalance (correlation with class size ~ -0.124, essentially none) but
  specific, visually-similar sign pairs being confused with each other.
- **Model selection (6)**: swapping in a pretrained ResNet18 (same
  pipeline, only the backbone changed) eliminated those exact confusions
  and reached **99.37% held-out test accuracy**. This is targeted
  evidence, not a generic accuracy bump — it specifically fixed the exact
  confusions predicted by the stage 5 analysis, which is a harder pattern
  to explain away as leakage than a uniform improvement would be.
- **Demo (7)**: a Telegram bot serving the trained classifier, CPU-only,
  with a confidence threshold that asks for a clearer photo below 60%
  (added after a real low-confidence misclassification surfaced during
  live testing).
- **Video detection (9)**: extends the classifier to real driving footage
  by adding a localization step in front of it — classical computer
  vision (color + shape), chosen over a downloaded pretrained detector
  after a direct side-by-side test showed it outperforming official
  COCO-pretrained YOLOv8 on this exact task — plus cross-frame tracking
  so the same physical sign isn't reported repeatedly. Tested against two
  different real videos; found and documented real failure modes rather
  than reporting only the successes (see below).

**Headline numbers**: 99.37% test accuracy across 200 classes (random
baseline 0.5%), test macro-F1 0.9937, worst-of-200-classes test accuracy
88.46% — meaning strength is consistent across classes, not concentrated
in a few frequent ones.

**Honest limitations, worth stating directly if asked**:
- Stage 4's near-duplicate deduplication isn't perfect — a stated, known
  gap (see "Stage 4 known limitation" below), not hidden.
- Stage 9's video pipeline has real, found failure modes: a channel
  watermark or on-screen caption sitting on top of a sign can cause
  confident misclassification; two sign faces stacked on one post can be
  boxed together and confidently mislabeled as something unrelated; a box
  that captures only a sign's icon (not its full multi-plate context) can
  confuse near-visual-twin classes. All documented, not swept under the
  rug — see `video_detection/README.md`.
- Confidence scores are a much weaker trust signal on small/blurry video
  crops than on the large, clean photos the bot was tested with.

**Where the full detail lives**: this file (stage-by-stage results
below), `ROADMAP.md` (the full staged plan and reasoning behind every
decision), `Traffic_Sign_Recognition_Project_Report.docx` (narrative
report), `Traffic_Sign_Recognition_Presentation.pptx` (defense slide
deck), `video_detection/README.md` (video detection detail).

## Completed

- Local Git repository initialized
- GitHub repository created and connected (`traffic-sign-recognition`, public)
- Baseline `.gitignore`, `README.md`, `AGENTS.md` in place
- Python 3.12 environment set up in `.venv`, with PyTorch (CPU), torchvision,
  pandas, matplotlib, scikit-learn, OpenCV, Jupyter/JupyterLab, MLflow, and
  the Kaggle CLI installed; pinned in `requirements.txt`
- Dataset (~14.9GB, 200 classes, 116,642 images) obtained from Kaggle
  ([mikhailkosov/traffic-signs-in-post-soviet-states-200-classes](https://www.kaggle.com/datasets/mikhailkosov/traffic-signs-in-post-soviet-states-200-classes)),
  uploaded to the user's Google Drive
- Stage 0 (workflow setup) complete: Colab ↔ GitHub round trip verified
  (open from GitHub, save back to GitHub), GPU (T4) confirmed working,
  dataset download in Colab via authenticated Drive API (`notebooks/01_dataset_download_colab.ipynb`)
- Stage 3.1 done: full integrity check in Colab, all 116,642 images verified
  openable, 0 corrupt, all 200 class folders present
- Stage 3.2 done: class distribution confirmed (248-1,845 images/class,
  mean 583) — real but moderate imbalance, no classes with only a
  handful of images
- Stage 3.3 done: image sizes range 25-2,749px wide, 25-1,496px tall
  (median only 191x157 — long tail of larger images); aspect ratio mostly
  near 1.0 with a real tail to 5.88 (wide/thin plates); color mode is
  majority RGBA (96,444) vs RGB (20,198), no grayscale
- Stage 3.4 done: visual spot-check of 8 random classes x 4 images each,
  all internally consistent, no labeling issues found
- Stage 3.5 done: class ID -> name mapping built from the dataset's own
  `Classes/` folder structure, committed at `metadata/class_names.csv`;
  one data quirk flagged (class 81 has two conflicting names in the
  source taxonomy)
- Stage 3.6 done: confirmed near-duplicate/leakage risk is real for the
  ~10.3% of files with cleanly parseable track filenames; the rest use
  other naming conventions (dataset looks merged from multiple sources)
  and need content-based deduplication instead of filename parsing
- Stage 3.7 done: findings written up below
- Work for stage 3 is in `notebooks/02_data_audit_eda.ipynb`
- Stage 4.1 done: train/val/test split strategy decided and implemented —
  64-bit average-hash near-duplicate grouping within each class (Hamming
  distance <= 6), computed once deterministically (not re-randomized each
  session) via `scripts/build_split_manifest.py`
- Stage 4.2 done: split saved as `metadata/split_manifest.csv` (116,642
  rows), 70/15/15 by group (31,820 / 6,822 / 6,822), ~70.9/15.5/13.5% by
  image; validated in Colab (all paths exist, row count matches, no group
  spans multiple splits)
- Stage 4.3 done: preprocessing pipeline decided — 64x64 target, letterbox
  resize (pad to square, not a plain stretch, given the aspect-ratio tail
  found in 3.3), RGBA -> RGB, ImageNet normalization, light augmentation
  (rotation + color jitter) on training data only
- Stage 4.4 done: PyTorch `TrafficSignDataset` + DataLoaders built from the
  manifest, with a `WeightedRandomSampler` on training to oversample
  minority classes (addresses 3.2's imbalance); sanity-check batch
  verified in Colab — batch shape `[64, 3, 64, 64]` correct, letterbox
  padding visible, augmentation visible, class names matched their images
- Stage 4 fully complete and verified
- Work for stage 4 is in `notebooks/03_preprocessing.ipynb`
- Stage 5.1-5.4 done and verified in Colab: MLflow tracking (persisted to
  Drive), baseline 3-conv-block CNN sanity-checked on a 3-class subset,
  full 15-epoch training run (final val_acc=0.878), and full evaluation
  (confusion matrix, per-class accuracy, correlation, confused pairs)
- Trained model confirmed saved permanently in MLflow (Drive-backed),
  independent of notebook/session state
- Work for stage 5 is in `notebooks/04_baseline_model.ipynb`
- Stage 6.0 done: cheap visual spot-check (no Colab/GPU) on class 184
  ("Coverage area") against its two biggest baseline confusions, 183
  ("Distance to the object") and 193 ("Limitation of parking duration") —
  sampled directly from the local dataset zip via
  `scripts/spot_check_184.py`. **Not a labeling bug**: all three are
  genuinely distinct, correctly-labeled sign types. 184 and 183 are
  near-visual-twins (same white plate/font/"distance in meters" text,
  differing mainly by an arrow); 184 and 193 differ mainly by "M" vs.
  "мин" text. Confirms stage 5's "genuine visual similarity, not
  mislabeling" hypothesis, and suggests the 64x64 target resolution may
  be part of the problem (small text/arrows may not survive that
  downsampling).
- Stage 6.1 done: ResNet18 (ImageNet-pretrained, fully fine-tuned)
  transfer-learning experiment, chosen over a from-scratch deeper CNN
  because stage 5.4's finding was about confusability between specific
  sign pairs, which pretrained features address directly. Same 64x64
  pipeline, split, batch size, and 15 epochs as the baseline (LR lowered
  to 1e-4) — only the model changed, for a clean comparison. Iterated
  twice before the real run (added mixed precision after an early attempt
  took 2-3 hours in Colab; added a "skip training if an MLflow run
  already finished" guard after reopening the notebook kept forcing full
  retrains), then moved to Kaggle Notebooks after Colab's free-tier GPU
  quota ran out. Run on Kaggle, trained and evaluated in
  `notebooks/05_transfer_learning.ipynb` (built from
  `scripts/build_notebook_05.py`, which matches exactly what was
  executed). **Result: validation accuracy 0.9903** (vs. baseline's
  0.8784) — every confusion flagged in stage 5 dropped to near-zero.
- Stage 6.2 done: compared the two models on validation metrics beyond
  raw accuracy (macro-F1, per-class accuracy std dev, worst-class
  accuracy, accuracy-vs-training-size correlation). **ResNet18 wins on
  every axis** — worst-class accuracy 0.8033 vs. baseline's 0.179;
  macro-F1 0.9877 nearly matches overall accuracy (consistent strength
  across all 200 classes); training-size correlation stayed near zero
  (0.0381). **ResNet18 selected as the final model.**
- Stage 6.3 done: final, one-time evaluation of the chosen model on the
  held-out test set (untouched until the model was already selected).
  **Final test accuracy: 0.9937** (99.37%), slightly above validation
  accuracy — no overfitting from the iterative comparison process. Test
  macro-F1 0.9937; worst test class 170 ("Phone") at 0.8846.
- **Stage 6 (model selection) fully complete.**
- Stage 7.1 done: trained weights downloaded from the live Kaggle session
  (`resnet18_transfer_15ep.pt`, ~45MB state dict, via
  `IPython.display.FileLink`) and placed at
  `models/resnet18_transfer_15ep.pt` locally. Gitignored like the dataset
  zip (build artifact, not source) — if ever missing, redownload from
  that Kaggle notebook's Output or MLflow run
  `21082892cf2d4295851e8c1a96580863`. Class-name mapping reused as-is
  from `metadata/class_names.csv` (stage 3.5), no re-export needed.
- Stage 7.2 done: `scripts/predict_sign.py`, a local CPU-only inference
  script (image in, predicted sign name + confidence out) — no GPU,
  dataset, or network access needed. Applies the exact same letterbox +
  ImageNet-normalize preprocessing as training/eval. **Verified working**
  on 6 sample photos pulled from the local dataset zip (classes 0, 14,
  44, 45, 184, 190, including the previously-confused 44/45 mirror pair
  and 184's old confusions) — all 6 predicted correctly at 99.99-100%
  confidence.
- Stage 7.3 done: decided on a **Telegram bot** for the demo — user sends
  a photo, bot replies with sign name, category (mandatory/warning/etc.
  — already in `metadata/class_names.csv`'s `category` column, no new
  work), confidence, and a clean reference icon. Extracted one canonical
  reference icon per class from the dataset zip's `Classes/` folder via
  `scripts/extract_reference_icons.py` into `assets/class_icons/` (all
  200 resolved unambiguously; 1.4MB total, committed directly).
- Stage 7.4 done: `bot.py`, built with `pyTelegramBotAPI`, reuses
  `scripts/predict_sign.py`'s model-loading/prediction logic directly —
  runs entirely on CPU. Bot token from @BotFather stored in a local
  gitignored `.env` (loaded via `python-dotenv`), never committed.
  **Verified working end-to-end**: tested live on Telegram with a real
  photo, got back the correct sign name, category, confidence, and
  reference icon.
- **Stage 7 (inference/demo workflow) fully complete.**
- Defense documents done: `Traffic_Sign_Recognition_Project_Report.docx`
  — a comprehensive, stage-by-stage narrative report covering the whole
  project (stages 0-7), organized around what was measured at each step,
  what it showed, and what decision followed from it, readable without
  needing to look at the code. Includes three charts generated from the
  real data/results and the full results tables.
- **Stage 8.1 done**: scripts and `requirements.txt` confirmed accurate
  (no changes needed); `02_data_audit_eda.ipynb` and
  `03_preprocessing.ipynb` re-run locally against the dataset to attach
  real, current outputs (previously code-only); `04_baseline_model.ipynb`
  and `05_transfer_learning.ipynb` given a short note instead of a costly
  GPU re-run. **Stage 8.3 done**: added a "Defense summary" section to
  this file. `README.md` (8.2) is the only remaining item.
- Also produced `Traffic_Sign_Recognition_Presentation.pptx` — a 13-slide,
  10-minute defense deck (demand → detailed methodology with real
  charts/stats → future vision: real-time video on a dash-cam device,
  with a custom concept illustration). Built with `python-pptx` (this
  machine has no Node.js for the usual `pptxgenjs` path); QA'd via
  PowerPoint COM automation → PDF → per-slide visual review, plus
  `markitdown` content checks and the pptx skill's schema validator (all
  passed).
- **Stage 9 (9.1-9.6) complete** (new, isolated component at
  `video_detection/`, never touches `bot.py` or anything it depends on —
  reuses the trained classifier read-only via `scripts/predict_sign.py`).
  Extends the finished classifier to real driving video, where a sign is
  small within a busy scene rather than already cropped to fill the
  frame. Tested against two different real videos (9.5), packaged as a
  runnable demo with a `--help`/`--seconds` CLI and a Quickstart in
  `video_detection/README.md` (9.6). Full details and all findings in
  `video_detection/README.md` and `ROADMAP.md` section 9; headline
  results in the section below.
- **Stage 10 (public Streamlit web demo) complete** (new, isolated
  component at `streamlit_app/`, reuses `video_detection/`'s
  detector/classifier/tracker and `process_video.py`'s own constants
  read-only). Three distinct UI directions were mocked up and shown to
  the user first; **"Signal"** (clean light dashboard) was chosen and
  built. Upload a video or paste a link; the source is trimmed to the
  processing cap and transcoded to browser-playable H.264 up front (via
  `imageio-ffmpeg`'s bundled binary, no system ffmpeg needed) so the
  *same* clip is used for both detection and playback. While it scans,
  the video actually plays (autoplay, muted) with a live-updating
  detection feed alongside it; at the end, a trip report (stat cards,
  category breakdown, most-common-sign callout, CSV download) plus an
  **interactive sign list** where clicking any sign seeks the video to
  that exact moment, pauses it, and draws a marker box with its
  name/confidence — built as a self-contained HTML/JS component since
  Streamlit's own widgets don't support that kind of cross-element
  interactivity. **The same marker also fires automatically off the
  video's own playback clock** (its `timeupdate` event), so signs get
  boxed and labeled in real time as the video plays with no click
  needed — the honest answer to "can signs be recognized during
  playback" turned out to be "not live server-side detection paced to
  live playback" (no reliable way to do that in Streamlit), but
  "replay the already-known results in exact sync with the video's
  real clock," which is both achievable and frame-accurate. "New video"
  resets everything. Capped at 90s of
  processing and 10 minutes of link-download to stay responsive on
  Streamlit Community Cloud's free tier; the trained model file is
  committed to the repo (deliberate exception to the usual gitignore
  rule) since the free host builds straight from GitHub. Tested
  end-to-end locally against a real video, including direct JS
  inspection confirming the seek/pause/marker behavior actually works
  (not just visual spot-checking). Full detail in
  `streamlit_app/README.md` and `ROADMAP.md` section 10.

## Stage 9 video detection results

**Approach**: classical computer vision (color + shape detection in
OpenCV) for localization — not a downloaded pretrained detector.
Deviated from the original plan on purpose, for reasons found by
testing: downloading a third-party detector checkpoint is a real
security risk (unlike the ResNet18 weights, sourced from torchvision's
own trusted channel), and a direct side-by-side test confirmed classical
CV was also simply *better* here — official Ultralytics YOLOv8
(COCO-pretrained) completely missed a clearly-visible STOP sign in a
real test frame, while the color+shape approach found it correctly on
the first try. Code: `video_detection/detector.py` (localization) →
`classifier.py` (wires a crop into the existing model, unmodified) →
`tracker.py` (deduplicates the same sign across frames) →
`process_video.py` (ties it together: sample → detect → classify →
track → annotate).

**Test material**: a real ~3-minute driving-test-track video provided by
the user (portrait 720x1280, ~30fps, Uzbek captions, a permanent channel
watermark overlay) — signs include STOP, pedestrian crossing, warning
triangles, mandatory circles, lane-direction signs, traffic lights.

**Full-video result**: sampled at 4 fps (738 sampled frames from 172s of
video) in 39 seconds; 76 deduplicated tracked signs reported after
tracking (a track is only reported once a majority of its confident
observations agree on the same class — not just one lucky frame).

**Two real tracking bugs found and fixed by testing** (not anticipated
up front): a sign moving fast across the frame can have *zero* box
overlap between consecutive samples at low sample rates — no IoU
threshold can bridge a true zero-overlap gap. Fixed with a
center-to-center-distance fallback match plus raising the sample rate
from 2 to 4 fps (cheap given the pipeline's ~0.05s/frame speed).
Together, these turned 5 fragmented "Stop" tracks and a 3-way-split
fast-moving sign into 2 and 1 clean continuous tracks.

**Honest limitations found, each confirmed by direct inspection, not
assumed**:
1. A channel watermark rendered semi-transparently *on top of* a sign
   (not beside it) causes confident misclassification that no
   box-boundary adjustment can remove, since the contamination is
   *inside* the sign's own pixel region — specific to "blogger"/
   social-media-style footage with baked-in overlays, not plain dashcam
   video. Tried a 12% box-shrink margin as a possible fix; it broke an
   already-correct result elsewhere (a STOP sign dropped from 99.4% to
   22% confidence, wrong) by cutting into the sign's own text — settled
   on a gentler 5% margin instead.
2. Two signs close together can merge into a single detection box; the
   resulting crop confuses the classifier, but this reliably shows up
   as *low* confidence rather than a wrong confident answer — the
   safety net works as intended.
3. Feeding a non-sign red/round object (a traffic light lens) to the
   classifier can produce a moderately confident wrong answer (~65-77%)
   — confidence thresholding alone doesn't catch every such case.
4. Many short tracks reported "Lane directions" repeatedly; spot-checked
   directly rather than assumed to be noise — these are real small
   signs (one visually confirmed mounted on a post next to a painted
   lane arrow on the pavement), but the crops are tiny and blurry, so a
   "confident" score means less on a crop this degraded than on the
   large, clean crops tested earlier in the project.

**Stage 9.5 — second-video honest test pass, done.** A different real
video: a public Instagram reel from a driving-school account
(`avtoshkola_dlya_jenshin`), downloaded via `yt-dlp` with the user's
explicit direction after they supplied the link, portrait 720x1280,
25fps, 123.5s, stored as `video_detection/test_videos/driving_clip_02.mp4`.
Different lighting, route, and camera mount from clip 1 — and, confirmed
by the video's own on-screen station captions (e.g. "3.SVETOFOR",
"7.TEMIR YO'L KESISHMASI"), filmed at a different, larger closed
driving-test facility ("avtodrom") with many numbered practice stations,
each carrying real signage. Full run: 515 sampled frames (~4 fps) in
22.4s, 116 deduplicated tracked signs (higher density than clip 1's 76
in 172s — explained by the many-stations layout, not noise).

Findings, each confirmed by pulling and looking at the actual crop/frame
(gotcha #20), not assumed from the confidence number alone:
1. STOP signs and plain "P" parking signs replicated clip 1's good
   behavior — tracked correctly and consistently across the approach,
   confidence tracking correctness closely on large, clean crops. Not a
   fluke of the first video.
2. **New failure mode: a confidently-wrong merge, not a low-confidence
   one.** This facility mounts two different sign faces stacked
   vertically on one post (e.g. a pedestrian-crossing icon directly
   above a left-turn mandatory-arrow icon). The detector boxes both as
   a single region, and the classifier confidently (97-98%) mislabels
   the combined shape as an unrelated class ("Car washing") —
   reproducibly, at more than one course station. This means clip 1's
   finding #2 above ("merged boxes reliably show up as low confidence")
   is **not a general guarantee** — it held for that specific merge
   shape, not this one; whether a merge reads as low- or
   high-confidence depends on how closely the combined silhouette
   happens to resemble some other trained class, which can't be
   predicted without testing the specific shape.
3. **New, subtler failure mode: icon-only crops can't disambiguate
   near-visual-twin classes that share the same icon.** One box
   captured only the top icon of a multi-element parking sign,
   excluding whatever supplementary plate actually separates "Parking
   (parking space)" from "Regulated parking zone" in the class
   taxonomy (both use the same blue-square "P" icon alone). Classified
   confidently (100%) as one of the two; the icon-only crop genuinely
   doesn't contain enough information to verify which is correct from
   the image alone — which is itself the finding.
4. **Clip 1's watermark-on-sign occlusion did not recur.** This video's
   on-screen captions sit low in the frame (near the road/horizon), not
   over the elevated sign poles, so they never overlap a detected sign
   — confirms that finding was specific to clip 1's channel watermark
   placement, not an inherent flaw in the pipeline.
5. A small, distant red sliver (likely a marker/reflector, not a real
   sign) was picked up at a moderate 65% confidence — same
   non-sign-object pattern as clip 1's finding #3.

**Overall stage 9.5 takeaway**: the honest limitations found on clip 1
are a mix of footage-specific (watermark placement — did not recur) and
genuinely general (confidence isn't a fully reliable safety net for
merged or partially-framed detections — recurred in a *different* form
on clip 2). Two different real-world videos were enough to tell these
apart without needing a much larger test set.

**Stage 9.6 — demo packaging, done.** `video_detection/process_video.py`
already produced the annotated video + report (since 9.4); what was
missing was a clean, easy-to-run interface: replaced raw `sys.argv`
parsing with `argparse` (a real `--seconds` flag, `--help`, a friendly
"video file not found" error), and added a "Quickstart" section to
`video_detection/README.md` with exact commands and what the output
means. Verified both `--help` and a real run still work after the CLI
change. Deliberately **not** built, as a stated scope call rather than
an oversight: a live-camera-feed variant — flagged as optional in
`ROADMAP.md` section 9.6, not a course-project requirement.

**Stage 9 (real-world video detection) is now fully complete, 9.1
through 9.6.**

## Stage 6 model selection results

**Chosen model**: ResNet18, ImageNet-pretrained, fully fine-tuned, same
64x64 letterbox input/split/batch size as the baseline, 15 epochs, mixed
precision. Trained and evaluated on Kaggle Notebooks (Colab's free-tier
GPU quota ran out from earlier attempts before mixed precision was
added). Code in `notebooks/05_transfer_learning.ipynb`.

**Validation accuracy: 0.9903** vs. baseline's 0.8784. **Final held-out
test accuracy: 0.9937** (99.37%) — slightly *above* validation, a good
sign of no overfitting across the iterative comparison process.

**The specific stage-5 confusions were essentially eliminated** (baseline
count -> this model's count, same validation set):
- `44 (Minor road, right)` <-> `45 (Minor road, left)`: 92x -> **0x**
- `184 (Coverage area)` `->183 (Distance to the object)`: 52x -> **1x**
- `184 (Coverage area)` `->193 (Limitation of parking duration)`: 46x -> **0x**
- `189 (Validity period)` `->192 (Paid services)`: 51x -> **2x**
- `190 (Method of parking)` `->196 (Dangerous roadside)` (the single
  biggest baseline confusion): 122x -> **0x**
- `42 (End of overtaking-by-lorries restriction)` `->32 (End of all
  restrictions)`: 85x -> **0x**

This directly confirms stage 5's hypothesis: the baseline's weakness was
insufficient fine-grained visual discrimination (not class imbalance),
and a pretrained backbone fixes it — even at the same 64x64 input size
that stage 6.0's spot-check had flagged as a possible limiting factor for
exactly these plate-style confusions.

**Per-class balance (stage 6.2 criteria, not just overall accuracy)**:
macro-averaged F1 0.9877 on validation (0.9937 on test) sits almost
exactly at overall accuracy, meaning strength is consistent across all
200 classes rather than concentrated in frequent ones. Worst-class
accuracy rose from baseline's 0.179 (class 184) to 0.8033 on validation
/ 0.8846 on test (a different class is now the worst in each split, as
expected once the previously-worst classes were fixed). Accuracy-vs-
training-size correlation stayed near zero (0.0381), confirming this
model is, if anything, even less sensitive to the stage 3.2 class
imbalance than the baseline was.

**Honest caveat for defense**: stage 4 documented that near-duplicate
deduplication isn't perfect (a known, stated limitation). 99%+ accuracy
is a high number, but the improvement is targeted evidence, not generic
inflation — it specifically resolved the exact confusions predicted by
the stage 5 analysis, which is a harder pattern to explain away as
leakage than a uniform accuracy bump would be.

## Stage 5 baseline results

**Overall**: final validation accuracy **87.84%** (200 classes, random
baseline 0.5%). Climbed steadily every single epoch (40.5% -> 87.8% over
15 epochs), no plateau, no overfitting signs (val accuracy stayed at or
above train accuracy throughout, expected given train-only augmentation
and dropout being active only during training).

**Surprising finding — imbalance is not the bottleneck**: correlation
between per-class validation accuracy and training-set size was only
**-0.124** (essentially none). This contradicts the expectation from
stage 3 that smaller classes would be the weakest. The `WeightedRandomSampler`
already in place is handling the imbalance adequately; it is not what's
limiting this baseline.

**What actually explains the weak classes — specific, systematic
confusions between visually/semantically similar sign pairs**, not
scattered random error:
- `44 (Minor road, right)` <-> `45 (Minor road, left)` — 92x. Mirror-image
  confusion.
- `184 (Coverage area)` is the single worst class (17.9% accuracy) — split
  almost entirely between two confusions: `->183 (Distance to the object)`
  (52x) and `->193 (Limitation of parking duration)` (46x), accounting for
  98 of its 134 validation images.
- `189 (Validity period)` `->192 (Paid services)` (51x) — another
  informational/text-plate sign confused with a similar one (recall `189`
  is literally a time-range text plate, per the 3.4 visual audit).
- `190 (Method of parking)` `->196 (Dangerous roadside)` — **122x, the
  single biggest confusion in the whole matrix.** Also explains an
  otherwise-puzzling bright spot near the diagonal in the confusion matrix
  heatmap: class 196 is only 6 columns from 190, so this large off-diagonal
  error visually blended with the diagonal at 200x200 resolution — not a
  data leak or artifact.
- `42 (End of overtaking-by-lorries restriction)` `->32 (End of all
  restrictions)` (85x) — semantically related "end of restriction" signs.

**Takeaway for stage 6**: prioritize things that improve the model's
ability to discriminate fine visual detail (more capacity, or a pretrained
backbone) over further imbalance-focused changes. Worth a quick visual
sanity-check on the `184` confusions specifically, to rule out an actual
labeling issue rather than genuine visual similarity, before assuming it's
purely a model-capacity problem.

## Stage 3 data audit summary

**Class balance**: 200 classes, 116,642 images, 248-1,845 images/class
(mean 583). Real but moderate imbalance — no classes with only a handful
of images, so class weighting plus modest augmentation for smaller
classes should be enough; no drastic measures needed.

**Image properties**: sizes range from 25px to 2,749px wide (median only
191x157 — a long tail of larger images pulls the mean up), so a modest
target resize (e.g. 64x64 or 128x128) makes sense rather than trying to
preserve the outliers' resolution. Aspect ratio is mostly near-square
(median 1.065) with a real tail up to 5.88 for wide/thin signs (e.g. text
plates) — resizing to a square target won't distort most images much.
Color mode is majority RGBA (96,444) vs RGB (20,198), no grayscale — the
alpha channel needs flattening to RGB before training.

**Visual/label quality**: spot-checked 8 random classes x 4 images each,
all internally consistent, no mislabeled folders found.

**Class names**: mapped from the dataset's own `Classes/` folder, saved
at `metadata/class_names.csv`. One data quirk: class 81 has two
conflicting names in the source taxonomy ("Height limit - 3.5" vs "-
4.5") — flagged, doesn't affect training (numeric ID is what's used),
only the human-readable label.

**Data leakage risk (the important one)**: filenames suggest multiple
photos of the same physical sign exist as near-duplicate frames (e.g.
`00000_00000_00017.png` = class/track/frame). Confirmed real for the
~10.3% of files that cleanly match this pattern — ~900 (class, track)
groups, averaging ~13 images each, ~82% with 5+ images, visually
confirmed as the same sign photographed moments apart. However, filename
parsing only reliably covers about 10% of the dataset; the rest use other
naming conventions, consistent with this dataset being merged from
multiple original sources. **Decision for stage 4.1**: build near-
duplicate groups using content-based detection (e.g. perceptual image
hashing) across the whole dataset, not filename parsing alone, then split
train/val/test by group — never by individual image — to avoid leaking
near-identical frames across the split.

## Stage 4 known limitation (stated honestly)

The near-duplicate grouping is fairly strict (~2.6 images/group on
average, vs. ~13/group for the filename-confirmed tracks in 3.6), so it
reliably catches exact/near-exact duplicates but may miss some more
visually-different frames of the same physical sign (e.g. a longer zoom
sequence). It meaningfully reduces leakage risk versus naive per-image
splitting, but is not a perfect guarantee — worth mentioning as a known
limitation if asked during defense.

## Current task

Stage 9 (9.1-9.6), stage 10 (Streamlit web demo), and stage 8.1/8.3 are
now all complete. The only remaining open item in the whole project is
8.2 (`README.md`) — and actually deploying the Streamlit app to
share.streamlit.io, which only the user can do (needs their own GitHub
login).

## Next

- 8.2 Update `README.md` with how to reproduce and the main results (the
  project report `.docx` and presentation `.pptx` already cover this
  narratively, but `README.md` itself — the GitHub-facing doc — is
  still the original stub)

## Known problems / blockers

- Colab's free-tier GPU quota was exhausted during stage 6.1 (from
  earlier long pre-mixed-precision training attempts); worked around by
  moving that notebook to Kaggle rather than waiting it out. Quota status
  wasn't rechecked since, so assume it may still be limited if a future
  stage wants to use Colab again.
- **Real-world limitation found via live bot testing**: a wide street-scene
  photo (sign small within a lot of background — trees, road, crosswalk)
  was misclassified, but with notably low confidence (42.7%, vs. 99%+ on
  every other tested photo) rather than confidently wrong. Likely cause:
  the training dataset's photos are tighter, sign-focused shots, so a
  whole-scene photo shrinks the actual sign to a small, cluttered patch
  after the 64x64 resize — a framing/domain-gap issue, not a sign of poor
  model accuracy. This is direct, real-world evidence for why the future
  plan (Section 10 of the project report / the presentation's future-vision
  slide) includes a detection step to locate and crop the sign *before*
  classifying it, rather than classifying a whole photo directly. **This
  is exactly what stage 9 (`video_detection/`) now addresses** — see
  "Stage 9 video detection results" above for what's been built and
  found so far.
