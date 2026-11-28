# Experiment plan: robustness of SGS to image degradations

Status: plan, nothing implemented yet. Results go to `RESULTS.md` as they come in.

## Research question

How do blur, noise and warp applied to the foreground (FG), the background (BG) or both change SGS decisions, and why?
Does replacing BLIP-base with BLIP2, with or without instructions to describe the person and the place and to omit image quality, reduce these effects?

## Hypotheses

- **H1, one-sided degradation (FG only or BG only):** quality words ("blurry", "grainy", "distorted") appear in only one caption. FG–BG similarity drops, so consistent pairs flip to Mismatch (false alarms).
- **H2, degradation on both:** both captions share quality words, so similarity rises. Inconsistent pairs flip to Match (misses).
- **H3:** the effect grows with severity and is strongest for noise and blur. Warp distorts shapes rather than adding texture, so it leaks fewer quality words.
- **H4:** BLIP2 without a prompt leaks less than BLIP-base. An instruction to omit image quality reduces leakage further. Steering FG and BG with different prompts ("person" vs "place") changes the score scale on its own, independent of degradation.

## Why negatives are needed

All of DGM4plus is inconsistent, so on DGM4plus alone Acc is just the share labelled Mismatch, and F1 = 2·Acc/(1+Acc). Any degradation that lowers similarity then looks like an improvement (H1), and false alarms can't be seen at all.

So each condition is evaluated on a **pooled set**:
- positives: DGM4plus (label 1);
- negatives: DGM4 pristine samples (label 0);
- both degraded in the same way.

| Set | Clean | Degraded |
|---|---|---|
| DGM4plus (positives) | done for BLIP-base (job 49469599) | planned |
| DGM4 pristine (negatives) | planned | planned |

The paper's one-class Table 1 numbers on DGM4plus alone are still reported, for comparability (`scripts/score_sgs.py`).

## Data

### Positives: DGM4plus
`SPLIT=all`, 4632 pairs (the crops from the replication setup; see `RESULTS.md`).

### Negatives: DGM4 pristine subset
- **Size:** 4632 samples, so the classes are balanced. Sampled with a fixed seed.
- **Source:** DGM4 from the HAMMER release. Only the pristine (original) images and their metadata are needed, not the full dataset.
  - To verify: the download location, and how pristine samples are labelled (`fake_cls`).
- **Matching DGM4plus:**
  - keep images with 1–3 faces, using `mtcnn_boxes` if DGM4's metadata has it (to verify), like DGM4plus's face rule;
  - resize or center-crop to 400×256, the DGM4plus resolution.
- **Storage:** one tar at `/gpfs/projects/etur166/boga222803/datasets/dgm4_pristine.tar`, with the same layout as `dgm4plus.tar`.
- **Known confound:** DGM4plus images are synthetic (gpt-image-1) and DGM4 images are real news photos. Part of any score difference can come from synthetic vs real rather than from consistency. Report this as a limitation, and compare score distributions on clean data first.

## Degradations

The main setting applies them to the **crops, after segmentation**. This isolates the captioner (the "why"), and YOLO isn't affected.

- **Only inside the mask.** Pixels outside the mask stay black, so neither noise nor blur creates new content in the black area.
- **Needs saved masks** (see "Masks" below). The crops script doesn't save them today.

### Masks

- **Why save them:** mainly correctness. Treating black crop pixels as "outside the mask" doesn't work, because 67% of DGM4plus originals (3343 of 5000) contain genuinely black pixels (0.11% of all pixels, e.g. dark clothes, shadows). Noise would skip those pixels and blur would treat them as boundary. Speed matters less: YOLO takes ~2 minutes for all 5000 images.
- **When:** YOLO runs once more, with `--save_masks`, writing new crops and masks together, so each mask matches its crop exactly. All degradation conditions then read these fixed crops and masks, and nothing is segmented again.
- **What is saved:** the mask actually used for each image, including the center-square fallback when no person is found. One mask per crop, aligned with it:
  - `masks/fg/<id>.png`: FG crop size;
  - `masks/bg/<id>.png`: full image size.
  Masks are named by id like the crops, so the same rule applies to shared ids (the first image in the JSON is kept). Binary PNGs, tens of MB in total.
- **Per-image metadata,** written alongside, for the analysis: mask area fraction, number of people merged, and whether the fallback was used. This lets us check whether degradation effects depend on foreground size.
- **The flag only adds output:** the crops must be the same with or without `--save_masks`.
- **Check against the replication:** compare the new `crops/all` with the replication's crops byte for byte.
  - First record checksums of the current `crops/all` (one sha256 per file, saved to `results/49469599/crops_sha256.txt`). The crops stage of `sgs.sh` deletes `crops/all` before writing, so the replication crops are gone once it runs.
  - If identical, job 49469599 still holds as the clean BLIP-base baseline.
  - If they differ (YOLO on GPU may not be bit-for-bit deterministic), rerun the clean baseline once on the new crops, so every condition starts from the same crops.
- **Targets:** FG, BG, or both.
- **Severities:** 3 levels each, plus clean. The values below are starting points, to be tuned in the pilot.

| Degradation | Implementation | Severities (starting point) |
|---|---|---|
| Blur | Gaussian, masked normalized convolution: `blur(img·m) / blur(m)` | σ = 1, 2, 4 px |
| Noise | additive Gaussian on 0–255, inside the mask, clipped | std = 10, 25, 50 |
| Warp | smooth random displacement field (Gaussian-filtered noise), applied to the crop and its mask together | max displacement = 2%, 5%, 10% of the crop's shorter side |

- **Seeds:** each image's seed comes from `(id, degradation, severity, target)`, so every run gives the same degraded crops.
- **Conditions:** 1 clean + 3 degradations × 3 severities × 3 targets = **28 per dataset**.
- **Secondary setting (realistic):** degrade the **full image before YOLO**, for "both" only, at the middle severity. Report how often YOLO falls back to the center square, since that mixes segmentation failures with captioning effects.

## Captioners

| Id | Model | Prompt (FG / BG) | Purpose |
|---|---|---|---|
| C0 | BLIP-base (`Salesforce/blip-image-captioning-base`) | none | paper baseline |
| C1 | BLIP2-FlanT5-XL (`Salesforce/blip2-flan-t5-xl`) | none | effect of the stronger captioner |
| C2 | BLIP2-FlanT5-XL | "Describe the person or people. Do not mention image quality, blur, noise or distortion." / "Describe the place. Do not mention image quality, blur, noise or distortion." | the instruction, as asked |
| C2b | BLIP2-FlanT5-XL | the same neutral prompt for FG and BG: "Describe what is in this image. Do not mention image quality, blur, noise or distortion." | separates the "omit quality" effect from the "person vs place" steering effect |

- **One BLIP2 model for C1, C2 and C2b,** so the instruction effect isn't mixed with a model change.
- **Why BLIP2-FlanT5-XL (decided):**
  - It answers the research question as written, which names BLIP2.
  - It captions natively without a prompt, so C1 is a true no-instruction baseline.
  - Its language model (FlanT5) is instruction-tuned, so C2 and C2b can be tested. BLIP2-OPT isn't, and follows prompts poorly.
- **Not used: InstructBLIP.** It always expects a prompt, so it has no real no-instruction mode, and it would change the question to "does an instruction-tuned captioner help?". If FlanT5-XL ignores the "omit quality" instruction, report that as a finding rather than switching models for C2, which would mix the instruction effect with a model change.
- **Caption length:** the same `max_new_tokens` for every captioner, since length affects MiniLM similarity. The paper uses 16. Whether 16 cuts off BLIP2's answers is a pilot check.
- **Similarity:** MiniLM (`all-MiniLM-L6-v2`) stays the same everywhere.
- **Hardware:** BLIP2 runs on GPU in fp16. C0 also moves to GPU for speed (see "Checks").
- **To fetch:** the BLIP2-FlanT5-XL weights. The local cache only has its `config.json`, so download locally and upload to `hf_cache` on MN5.

## Metrics

### Detection, per condition, on the pooled set
- **AUROC:** threshold-free, so it isn't affected by score-scale shifts between captioners.
- **Acc / F1 at τ = 0.55:** the paper's threshold.
- **Acc / F1 at τ\* per captioner:**
  - Split positives and negatives 20% / 80% (calibration / evaluation) with a fixed seed.
  - On clean calibration data, choose τ\* to maximise F1.
  - Freeze τ\* and report on the evaluation part for every condition.
- **One-class Table 1 numbers** on DGM4plus alone, for comparison with the paper.
- **95% CIs:** paired bootstrap, 1000 resamples, using the same resampled ids for both conditions being compared.

### Explanatory analysis
- **Decision flip rate** against the same captioner on clean data, which answers "change SGS decisions" most directly:
  - overall, per class, and per direction (Match→Mismatch, Mismatch→Match);
  - McNemar test against clean.
- **Score shift:** per-pair Δs = s(degraded) − s(clean), with mean and CI, plus distribution plots per condition.
- **Quality-term leakage:** share of captions containing any word from a fixed list.
  - Quality words: blurry, blur, blurred, out of focus, fuzzy, noisy, noise, grainy, grain, static, pixelated, distorted, warped, wavy, low quality, low resolution, poor quality, artifact.
  - Mask words, counted separately: black background, dark background, silhouette, silhouetted, shadow. Clean crops already produce these; one clean DGM4plus caption reads "a couple is silhouetted against a train window".
  - Link to decisions: compare flip rates when leakage is in neither caption, in one, or in both. This tests H1 and H2 directly.
- **Noun overlap:** Jaccard similarity of the lemmatized noun sets of the FG and BG captions. Report its change from clean and its correlation with s.

## Code changes

The upstream scripts stay as they are, so the replication (job 49469599) can always be rerun. The new pieces:

1. **`make_fg_bg_crops_yolo.py`:** an opt-in `--save_masks` flag that writes the FG and BG masks and the per-image metadata (see "Masks"). The crops themselves don't change.
2. **`scripts/degrade_crops.py`:** crops + masks + condition → degraded crops, with the deterministic seeds above.
3. **`scripts/caption_crops.py`:** captions a folder of crops.
   - Options: BLIP-base or BLIP2, `--device cuda`, `--prompt`, batching.
   - Writes one CSV per (crop folder, captioner, prompt).
   - Caching: in FG-only conditions the BG crop is the clean one, so its clean caption is reused (and the same for BG-only). That cuts captioning about in half.
4. **`scripts/score_pairs.py`:** joins FG and BG captions by id, computes s with MiniLM, and writes the same CSV columns as `semantic_fg_bg_blip.py`.
5. **`scripts/eval_degradation.py`:** pooled metrics, τ\* calibration, CIs, flips, leakage and noun overlap, written as tables for `RESULTS.md`.
6. **`slurm_scripts/mn5/degrade.sh`:** runs the grid as a SLURM job array (one task per dataset × condition × captioner), following `sgs.sh`'s conventions.
7. **Container:** add spaCy and `en_core_web_sm` (noun lemmas) and `accelerate`. Layer them on the existing `hammer-sgs.sif` (`Bootstrap: localimage`) rather than rebuilding, so the replication environment doesn't change.

## Compute estimate (rough, to be measured in the pilot)

Each degradation × severity needs, per dataset:
- 4632 FG captions (FG only; BG reused from clean);
- 4632 BG captions (BG only);
- 9264 captions (both).

That's ≈ 18.5k captions. For 9 degradation × severity combinations plus clean, it's ≈ 176k captions per dataset, or **≈ 350k per captioner** for both datasets.

- **C0 on CPU:** ~4 captions/s from the replication run, so about a day. Hence the move to GPU with batching; expect under 1–2 h.
- **BLIP2-FlanT5-XL, fp16, batched on an H100:** a guess of 20–40 captions/s, so roughly 3–5 h per variant, and 10–15 h for C1, C2 and C2b together.
- **SLURM:** split the work into array tasks to stay within `acc_ehpc` time limits.

## Checks before the full grid

- **Crops made with `--save_masks` match the replication's crops** byte for byte; otherwise rerun the clean baseline on them (see "Masks").
- **C0 on GPU must reproduce the replication:** run clean DGM4plus with C0 on GPU and compare with job 49469599. Expect (almost) no label flips. If it differs, keep C0 on CPU for the comparison runs.
- **Masked degradation stays inside the mask:** check the pixels outside the mask are still 0.
- **Captions are sensible at the strongest severity:** look at 20 samples per condition.
- **The prompt isn't echoed:** C2 and C2b answers don't repeat the prompt text, and 16 tokens doesn't cut them off.

## Phases

1. **Prerequisites:** the DGM4 pristine subset (download, filter, tar, upload), BLIP2-FlanT5-XL weights, and the spaCy layer on the image.
2. **Code:** items 1–6 above.
3. **Pilot:** 200 positives + 200 negatives, all 28 conditions, C0 and C2. Tune the severities, measure throughput, and run the checks above.
4. **Full grid with C0,** then full grids with C1, C2 and C2b.
5. **Secondary setting:** full-image degradation before YOLO ("both", middle severity), with the fallback rate.
6. **Analysis and write-up** in `RESULTS.md`.

## Open decisions

- **`max_new_tokens`:** 16 for all (paper-faithful), or a larger shared value if 16 cuts off BLIP2.
- **Final severities:** after the pilot.
- **Image-level setting:** whether to include it, and at which severities.
- **Negative-set filters:** faces vs YOLO persons, and resize vs crop to 400×256.
