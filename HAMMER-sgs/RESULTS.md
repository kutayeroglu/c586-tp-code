# Results

Runs of the SGS pipeline on MareNostrum5 with `slurm_scripts/mn5/sgs.sh`.
Output files are in `/gpfs/projects/etur166/boga222803/hammer-sgs/results/<jobid>/`. Scores use `scripts/score_sgs.py`.

## Replication of the SGS paper's Table 1

Job 49469599 (2026-11-27): `SPLIT=all`, `STAGE=all`, `--qos=acc_ehpc`. It took 37 minutes on 1 H100; most of that is the semantic stage, which runs on CPU.
Files: `semantic_blip_all.csv`, `scores_all.txt`.

The evaluation is one-class: every sample is FG–BG inconsistent. So Acc is the fraction labelled Mismatch at τ = 0.55, and F1 = 2·Acc/(1+Acc).

| | N | s~FB mean±sd | median | % Mismatch | Acc / F1 |
|---|---|---|---|---|---|
| Paper (Table 1) | 4632 | 0.264±0.158 | 0.238 | 94.3 | 0.943 / 0.971 |
| Job 49469599 | 4632 | 0.265±0.157 | 0.239 | 94.6 | 0.946 / 0.972 |

N matches exactly, and the score statistics agree to within 0.001. Acc is 0.3 points higher, which is about 14 of 4632 samples. Two likely causes:
- For each shared id (see below), which image is kept depends on record order. Ours is train → val → test, which may differ from the authors' original `metadata.json`.
- Our image has newer packages than the authors probably used (transformers 5.21.0, ultralytics 8.4.190), which can change BLIP captions and YOLO masks slightly.

The paper's numbers were read from the arXiv HTML version (https://arxiv.org/html/2509.26039, Table 1), not the PDF. The paper reports no HAMMER+SGS results, so this table is the only SGS result to compare against.

## Where N = 4632 comes from

The paper's "Extended DGM4" set is DGM4plus: 5000 samples (2000 origin, 1500 text_attribute, 1500 text_swap). Its N = 4632 is the number of distinct `id` values among them.

- Crops are named `<id>.png`, and the upstream crops script skips an id whose crops already exist.
- 368 ids are each shared by two different images, all of them text-manipulated (text_attribute / text_swap). The second image of each is skipped: 5000 − 368 = 4632.
- Running the splits separately gives more pairs, because some shared ids fall in different splits:

| Split | Samples | Distinct ids | Shared ids |
|---|---|---|---|
| train | 3500 | 3314 | 186 |
| val | 750 | 737 | 13 |
| test | 750 | 743 | 7 |
| per split, total | 5000 | 4794 | |
| all at once | 5000 | 4632 | 368 |

Neither paper (SGS, arXiv 2509.26039; DGM4plus, arXiv 2509.26047), nor the HAMMER-sgs repo, nor the DGM4plus site mentions the shared ids.
They likely come from DGM4plus's generator (`src/Datasetgen.py` in `Gaganx0/DGM4plus`):
- It assigns an id from a counter before generating the image.
- It doesn't give the id back when the sample is rejected (face check, text, pHash dedup).
- After a resumed run, it restarts the counter at `1 + len(existing_records)` (line 1644), which is below the highest id already used.

The released ids carry prefixes (`100<n>` origin, `200<n>` text) that the generator doesn't add, so connecting the two is an inference.

To match the paper, `SPLIT=all` merges the three splits into one JSON and one crops folder. It reports `OK:4632 Skipped:368 Missed/failed:0`.

## Other runs

- Job 49468337 (2026-11-27): debug run with `SPLIT=val` and `SEMANTIC_ARGS="--max_images 50"`, on `acc_debug`, 1 min 22 s. Crops: `OK:737 Skipped:13`. All 50 rows were labelled Mismatch, which is expected since every sample is inconsistent.
