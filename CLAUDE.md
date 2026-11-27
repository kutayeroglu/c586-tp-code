# Information about the HPC server

I am working on a remote HPC server called MareNostrum5.

To access the server I use `ssh mn5`.
For data transfer tasks I need to use another node: `mn5_data_transfer1`.
Nodes don't have internet access.

The project code is `etur166` (use it for `--account` and for `/gpfs/projects/...` and `/gpfs/scratch/...` paths).

## Paths
The DGM4plus dataset is stored as one tar file at `/gpfs/projects/etur166/boga222803/datasets/dgm4plus.tar` (train 3500, val 750, test 750 images; see "Large files" below).

Datasets must always be extracted to `$TMPDIR` inside the job, never next to the source code:
```bash
mkdir -p "$TMPDIR/dgm4plus"
tar -xf /gpfs/projects/etur166/boga222803/datasets/dgm4plus.tar -C "$TMPDIR/dgm4plus"
```

The repo is cloned at `/gpfs/home/boga/boga222803/c586-tp-code`, so this project is at `/gpfs/home/boga/boga222803/c586-tp-code/HAMMER-sgs`.
This is storage only: never execute jobs there or use it as a job's working directory (see "Where jobs run").

### Getting the source code onto MN5
MN5 can't reach GitHub, so the repo is cloned locally into an `sshfs` mount of the MN5 home directory (git runs locally, the files land on MN5):
```bash
mkdir -p ~/mn5_mount/home
sshfs -o workaround=rename boga222803@transfer1.bsc.es:/gpfs/home/boga/boga222803/ ~/mn5_mount/home
git clone git@github.com:kutayeroglu/c586-tp-code.git ~/mn5_mount/home/c586-tp-code   # first time
git -C ~/mn5_mount/home/c586-tp-code pull                                               # later updates
fusermount -u ~/mn5_mount/home                                                          # unmount when done
```
Only the source goes through git; large files are uploaded separately with `rsync -a --partial` through `mn5_data_transfer1` (see "Large files" below).
Server rules require all large files (weights, images, datasets) to stay in `/gpfs/projects/etur166/boga222803`, never in the source code directory.

Large files in `/gpfs/projects/etur166/boga222803`:
```
├── hammer-sgs.sif              # Apptainer image (see "Running the container")
├── datasets/dgm4plus.tar       # DGM4plus: train/ val/ test/, each <split>.json (test: test_public.json) + images/
├── models/yolov8s-seg.pt       # YOLOv8-seg weights for the crops stage
├── hf_cache/hub/               # HF cache (HF_HOME): Salesforce/blip-image-captioning-base, sentence-transformers/all-MiniLM-L6-v2
└── hammer-sgs/                 # job outputs: crops/<split>/, results/<jobid>/
```
They were staged locally and uploaded through the transfer node:
```bash
tar -cf dgm4plus.tar -C ~/datasets/dgm4_plus train val test   # one file instead of ~5000 on GPFS
mkdir -p hf_cache/hub && cp -a ~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base \
    ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 hf_cache/hub/   # relative symlinks survive
# yolov8s-seg.pt: downloaded with ultralytics' attempt_download_asset('yolov8s-seg.pt') inside the image
rsync -a --partial <file or folder> mn5_data_transfer1:/gpfs/projects/etur166/boga222803/<target>/
```
Keep datasets as a single archive on GPFS (many small files are slow there) and extract them to `$TMPDIR` in the job.

## Where jobs run
The server maintainers recommend: "Executing your jobs in your group's /gpfs/projects or /gpfs/scratch directories is recommended."
So jobs run in `/gpfs/scratch/etur166/boga222803/`, never in `/gpfs/home`: both the job's working directory (`--chdir`, where the SLURM `.out`/`.err` logs go) and the folder the code runs in are on scratch.
`/gpfs/scratch/etur166/` is shared with the group; keep everything under `boga222803/`. Scripts create the folders they need with `mkdir -p`; don't leave folder creation as a manual step.

Each job copies only the code it needs from home into its own run folder on scratch and runs there:
```bash
RUN_DIR=/gpfs/scratch/etur166/boga222803/hammer-sgs/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
rsync -a "$SRC_DIR/scripts" "$RUN_DIR/"   # SRC_DIR = the HAMMER-sgs/ source folder on home
cd "$RUN_DIR"
```
After the job, copy the results to `/gpfs/projects/etur166/boga222803/hammer-sgs/results/$SLURM_JOB_ID/` so they are kept, then delete the run folder. A failed job keeps its run folder for debugging.

The code gets its models by absolute paths and env vars, all on `/gpfs/projects` (so `/gpfs` must be visible inside the container, `--bind /gpfs`, see below):
- YOLO: `--model /gpfs/projects/etur166/boga222803/models/yolov8s-seg.pt`, with `YOLO_CONFIG_DIR` pointing into the run folder for Ultralytics' settings file.
- BLIP and MiniLM: `HF_HOME=/gpfs/projects/etur166/boga222803/hf_cache`.

## Running the container
MN5 has SingularityCE, not Apptainer. Load it in the job:
```bash
module purge
module load singularity/4.1.5
```
`singularity exec --nv` works on compute nodes but fails on the login node (session directory permission error), so test containers inside a job.

The image is `/gpfs/projects/etur166/boga222803/hammer-sgs.sif`. MN5 has no internet, so it is built locally with Apptainer 1.5.3 from the local Docker image `hammer-sgs:latest` (built from `HAMMER-sgs/Dockerfile`, the same Dockerfile CI pushes to `ghcr.io/kutayeroglu/hammer-sgs`), without downloading anything, then uploaded through the transfer node:
```bash
apptainer build ~/containers/hammer-sgs.sif docker-daemon://hammer-sgs:latest
rsync -a --partial ~/containers/hammer-sgs.sif mn5_data_transfer1:/gpfs/projects/etur166/boga222803/hammer-sgs.sif
```
If the local Docker image is missing, `docker pull ghcr.io/kutayeroglu/hammer-sgs:latest` first (the ~3.7 GB torch layer is slow to pull on the home network), or rebuild it from the Dockerfile.
Current image: torch 2.14.1+cu126, transformers 5.21.0, ultralytics 8.4.190 (Python 3.11).
For small additions, prefer layering on top of an existing verified image (`Bootstrap: localimage`) over a full rebuild: a full rebuild re-resolves every unpinned package and could change results.

Run the pipeline from the run folder, offline, with the dataset in `$TMPDIR` (see `slurm_scripts/mn5/sgs.sh` for the full commands):
```bash
singularity exec --nv --bind /gpfs --bind "$TMPDIR" \
  --env HF_HOME=/gpfs/projects/etur166/boga222803/hf_cache --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env YOLO_OFFLINE=1 --env YOLO_CONFIG_DIR="$RUN_DIR/.ultralytics" \
  /gpfs/projects/etur166/boga222803/hammer-sgs.sif python scripts/make_fg_bg_crops_yolo.py ...
```

## GPU selection
Don't hardcode GPU IDs. SLURM sets `CUDA_VISIBLE_DEVICES` for the job; use it.

# Preparing SLURM scripts
The SLURM script for this project is `HAMMER-sgs/slurm_scripts/mn5/sgs.sh` (adapted from the bounhpc one, `slurm_scripts/bounhpc/sgs.sh`); start from it for new jobs.
It is submitted by running it (not with `sbatch`) from anywhere on a login node. It creates the scratch folders, then submits itself with `--chdir` on scratch and passes extra args to `sbatch`:
```bash
bash /gpfs/home/boga/boga222803/c586-tp-code/HAMMER-sgs/slurm_scripts/mn5/sgs.sh                  # debug run (acc_debug)
bash /gpfs/home/boga/boga222803/c586-tp-code/HAMMER-sgs/slurm_scripts/mn5/sgs.sh --qos=acc_ehpc   # full run
STAGE=crops SPLIT=train bash .../sgs.sh   # STAGE: crops | semantic | all; SPLIT: train | val | test | all
```
`SPLIT=all` is the setup to use when comparing with the SGS paper. Its "Extended DGM4" set is DGM4plus, and its N=4632 is the number of distinct ids among the 5000 samples: crops are named by id, 368 ids are shared by two different images, and the upstream crops script skips the second one. So `SPLIT=all` merges train, val and test (in that order) into one JSON and one crops folder and should report `OK:4632 Skipped:368`. Per-split runs give 4794 pairs instead, because some shared ids fall in different splits. The ids come from a counter in DGM4plus's generator (`src/Datasetgen.py` in `Gaganx0/DGM4plus`) that restarts at the record count after a resumed run, so it can reuse ids. Which image of each pair is kept depends on record order, which may differ from the authors' original `metadata.json`.
`--chdir` must exist before the job starts, so it can't be created inside the job.
Compute nodes are offline, so the job runs with `HF_HUB_OFFLINE=1` and `YOLO_OFFLINE=1`, reading models from project storage.

Logs: `/gpfs/scratch/etur166/boga222803/hammer-sgs/logs/<jobid>_hammer-sgs.{out,err}`.
Crops: `/gpfs/projects/etur166/boga222803/hammer-sgs/crops/<split>/{fg,bg}/`, kept across jobs so `STAGE=semantic` can reuse them and a timed-out crops job can resume.
Results: `/gpfs/projects/etur166/boga222803/hammer-sgs/results/<jobid>/semantic_blip_<split>.csv`.

## QoS
- Debug and test jobs: `--qos=acc_debug`.
- Full runs: `--qos=acc_ehpc`.

Don't use `--chdir=.` in new scripts: it would run the job wherever it was submitted from.
