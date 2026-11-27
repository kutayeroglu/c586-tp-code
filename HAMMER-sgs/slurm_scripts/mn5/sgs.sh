#!/bin/bash
#SBATCH --job-name=hammer-sgs
#SBATCH --qos=acc_debug
#SBATCH --account=etur166
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=%j_hammer-sgs.out
#SBATCH --error=%j_hammer-sgs.err

# HAMMER-SGS pipeline on MareNostrum5: YOLOv8-seg FG/BG crops, then BLIP + MiniLM semantic check.
# Submit it by running it (not with sbatch) from anywhere on a login node:
#   bash <path to>/HAMMER-sgs/slurm_scripts/mn5/sgs.sh [extra sbatch options, e.g. --qos=acc_ehpc]
# Settings are passed as environment variables:
#   STAGE=crops    bash .../sgs.sh         # YOLOv8-seg crops only (uses the GPU)
#   STAGE=semantic bash .../sgs.sh         # BLIP + MiniLM only (CPU-only), needs the crops of that split
#   SPLIT=train    bash .../sgs.sh         # train (3500) | val (750, default) | test (750)
#   SEMANTIC_ARGS="--max_images 200" bash .../sgs.sh   # quick smoke test
#   CROPS_ARGS="--overwrite" bash .../sgs.sh           # extra options for make_fg_bg_crops_yolo.py
# Crops:   /gpfs/projects/etur166/boga222803/hammer-sgs/crops/<split>/{fg,bg}/ (kept across jobs)
# Results: /gpfs/projects/etur166/boga222803/hammer-sgs/results/<jobid>/
# Logs:    /gpfs/scratch/etur166/boga222803/hammer-sgs/logs/<jobid>_hammer-sgs.{out,err}

set -e

SCRATCH_BASE="/gpfs/scratch/etur166/boga222803/hammer-sgs"
LOG_DIR="$SCRATCH_BASE/logs"

# -- Submission (login node): --chdir must exist before the job starts --
if [ -z "$SLURM_JOB_ID" ]; then
    SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
    mkdir -p "$LOG_DIR"
    exec sbatch --chdir="$LOG_DIR" --export=ALL,SRC_DIR="$SRC_DIR" "$@" "$0"
fi

# -- Job (compute node) --
if [ -z "$SRC_DIR" ] || [ ! -f "$SRC_DIR/scripts/make_fg_bg_crops_yolo.py" ]; then
    echo "SRC_DIR not set or has no scripts/make_fg_bg_crops_yolo.py; submit with 'bash slurm_scripts/mn5/sgs.sh', not sbatch" >&2
    exit 1
fi

STAGE="${STAGE:-all}"   # crops | semantic | all
SPLIT="${SPLIT:-val}"   # train | val | test

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -- Paths --
PROJECTS_BASE="/gpfs/projects/etur166/boga222803"
SIF_IMAGE="$PROJECTS_BASE/hammer-sgs.sif"
DATASET_TAR="$PROJECTS_BASE/datasets/dgm4plus.tar"   # top-level folders: train/ val/ test/
YOLO_WEIGHTS="$PROJECTS_BASE/models/yolov8s-seg.pt"
HF_CACHE="$PROJECTS_BASE/hf_cache"                    # BLIP and MiniLM, see CLAUDE.md
CROPS_DIR="$PROJECTS_BASE/hammer-sgs/crops/$SPLIT"    # one per split: crops are named by id
OUT_DIR="$PROJECTS_BASE/hammer-sgs/results/$SLURM_JOB_ID"
RUN_DIR="$SCRATCH_BASE/$SLURM_JOB_ID"
DATA_ROOT="$TMPDIR/dgm4plus"

# Layout: <split>/<split>.json (test: test_public.json) with bare file names under <split>/images/
case "$SPLIT" in
    train|val) DATA_JSON="$DATA_ROOT/$SPLIT/$SPLIT.json" ;;
    test)      DATA_JSON="$DATA_ROOT/test/test_public.json" ;;
    *) echo "ERROR: unknown SPLIT '$SPLIT' (expected train, val or test)" >&2; exit 1 ;;
esac
IMG_ROOT="$DATA_ROOT/$SPLIT/images"

echo "--- Job $SLURM_JOB_ID on $(hostname), GPU(s): $CUDA_VISIBLE_DEVICES (stage: $STAGE, split: $SPLIT) ---"
echo "Source: $SRC_DIR"
echo "Run dir: $RUN_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# -- Data: extract to local SSD (only needed for the crops stage) --
if [ "$STAGE" = "crops" ] || [ "$STAGE" = "all" ]; then
    mkdir -p "$DATA_ROOT"
    tar -xf "$DATASET_TAR" -C "$DATA_ROOT"
    echo "Extracted $(ls "$IMG_ROOT" | wc -l) images to $IMG_ROOT"
fi

# -- Run folder on scratch; only the code the job runs --
mkdir -p "$RUN_DIR" "$CROPS_DIR"
rsync -a "$SRC_DIR/scripts" "$RUN_DIR/"
cd "$RUN_DIR"

# -- Container execution; nodes have no internet, so models come from project storage --
module purge
module load singularity/4.1.5
run() {
    singularity exec --nv --bind /gpfs --bind "$TMPDIR" \
        --env HF_HOME="$HF_CACHE" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        --env YOLO_OFFLINE=1 --env YOLO_CONFIG_DIR="$RUN_DIR/.ultralytics" \
        "$SIF_IMAGE" "$@"
}

run python -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"

# Split an args string by spaces and append it to the named array
append_args() {
    local -n arr=$1
    if [ -n "${2:-}" ]; then
        read -ra EXTRA_ARRAY <<< "$2"
        arr+=("${EXTRA_ARRAY[@]}")
    fi
}

if [ "$STAGE" = "crops" ] || [ "$STAGE" = "all" ]; then
    echo "--- Stage 1: FG/BG crops (YOLOv8-seg) ---"
    # Existing crops are skipped unless --overwrite is passed, so a timed-out job can be resubmitted
    CROPS_CMD=(
        --data "$DATA_JSON"
        --img_root "$IMG_ROOT"
        --out_dir "$CROPS_DIR"
        --model "$YOLO_WEIGHTS"
        --imgsz 896
    )
    append_args CROPS_CMD "${CROPS_ARGS:-}"
    run python scripts/make_fg_bg_crops_yolo.py "${CROPS_CMD[@]}"
fi

if [ "$STAGE" = "semantic" ] || [ "$STAGE" = "all" ]; then
    echo "--- Stage 2: semantic FG-BG check (BLIP + MiniLM, CPU) ---"
    # Note: the CSV is only written at the end, so a timeout loses all progress of this stage
    SEMANTIC_CMD=(
        --fg_dir "$CROPS_DIR/fg"
        --bg_dir "$CROPS_DIR/bg"
        --out_csv "results/semantic_blip_${SPLIT}.csv"
        --threshold 0.55
    )
    append_args SEMANTIC_CMD "${SEMANTIC_ARGS:-}"
    run python scripts/semantic_fg_bg_blip.py "${SEMANTIC_CMD[@]}"

    # -- Keep results in project storage (scratch is purged) --
    mkdir -p "$OUT_DIR"
    cp -r results/. "$OUT_DIR/"
    echo "--- Results copied to $OUT_DIR ---"
fi

# -- Remove the run folder; only reached on success, failed runs keep it for debugging --
cd "$SCRATCH_BASE"
rm -rf "$RUN_DIR"
echo "--- Job finished; removed $RUN_DIR ---"
