#!/bin/bash

#SBATCH --job-name=hammer-sgs
#SBATCH --output=logs/hammer-sgs-%j.out
#SBATCH --error=logs/hammer-sgs-%j.err

#SBATCH --container-image ghcr.io\#kutayeroglu/hammer-sgs
#SBATCH --container-mounts /stratch/dataset:/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH --time=04:00:00

# Usage (submit from the HAMMER-sgs dir; Slurm will not create logs/ for you):
#   mkdir -p logs
#   sbatch slurm_scripts/bounhpc/sgs.sh                        # both stages
#   STAGE=crops    sbatch slurm_scripts/bounhpc/sgs.sh         # YOLOv8-seg crops only (uses the GPU)
#   STAGE=semantic sbatch slurm_scripts/bounhpc/sgs.sh         # BLIP + MiniLM only (CPU-only)
#   SEMANTIC_ARGS="--max_images 200" sbatch slurm_scripts/bounhpc/sgs.sh   # quick smoke test

# Exit immediately if a command exits with a non-zero status.
set -e

STAGE="${STAGE:-all}"   # crops | semantic | all

echo "--- Starting Job: $SLURM_JOB_ID (stage: $STAGE) ---"
echo "Running on host: $(hostname)"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo ""

echo "--- Checking GPU with nvidia-smi ---"
nvidia-smi
echo ""

echo "--- GPU Details ---"
echo "GPU Name: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "GPU Memory Free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
echo ""

cd "$HOME/projects/c586-tp-code/HAMMER-sgs"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
# Keep HF model downloads on persistent storage so later jobs reuse them
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

DATA_ROOT="${DATA_ROOT:-/datasets/DGM4plus}"
CROPS_DIR="${CROPS_DIR:-crops}"
OUT_CSV="${OUT_CSV:-results/semantic_blip_${SLURM_JOB_ID}.csv}"

python3 -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"
echo ""

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
        --data "$DATA_ROOT/metadata.json"
        --img_root "$DATA_ROOT"
        --out_dir "$CROPS_DIR"
        --model yolov8s-seg.pt
        --imgsz 896
    )
    append_args CROPS_CMD "${CROPS_ARGS:-}"
    python3 scripts/make_fg_bg_crops_yolo.py "${CROPS_CMD[@]}"
    echo ""
fi

if [ "$STAGE" = "semantic" ] || [ "$STAGE" = "all" ]; then
    echo "--- Stage 2: semantic FG-BG check (BLIP + MiniLM, CPU) ---"
    # Note: the CSV is only written at the end, so a timeout loses all progress of this stage
    SEMANTIC_CMD=(
        --fg_dir "$CROPS_DIR/fg"
        --bg_dir "$CROPS_DIR/bg"
        --out_csv "$OUT_CSV"
        --threshold 0.55
    )
    append_args SEMANTIC_CMD "${SEMANTIC_ARGS:-}"
    python3 scripts/semantic_fg_bg_blip.py "${SEMANTIC_CMD[@]}"
    echo ""
fi

echo "--- Job Finished Successfully ---"
