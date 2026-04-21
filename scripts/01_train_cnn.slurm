#!/bin/bash
# ─── Environment Setup ───────────────────────────────────────────────────────
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh

# Create env on first run
conda activate kaggleenv

# Install PyTorch with CUDA support
# Check CUDA version: run 'nvidia-smi' or 'module avail cuda' on a GPU node
# Adjust cu121 below to match (cu118 for CUDA 11.8, cu121 for 12.1, etc.)
# If the above CUDA version doesn't match, fall back to auto-detect:
# pip install --quiet torch torchvision

# ─── Paths ───────────────────────────────────────────────────────────────────
SHARED_DIR="/sciclone/scr10/gzdata440"
DATA_DIR="${SHARED_DIR}/fruitsdata/Fruit-262"
OUTPUT_DIR="${SHARED_DIR}/fruitsdata/output"
SCRIPT_DIR="$HOME/SUPERCOMPUTING_PROJECT/scripts"

echo "════════════════════════════════════════════════════════════"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Data:         $DATA_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Start:        $(date)"
echo "════════════════════════════════════════════════════════════"

# Quick GPU check
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"

# ─── Model 1: Paper Replication (AlexNet, 52×64) ────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MODEL 1/3: alexnet (paper replication)"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/01_train_cnn.py" \
    --model alexnet \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 200 \
    --workers 8

# ─── Model 2: Improved AlexNet (+ BatchNorm + LR scheduling, 104×128) ───────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MODEL 2/3: alexnet_bn (improved)"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/01_train_cnn.py" \
    --model alexnet_bn \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 150 \
    --workers 8

# ─── Model 3: Transfer Learning (ResNet50, 224×224) ─────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MODEL 3/3: resnet50 (transfer learning)"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/01_train_cnn.py" \
    --model resnet50 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 50 \
    --workers 8

# ─── Comparison Summary ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  COMPARISON SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/02_compare_models.py" \
    --output-dir "$OUTPUT_DIR"

#TEST OUTPUT CHANGE
REPORT_DIR="$HOME/SUPERCOMPUTING_PROJECT/output"
mkdir -p "$REPORT_DIR"
for model in alexnet alexnet_bn resnet50; do
    mkdir -p "$REPORT_DIR/$model"
    cp "$OUTPUT_DIR/$model/test_results.json"  "$REPORT_DIR/$model/" 2>/dev/null
    cp "$OUTPUT_DIR/$model/training_log.csv"   "$REPORT_DIR/$model/" 2>/dev/null
done
cp "$OUTPUT_DIR/model_comparison.csv" "$REPORT_DIR/" 2>/dev/null

echo ""
echo "All done: $(date)"
