#!/bin/bash
#SBATCH --job-name=fruit_cnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-12:00:00          # 1.5 days (3 models sequentially)
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=mrmellors@wm.edu
#SBATCH -o fruit_cnn_%j.out
#SBATCH -e fruit_cnn_%j.err
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ GPU PARTITION — UNCOMMENT ONE OF THESE BLOCKS                      │
# │ Run 'sinfo -o "%P %G %N %l"' to see what's available to you       │
# └─────────────────────────────────────────────────────────────────────┘
#
# Option A: astral (8× A30, 24GB each) — best for this workload
##SBATCH --partition=astral
##SBATCH --gres=gpu:1
#
# Option B: gulf (2× A40, 48GB each) — if astral is full
##SBATCH --partition=gulf
##SBATCH --gres=gpu:1
#
# Option C: bora CPU fallback (no GPU — VERY SLOW, not recommended)
# Leave GPU lines commented out, it will auto-detect CPU mode

# ─── Environment Setup ───────────────────────────────────────────────────────
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh

# Create env on first run
if ! conda env list | grep -q "fruitcnn"; then
    echo "Creating conda environment..."
    conda create -n fruitcnn python=3.11 -y
fi
conda activate fruitcnn

# Install PyTorch with CUDA support
# Check CUDA version: run 'nvidia-smi' or 'module avail cuda' on a GPU node
# Adjust cu121 below to match (cu118 for CUDA 11.8, cu121 for 12.1, etc.)
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet numpy pillow

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

echo ""
echo "All done: $(date)"
