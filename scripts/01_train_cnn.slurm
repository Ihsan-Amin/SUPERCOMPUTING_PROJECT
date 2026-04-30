#!/bin/bash
#SBATCH --job-name=fruit_cnn_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -o ./logs/fruit_cnn_train_%j.out
#SBATCH -e ./logs/fruit_cnn_train_%j.err

# ─── Environment Setup ───────────────────────────────────────────────────────
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh

##Activate environmnt from download data script
conda activate kaggleenv

# ─── Paths ───────────────────────────────────────────────────────────────────
SHARED_DIR="/sciclone/scr10/gzdata440"
DATA_ROOT="${SHARED_DIR}/fruitsdata2"
DATA_DIR="${DATA_ROOT}/Fruit-262"
OUTPUT_DIR="${DATA_ROOT}/output"
SCRIPT_DIR="$HOME/SUPERCOMPUTING_PROJECT/scripts"

# Quick GPU check
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only'); raise SystemExit(0 if torch.cuda.is_available() else 1)"

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
    --workers 16

# ─── Model 2: Improved AlexNet (+ BatchNorm + LR scheduling, safe 52x64) ─────
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
    --img-h 64 \
    --img-w 52 \
    --workers 16

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
    --workers 16

# ─── Comparison Summary ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  COMPARISON SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

python "${SCRIPT_DIR}/02_compare_models.py" \
    --output-dir "$OUTPUT_DIR"

python "${SCRIPT_DIR}/03_plot_model_comparison.py" \
    --csv "$OUTPUT_DIR/model_comparison.csv" \
    --out "$OUTPUT_DIR/model_comparison_graph.svg"

#TEST OUTPUT CHANGE
REPORT_DIR="$HOME/SUPERCOMPUTING_PROJECT/output"
mkdir -p "$REPORT_DIR"
for model in alexnet alexnet_bn resnet50; do
    mkdir -p "$REPORT_DIR/$model"
    cp "$OUTPUT_DIR/$model/test_results.json"  "$REPORT_DIR/$model/" 2>/dev/null
    cp "$OUTPUT_DIR/$model/training_log.csv"   "$REPORT_DIR/$model/" 2>/dev/null
done
cp "$OUTPUT_DIR/model_comparison.csv" "$REPORT_DIR/" 2>/dev/null
cp "$OUTPUT_DIR/model_comparison_graph.svg" "$REPORT_DIR/" 2>/dev/null

echo ""
echo "All done: $(date)"
