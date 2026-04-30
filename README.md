# Supercomputing Project: Fruits-262 CNN Benchmark

This project trains and benchmarks three convolutional neural network architectures on the [Fruits-262](https://www.kaggle.com/datasets/aelchimminut/fruits262) dataset (227,000 images across 262 fruit classes). The pipeline downloads the dataset, trains three models, produces a side-by-side comparison table, and renders the comparison as an SVG chart.

## Models

| Model | Description | Resolution | Epochs |
|---|---|---|---|
| `alexnet` | CNN replicated from the original Fruits-262 paper | 52×64 | 200 |
| `alexnet_bn` | Modified AlexNet with batch normalization, LR scheduling, and AdamW | 52×64 | 150 |
| `resnet50` | Fine-tuned ResNet-50 (ImageNet pretrained) | 224×224 | 50 |

> **Note on `alexnet_bn`:** the architecture supports higher input resolutions thanks to `AdaptiveAvgPool2d`, but the default has been pinned to 52×64 for an apples-to-apples comparison against the paper replication. Override with `--img-h` / `--img-w` if you want to try a larger resolution (e.g. 104×128).

Each run produces `test_results.json` and `training_log.csv`. The comparison script aggregates these into `model_comparison.csv`, and the plotting script renders that CSV as `model_comparison_graph.svg`.

## Pipeline

```
pipeline.sh
  ├─ 00_download_data.sh         Download & extract Fruits-262 from Kaggle
  └─ 01_train_cnn.slurm          Train all three models, then summarize
        ├─ alexnet
        ├─ alexnet_bn
        ├─ resnet50
        └─ 02_compare_models.py  Aggregate results into a comparison table
```

`03_plot_model_comparison.py` is run separately by user choice.

## Repository Structure

```
SUPERCOMPUTING_PROJECT/
├── pipeline.sh
├── environment.yml
├── README.md
├── .gitignore
├── scripts/
│   ├── 00_download_data.sh           Downloads Fruits-262 via the Kaggle API
│   ├── 01_train_cnn.py               Defines the three architectures + training loop
│   ├── 01_train_cnn.slurm            Trains all three models, then runs comparison
│   ├── 02_compare_models.py          Builds the comparison table and CSV
│   └── 03_plot_model_comparison.py   Renders the comparison CSV as an SVG chart
└── output/
    ├── model_comparison.csv
    ├── model_comparison_graph.svg
    ├── alexnet/
    │   ├── test_results.json
    │   └── training_log.csv
    ├── alexnet_bn/
    │   ├── test_results.json
    │   └── training_log.csv
    └── resnet50/
        ├── test_results.json
        └── training_log.csv
```

Large artifacts (the dataset itself, model checkpoints, class name files) live outside the repo on a scratch drive. We used `/sciclone/scr10/gzdata440/`:

```
EXTERNAL_DIR/
└── fruitsdata2/
    ├── Fruit-262/
    └── output/
        ├── model_comparison.csv
        ├── alexnet/
        │   ├── best_model.pth
        │   ├── training_log.csv
        │   ├── test_results.json
        │   └── class_names.json
        ├── alexnet_bn/   
        └── resnet50/     
```

## Setup

### Kaggle

`scripts/00_download_data.sh` uses the `kaggle` Python package to pull the dataset. The download command (`kaggle datasets download aelchimminut/fruits262`) worked without authentication on our HPC, but on other machines you may need to configure Kaggle credentials first. See:

- https://github.com/Kaggle/kaggle-cli
- https://www.kaggle.com/docs/api

### Environment

`environment.yml` was exported from the HPC and pins the package versions used for the final runs.

Key versions:

- Python 3.11.15
- PyTorch 2.5.1 (with CUDA 12.1)
- torchvision 0.20.1
- torchaudio 2.5.1
- NumPy 2.4.3
- Pillow 12.2.0
- Kaggle 2.0.1

The following is run in 00_download_data.sh and in pipeline.sh to create the environment:

```bash
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh
mamba env create -f environment.yml
conda activate kaggleenv
```

### Running the Pipeline

If you're running on a different machine or HPC cluster, update the hardcoded paths first:

1. In `scripts/00_download_data.sh` and `scripts/01_train_cnn.slurm`, set `SHARED_DIR` to a path on a drive with large amounts of available storage. (Both scripts derive `DATA_ROOT="${SHARED_DIR}/fruitsdata2"` from this.)
2. In `scripts/01_train_cnn.py` and `scripts/02_compare_models.py`, update `DEFAULT_DATA_DIR` and `DEFAULT_OUTPUT_DIR` to match.
3. In `scripts/01_train_cnn.slurm`, update `SCRIPT_DIR` and `REPORT_DIR` to point at your cloned repo.
4. In `pipeline.sh`, set `#SBATCH --mail-user=` to your email.
5. Create the SLURM log directory (it must exist before submission):

   ```bash
   mkdir logs
   ```

6. Submit:

Make sure you are submitting to a node with an available GPU otherwise the CNN's will take longer than necessary to train.

   ```bash
   sbatch pipeline.sh
   ```

7. Once the job finishes, render the comparison chart locally if desired:

   ```bash
   python scripts/03_plot_model_comparison.py
   ```

## Scripts

### `scripts/00_download_data.sh`

Creates the `kaggleenv` conda environment with PyTorch and dependencies, then downloads and extracts Fruits-262 into `EXTERNAL_DIR/fruitsdata2/Fruit-262/`. Wipes any existing `fruitsdata2/` directory first to avoid mixing data from incomplete runs.

### `scripts/01_train_cnn.py`

Defines the three architectures and runs training, validation, and test evaluation for whichever model is selected via `--model`. For the chosen model it writes `best_model.pth`, `training_log.csv`, `test_results.json`, and `class_names.json` to `EXTERNAL_DIR/fruitsdata2/output/<model_name>/`.

Usage:

```bash
python 01_train_cnn.py --model alexnet     [--epochs 200]
python 01_train_cnn.py --model alexnet_bn  [--epochs 150] [--img-h 64 --img-w 52]
python 01_train_cnn.py --model resnet50    [--epochs 50]
```

### `scripts/01_train_cnn.slurm`

Activates the conda environment and calls `01_train_cnn.py` three times (once per model, with `--workers 16`), then runs `02_compare_models.py`. Finally, copies the deliverable outputs (`test_results.json`, `training_log.csv`, `model_comparison.csv`) from the scratch drive into the project's `output/` folder so they get committed alongside the code.

This script has its own `#SBATCH` headers, so it can also be submitted directly with `sbatch 01_train_cnn.slurm` if the dataset has already been downloaded and you just want to retrain.

### `scripts/02_compare_models.py`

Reads `test_results.json` from each model's output directory and prints a formatted table comparing top-1/5/10 accuracy against the paper's reported benchmarks. Writes `model_comparison.csv` to the output directory.

### `scripts/03_plot_model_comparison.py`

Reads `output/model_comparison.csv` and writes `output/model_comparison_graph.svg` — a self-contained SVG with grouped accuracy bars (top-1/5/10), a training-time panel, and parameter-count annotations. Uses only the Python standard library (no matplotlib required), so it can run anywhere without dragging in plotting dependencies.

Usage:

```bash
python scripts/03_plot_model_comparison.py [--csv <path>] [--out <path>]
```

This script is **not** wired into the SLURM pipeline — run it locally after the job completes.

### `pipeline.sh`

SLURM entry point. Submit with `sbatch pipeline.sh`. Runs `00_download_data.sh` followed by `01_train_cnn.slurm`. SLURM stdout/stderr logs land in `./logs/`.

Resource request:

- 1 node, 32 CPUs, 64 GB RAM, 1 GPU
- 48-hour wall time (the long runtime is mostly the sequential training of three CNNs, plus some overhead from cleaning up previous data directories)

## Results

<img width="1200" height="760" alt="image" src="https://github.com/user-attachments/assets/d94a54c9-5b47-4155-829e-264e0e3201d1" />

Observations:

The paper replication came out close to the published numbers. We got 61.8% top-1 vs the paper's 59.15%, and top-5 and top-10 also landed within a couple points of what they reported. Differences this small are probably just RNG and minor implementation choices, so we are calling this a successful reproduction.

Adding BatchNorm, AdamW, and LR scheduling to the same architecture got us about 10 points of top-1 accuracy with no resolution change. alexnet_bn ran at 52×64 just like the paper model, and it actually has fewer parameters (6.2M vs 7.0M) since the adaptive pooling collapses the conv output to a fixed size before the dense layers. It also finished slightly faster than the paper model. So the gain is coming from optimization tricks, not from giving the model more capacity.

ResNet-50 is the clear winner on accuracy at 84.6% top-1, but it costs about 4x the parameters and 3x the training time. It also depends on the ImageNet pretraining to work, so it is not really comparable to the from-scratch models in a fair sense. If you needed something small and fast, alexnet_bn is the better pick. If you just want the best accuracy and have the compute, ResNet-50 wins.

Top-10 saturates pretty quickly across all three models. Even the paper replication clears 88%, and ResNet-50 is at 97.7%. Since this is a 262-way classification problem, that means the right answer is almost always somewhere in the model's top guesses. Most of the remaining error is probably the model confusing fruits that look similar, not the model being lost.

## References

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 248–255). IEEE. https://doi.org/10.1109/CVPR.2009.5206848

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778). IEEE. https://doi.org/10.1109/CVPR.2016.90

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). ImageNet classification with deep convolutional neural networks. *Communications of the ACM, 60*(6), 84–90. https://doi.org/10.1145/3065386

Minuț, M.-D. (2021). *Fruits-262* [Data set]. Kaggle. https://www.kaggle.com/datasets/aelchimminut/fruits262

Minuț, M.-D., & Iftene, A. (2021). Creating a dataset and models based on convolutional neural networks to improve fruit classification. In *2021 23rd International Symposium on Symbolic and Numeric Algorithms for Scientific Computing (SYNASC)* (pp. 155–162). IEEE. https://doi.org/10.1109/SYNASC54541.2021.00035
