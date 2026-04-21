# SUPERCOMPUTING_PROJECT 

This project trains and outputs benchmarks on 3 convolutional neural network approaches for the [Fruits-262](https://www.kaggle.com/datasets/aelchimminut/fruits262) dataset. The pipeline begins by downloading the dataset, 

## Pipeline 

```
pipeline.sh
  │
  ├─ 00_download_data.sh      Download & extract Fruits-262 from Kaggle
  │
  ├─ 01_train_cnn.slurm       Train 3 models sequentially:
  │     ├─ alexnet             CNN Derived from kaggle paper  (52×64,  200 epochs)
  │     ├─ alexnet_bn          Improved paper CNN with larger image resolution   (104×128, 150 epochs)
  │     └─ resnet50            Transfer learning derived from documentation (tbd)  (224×224,  50 epochs)
  │
  └─ 02_compare_models.py     Aggregated results & comparison table
```

Each model run returns `test_results.json` and `training_log.csv` and the comparison script uses the outputs from model runs to produce a summary table and `model_comparison.csv`.

## Structure

```
SUPERCOMPUTING_PROJECT/
├── pipeline.sh                 # SLURM Pipeline
├── scripts/
│   ├── 00_download_data.sh     # Downloads Fruits-262 from Kaggle API
│   ├── 01_train_cnn.py         # Model training
│   ├── 01_train_cnn.slurm      # Slurm Script for training
│   └── 02_compare_models.py    # comparison
├── output/                    
├── .gitignore
└── README.md
```

```
EXTERNAL_DIR/
├── fruitsdata/
│   ├── output/
│       └── model_comparison.csv 
└── README.md
```

## Setup 

# Kaggle setup  
# 
