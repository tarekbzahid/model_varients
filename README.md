# Spatiotemporal Traffic Forecasting with Missing Data

Graph neural network framework for real-time traffic speed prediction on Nevada DOT's 900-sensor network. Three model variants comparing different missing data handling strategies using GMAN architecture.

**Published:** IEEE ICVES 2024, IEEE ITSC 2024

---

## Problem Statement

Real-world traffic sensor networks face 30-40% missing data from sensor failures and communication issues. This repository compares three strategies:
1. **MICE Baseline** - Full imputation on both input and target before training
2. **Experiment B(1)** - Masked inputs, imputed targets
3. **Experiment B(2)** - No imputation, loss only on valid data

---

## Model Variants: Technical Comparison

### 1. Baseline Imputed (`baseline_imputed_mice_baseline/`) - MICE Baseline

**Input (X):**
- Shape: `[batch, num_his, num_sensors]`
- Missing values filled using MICE imputation
- All historical values are complete

**Target (Y):**
- Shape: `[batch, num_pred, num_sensors]`
- Missing values filled using MICE imputation
- All target values are complete

**Loss Calculation:**
```python
# model/train.py:90
loss = MSELoss(pred, label)  # All positions included
```
- Standard MSE on all predictions
- Treats imputed values as ground truth

**Tradeoff:**
- Simple implementation
- Imputation bias: model trained on artificially reconstructed data

---

### 2. Hybrid Masked Input (`hybrid_masked_input_exp_b1/`) - Experiment B(1)

**Input (X):**
- Shape: `[batch, num_his, num_sensors]`
- Missing values preserved as zeros (noisy historical sequences)
- Model sees realistic incomplete patterns

**Target (Y):**
- Shape: `[batch, num_pred, num_sensors]`
- Missing values filled using MICE imputation
- All target values are complete

**Loss Calculation:**
```python
# model/train.py:90
loss = MSELoss(pred, label)  # Imputed targets
```
- Standard MSE on imputed targets
- No masking in loss function

**Tradeoff:**
- Model handles missing inputs realistically
- Still relies on imputed targets for supervision

---

### 3. Masked No Imputation (`masked_no_imputation_exp_b2/`) - Experiment B(2)

**Input (X):**
- Shape: `[batch, num_his, num_sensors]`
- Missing values represented as placeholder (-1)
- No imputation applied

**Target (Y):**
- Shape: `[batch, num_pred, num_sensors]`
- Missing values represented as placeholder (-1)
- No imputation applied

**Loss Calculation:**
```python
# model/train.py:97-99
mask = (label != placeholder) & (pred != placeholder)
pred_valid = pred[mask]
label_valid = label[mask]
loss = MSELoss(pred_valid, label_valid)  # Only real values
```
- Masked MSE computed only on valid positions
- Skips batches with zero valid values

**Tradeoff:**
- No imputation bias: learns only from real sensor measurements
- Best MAE across all experiments (3.139 vs 3.47 MICE baseline)
- Requires sufficient real data coverage per batch

---

## Repository Structure

```
model_variants/
├── baseline_imputed_mice_baseline/   # MICE Baseline: full imputation on X and Y
│   ├── data/
│   │   ├── train.csv                # Training data (MICE imputed)
│   │   ├── val_test.csv             # Validation/test data (MICE imputed)
│   │   ├── I-15_NB_SE.txt           # Spatial embeddings
│   │   ├── timestamps_new.txt        # Temporal features
│   │   └── *.pt                     # Model checkpoints
│   ├── figure/                      # Output plots and predictions
│   ├── model/
│   │   ├── model_.py                # GMAN architecture
│   │   ├── train.py                 # Standard training (MSE on all)
│   │   └── test.py                  # Evaluation
│   ├── utils/
│   │   └── utils_.py                # Data loading & preprocessing
│   ├── main.py                      # Entry point
│   └── test_chkpt.py                # Checkpoint testing
│
├── hybrid_masked_input_exp_b1/       # Experiment B(1): noisy X, MICE-imputed Y
│   ├── data/
│   │   ├── train_x_0.csv            # Training inputs (zeros for missing)
│   │   ├── train_y_0.csv            # Training targets (MICE imputed)
│   │   ├── val_test_x_0.csv         # Val/test inputs (with missing)
│   │   ├── val_test_y_0.csv         # Val/test targets (MICE imputed)
│   │   ├── I-15_NB_SE.txt
│   │   └── timestamps_new.txt
│   ├── figure/
│   ├── model/
│   │   ├── model_.py                # GMAN architecture
│   │   ├── train.py                 # Training (MSE on imputed Y)
│   │   └── test.py
│   ├── utils/
│   │   └── utils_.py
│   ├── main.py
│   └── test_chkpt.py
│
└── masked_no_imputation_exp_b2/      # Experiment B(2): real measurements only
    ├── data/
    │   ├── train.csv                # Training data (-1 placeholders for missing)
    │   ├── val_test.csv             # Val/test data (-1 placeholders for missing)
    │   ├── I-15_NB_SE.txt
    │   └── timestamps_new.txt
    ├── figure/
    ├── model/
    │   ├── model_.py                # GMAN architecture
    │   ├── train.py                 # Masked training (MSE on valid only)
    │   └── test.py
    ├── utils/
    │   └── utils_.py
    ├── main.py
    └── test_chkpt.py
```

---

## GMAN Architecture

All variants use Graph Multi-Attention Network (GMAN):

```
Input -> FC -> [Encoder: L x STAttBlock] -> TransformAttention -> [Decoder: L x STAttBlock] -> FC -> Output
```

**Components:**
- **STAttBlock** = Spatial Attention + Temporal Attention + Gated Fusion
- **Spatial Attention**: Multi-head attention over sensors
- **Temporal Attention**: Multi-head attention over time steps
- **Transform Attention**: Encoder-decoder bridge for multi-step forecasting

**Parameters:** L=3 blocks, K=8 heads, d=8 dims per head

---

## Quick Start

### Installation
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Run MICE Baseline
```bash
cd baseline_imputed_mice_baseline
python main.py --train_file ./data/train.csv \
               --val_test_file ./data/val_test.csv \
               --max_epoch 100 --batch_size 32
```

### Run Experiment B(2) - Real Measurements Only
```bash
cd masked_no_imputation_exp_b2
python main.py --train_file ./data/train.csv \
               --val_test_file ./data/val_test.csv \
               --missing_value_placeholder -1 \
               --max_epoch 100 --batch_size 32
```

### Key Arguments
- `--num_his 20`: Historical time steps
- `--num_pred 4`: Prediction horizon
- `--missing_value_placeholder -1`: Placeholder for missing data (Exp B(2) only)
- `--learning_rate 0.01`: Initial learning rate
- `--cuda_device 0`: GPU device index

---

## Results Summary

| Folder | Experiment | Input Format | Target Format | Loss Calculation | Total MAE | Total RMSE |
|--------|------------|-------------|---------------|------------------|-----------|------------|
| `baseline_imputed_mice_baseline` | MICE Baseline | MICE imputed | MICE imputed | MSE (all) | 3.47 | 6.56 |
| `hybrid_masked_input_exp_b1` | Exp B(1) | Zeros for missing | MICE imputed | MSE (imputed Y) | 3.443 | 6.587 |
| `masked_no_imputation_exp_b2` | Exp B(2) | -1 placeholder | -1 placeholder | MSE (valid only) | 3.139 | 7.027 |

Experiment B(2) achieves the lowest average MAE but higher RMSE on some segments (e.g. CC-215S, US-95S) due to high variability in real sensor data.

---

## Data Format

**Input CSV:** `[num_samples, num_sensors]` - traffic speed values
**Spatial Embedding:** `[num_sensors, embedding_dim]` - sensor embeddings
**Timestamps:** `[num_samples, 2]` - (day_of_week, time_of_day)

**Missing values:**
- MICE Baseline: filled before training
- Exp B(1): zeros in X, MICE-imputed in Y
- Exp B(2): `-1` placeholder in both X and Y

---

## Publications

1. **T. b. Zahid and B. Morris**, "Using Deep Traffic Prediction for EMFAC Emission Estimation and Visualization," *IEEE ITSC 2024*, pp. 2488-2493, doi: 10.1109/ITSC58415.2024.10919675

2. **T. Bin Zahid and B. T. Morris**, "Benchmarking/Limitations of Traffic Prediction with Noisy Field Measurements," *IEEE ICVES 2024*, pp. 1-6, doi: 10.1109/ICVES61986.2024.10928136

---

## Citation

```bibtex
@inproceedings{zahid2024emfac,
  title={Using Deep Traffic Prediction for EMFAC Emission Estimation and Visualization},
  author={Zahid, T. b. and Morris, B.},
  booktitle={2024 IEEE ITSC},
  pages={2488--2493},
  year={2024}
}

@INPROCEEDINGS{10928136,
  author={Bin Zahid, Tarek and Morris, Brendan Tran},
  booktitle={2024 IEEE International Conference on Vehicular Electronics and Safety (ICVES)},
  title={Benchmarking/Limitations of Traffic Prediction with Noisy Field Measurements},
  year={2024},
  pages={1-6},
  doi={10.1109/ICVES61986.2024.10928136}
}
```
