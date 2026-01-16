# Spatiotemporal Traffic Forecasting with Missing Data

Graph neural network framework for real-time traffic speed prediction on Nevada DOT's 900-sensor network. **Three model variants** comparing different missing data handling strategies using GMAN architecture.

**Published:** IEEE ICVES 2024, IEEE ITSC 2024

---

## Problem Statement

Real-world traffic sensor networks face **30-40% missing data** from sensor failures and communication issues. This repository compares three strategies:
1. **Imputation** - Fill missing values before training
2. **Hybrid** - Masked inputs, imputed targets
3. **Uncertainty-preserving** - No imputation, loss only on valid data ⭐

---

## Model Variants: Technical Comparison

### 1️⃣ Baseline Imputed (`baseline_imputed/`)

**Input (X):**
- Shape: `[batch, num_his, num_sensors]`
- Missing values **filled** using imputation (forward fill, mean, etc.)
- All historical values are complete

**Target (Y):**
- Shape: `[batch, num_pred, num_sensors]`
- Missing values **filled** using imputation
- All target values are complete

**Loss Calculation:**
```python
# model/train.py:90
loss = MSELoss(pred, label)  # All positions included
```
- Standard MSE on **all predictions**
- Treats imputed values as ground truth

**Tradeoff:**
✅ Simple implementation
❌ Imputation bias - model trained on artificial data

---

### 2️⃣ Hybrid Masked Input (`hybrid_masked_input/`)

**Input (X):**
- Shape: `[batch, num_his, num_sensors]`
- Missing values **preserved** as indicators/masks
- Model learns from realistic incomplete patterns

**Target (Y):**
- Shape: `[batch, num_pred, num_sensors]`
- Missing values **filled** using imputation
- All target values are complete

**Loss Calculation:**
```python
# model/train.py:90
loss = MSELoss(pred, label)  # Imputed targets
```
- Standard MSE on **imputed targets**
- No masking in loss function

**Tradeoff:**
✅ Model handles missing inputs realistically
❌ Still relies on imputed targets for supervision

---

### 3️⃣ Masked No Imputation (`masked_no_imputation/`) ⭐ **RECOMMENDED**

**Input (X):**
- Shape: `[batch, num_his, num_sensors]`
- Missing values represented as **placeholder (-1)**
- No imputation applied

**Target (Y):**
- Shape: `[batch, num_pred, num_sensors]`
- Missing values represented as **placeholder (-1)**
- No imputation applied

**Loss Calculation:**
```python
# model/train.py:97-99
mask = (label != placeholder) & (pred != placeholder)
pred_valid = pred[mask]
label_valid = label[mask]
loss = MSELoss(pred_valid, label_valid)  # Only real values
```
- Masked MSE computed **only on valid positions**
- Skips batches with zero valid values

**Tradeoff:**
✅ No imputation bias - learns only from real data
✅ Preserves prediction uncertainty
✅ Best forecasting accuracy
❌ Requires more training data
❌ Slightly more complex training logic

---

## Repository Structure

```
model_variants/
├── baseline_imputed/          # Variant 1: Full imputation
│   ├── model/
│   │   ├── model_.py         # GMAN architecture
│   │   ├── train.py          # Standard training (MSE on all)
│   │   └── test.py           # Evaluation
│   └── main.py
│
├── hybrid_masked_input/       # Variant 2: Masked X, imputed Y
│   ├── model/
│   │   ├── model_.py         # GMAN architecture
│   │   ├── train.py          # Standard training (MSE on imputed Y)
│   │   └── test.py
│   └── main.py
│
└── masked_no_imputation/      # Variant 3: No imputation (BEST) ⭐
    ├── model/
    │   ├── model_.py         # GMAN architecture
    │   ├── train.py          # Masked training (MSE on valid only)
    │   └── test.py
    └── main.py
```

---

## GMAN Architecture

All variants use **Graph Multi-Attention Network (GMAN)**:

```
Input → FC → [Encoder: L×STAttBlock] → TransformAttention → [Decoder: L×STAttBlock] → FC → Output
```

**Components:**
- **STAttBlock** = Spatial Attention + Temporal Attention + Gated Fusion
- **Spatial Attention**: Multi-head attention over sensors
- **Temporal Attention**: Multi-head attention over time steps
- **Transform Attention**: Encoder-decoder for multi-step forecasting

**Parameters:** L=3 blocks, K=8 heads, d=8 dims per head

---

## Quick Start

### Installation
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Run Baseline
```bash
cd baseline_imputed
python main.py --train_file ./data/train.csv \
               --val_test_file ./data/val_test.csv \
               --max_epoch 100 --batch_size 32
```

### Run Masked No Imputation (Recommended)
```bash
cd masked_no_imputation
python main.py --train_file ./data/train.csv \
               --val_test_file ./data/val_test.csv \
               --missing_value_placeholder -1 \
               --max_epoch 100 --batch_size 32
```

### Key Arguments
- `--num_his 20`: Historical time steps
- `--num_pred 4`: Prediction horizon
- `--missing_value_placeholder -1`: Placeholder for missing data (variant 3 only)
- `--learning_rate 0.01`: Initial learning rate
- `--cuda_device 0`: GPU device index

---

## Results Summary

| Variant | Input Format | Target Format | Loss Calculation | Performance |
|---------|-------------|---------------|------------------|-------------|
| Baseline | Imputed | Imputed | MSE (all) | Baseline |
| Hybrid | Masked | Imputed | MSE (imputed Y) | Moderate improvement |
| **No Imputation** ⭐ | **Placeholders** | **Placeholders** | **MSE (valid only)** | **Best** |

**Key Findings** (30-40% missing data):
- ⭐ Masked no imputation achieves **best accuracy** across all metrics
- All variants maintain **<100ms inference** time
- Performance gap widens with higher missing data rates
- See [Publications](#publications) for detailed metrics

---

## Data Format

**Input CSV:** `[num_samples, num_sensors]` - traffic speed values
**Spatial Embedding:** `[num_sensors, embedding_dim]` - sensor embeddings
**Timestamps:** `[num_samples, 2]` - (day_of_week, time_of_day)

**Missing values:**
- Baseline: Filled before training
- Hybrid: Preserved in X, filled in Y
- No imputation: `-1` placeholder in both X and Y

---

## Publications

1. **T. b. Zahid and B. Morris**, "Using Deep Traffic Prediction for EMFAC Emission Estimation and Visualization," *IEEE ITSC 2024*, pp. 2488-2493, doi: 10.1109/ITSC58415.2024.10919675

2. **T. Bin Zahid and B. T. Morris**, "Benchmarking/Limitations of Traffic Prediction with Noisy Field Measurements," *IEEE ICVES 2024*, pp. 1-6, doi: 10.1109/ICVES61986.2024.10928136

---

## Citation

If you use this code, please cite our papers:
```bibtex
@inproceedings{zahid2024emfac,
  title={Using Deep Traffic Prediction for EMFAC Emission Estimation and Visualization},
  author={Zahid, T. b. and Morris, B.},
  booktitle={2024 IEEE ITSC},
  pages={2488--2493},
  year={2024}
}
```
