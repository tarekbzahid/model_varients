# Spatiotemporal Traffic Forecasting with Graph Neural Networks

Scalable graph neural network framework for real-time traffic speed prediction across Nevada DOT's 900-sensor network using GMAN (Graph Multi-Attention Network) architecture.

**Published at:** IEEE ICVES 2024, IEEE ITSC 2024

## Key Contributions

- **Novel Masked Training**: Preserves prediction uncertainty instead of imputing missing data
- **Scalable Architecture**: Scaled from 26 to 900 sensors while maintaining real-time performance
- **Production Deployment**: Sub-100ms inference latency
- **Three Model Variants**: Comprehensive comparison of missing data handling strategies

## Problem Context

Real-world traffic sensor networks face 30-40% missing data rates from sensor failures, communication issues, and maintenance downtime. Traditional imputation approaches (filling missing values with estimates) lose uncertainty information critical for reliable predictions and can introduce systematic biases into forecasting models.

---

## Model Variants Explained

This repository contains three implementations that compare different strategies for handling missing data in traffic prediction:

### 1. Basic (`1. basic/`)

**Purpose**: Baseline approach using traditional data imputation

**How it works**:
- Missing sensor values are pre-filled using imputation techniques (e.g., forward fill, mean imputation)
- The model trains on the complete imputed dataset
- Loss is calculated on all predicted values including imputed positions
- Standard GMAN architecture without modifications

**Key characteristics**:
- **Data preprocessing**: Missing values filled before training
- **Loss computation**: MSE loss on all predictions (`model/train.py:90`)
- **Advantage**: Simple, straightforward implementation
- **Disadvantage**: Treats imputed values as ground truth, losing uncertainty information

**Use when**: You have high-quality imputation or low missing data rates (<10%)

---

### 2. Masked Missing X, Imputed Y (`2. masked_missingX_imputedY/`)

**Purpose**: Hybrid approach combining masked inputs with imputed targets

**How it works**:
- Input sequences (X) preserve missing value indicators/masks
- Target values (Y) are imputed for training
- Model learns to predict from partially missing historical data
- Loss calculated on imputed target positions

**Key characteristics**:
- **Data preprocessing**: Selective imputation (targets only)
- **Loss computation**: Standard MSE on imputed targets (`model/train.py:90`)
- **Advantage**: Model sees realistic input patterns with missing data
- **Disadvantage**: Still relies on imputation quality for targets

**Use when**: You want the model to handle missing inputs but have reliable target imputation

---

### 3. Masked Ignore Placeholder (`3. masked_ignore_placeholder/`) ⭐ **BEST RESULTS**

**Purpose**: Uncertainty-preserving approach that avoids imputation entirely

**How it works**:
- Missing values represented by placeholder tokens (default: -1)
- Model processes data with placeholders intact
- **Critical innovation**: Loss calculated ONLY on valid (non-missing) values
- Masking logic filters out placeholder values before loss computation

**Key characteristics**:
- **Data preprocessing**: No imputation - placeholders preserved (`main.py:41`)
- **Loss computation**: Masked MSE only on real values (`model/train.py:97-99`)
  ```python
  mask = (label != args.missing_value_placeholder) & (pred != args.missing_value_placeholder)
  pred = pred[mask]
  label = label[mask]
  ```
- **Advantage**: Preserves data uncertainty, no imputation bias, learns robust patterns
- **Disadvantage**: Requires more data, slightly more complex training logic

**Use when**: High missing data rates (>30%), production systems requiring reliable uncertainty estimates

**Why it's best**: Achieves superior forecasting accuracy by learning from real data only, without assuming imputed values are correct

---

## Repository Structure

```
model_varients/
├── 1. basic/                      # Baseline with full imputation
│   ├── model/
│   │   ├── model_.py             # GMAN architecture
│   │   ├── train.py              # Standard training loop
│   │   └── test.py               # Evaluation
│   ├── utils/
│   │   └── utils_.py             # Data loading & preprocessing
│   └── main.py                   # Entry point
│
├── 2. masked_missingX_imputedY/  # Hybrid masked approach
│   ├── model/
│   │   ├── model_.py             # GMAN architecture
│   │   ├── train.py              # Hybrid training loop
│   │   └── test.py               # Evaluation
│   ├── utils/
│   │   └── utils_.py             # Data loading & preprocessing
│   └── main.py                   # Entry point
│
└── 3. masked_ignore_placeholder/ # Uncertainty-preserving (BEST)
    ├── model/
    │   ├── model_.py             # GMAN architecture
    │   ├── train.py              # Masked training loop
    │   └── test.py               # Evaluation
    ├── utils/
    │   └── utils_.py             # Data loading & preprocessing
    └── main.py                   # Entry point
```

---

## Model Architecture: GMAN

All variants use the same underlying **Graph Multi-Attention Network (GMAN)** architecture:

**Core Components**:
- **Spatiotemporal Embedding**: Encodes spatial (sensor locations) and temporal (time-of-day, day-of-week) features
- **Spatial Attention**: Multi-head attention capturing inter-sensor dependencies
- **Temporal Attention**: Multi-head attention modeling time series patterns
- **Gated Fusion**: Combines spatial and temporal features adaptively
- **Transform Attention**: Encoder-decoder architecture for multi-step forecasting

**Architecture Flow**:
```
Input → FC → [Encoder: L×STAttBlock] → TransformAttention → [Decoder: L×STAttBlock] → FC → Output
```

Where STAttBlock = SpatialAttention + TemporalAttention + GatedFusion

**Key Parameters** (`main.py`):
- `L`: Number of attention blocks (default: 3)
- `K`: Number of attention heads (default: 8)
- `d`: Dimension per attention head (default: 8)
- `num_his`: Historical time steps (default: 20)
- `num_pred`: Prediction steps (default: 4)

---

## Installation

Install dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

**Requirements**:
- Python 3.7+
- PyTorch 1.8+
- CUDA-capable GPU recommended (but CPU supported)

---

## Quick Start

### Running a Model Variant

**Baseline Model**:
```bash
cd "1. basic"
python main.py --train_file ./data/train.csv \
               --val_test_file ./data/val_test.csv \
               --max_epoch 100 \
               --batch_size 32
```

**Masked Ignore Placeholder (Recommended)**:
```bash
cd "3. masked_ignore_placeholder"
python main.py --train_file ./data/train.csv \
               --val_test_file ./data/val_test.csv \
               --missing_value_placeholder -1 \
               --max_epoch 100 \
               --batch_size 32
```

### Key Command-Line Arguments

- `--num_his 20`: Number of historical time steps (input sequence length)
- `--num_pred 4`: Number of prediction steps (forecast horizon)
- `--batch_size 32`: Training batch size
- `--max_epoch 100`: Maximum training epochs
- `--learning_rate 0.01`: Initial learning rate
- `--missing_value_placeholder -1`: Value representing missing data (variant 3 only)
- `--cuda_device 0`: GPU device index

---

## Results

Performance comparison on traffic networks with **high missing data rates (30-40%)**:

| Model Variant | Performance | Key Characteristics |
|--------------|-------------|---------------------|
| Basic (imputation) | Baseline | Simple implementation, imputation bias present |
| Masked X, Imputed Y | Moderate improvement | Handles missing inputs, still relies on target imputation |
| **Masked Ignore (Ours)** | **Best performance** | **Lowest RMSE/MAE, no imputation bias** |

**Key Findings**:
- **Masked ignore approach achieves superior forecasting accuracy** across all metrics
- All variants maintain **real-time inference** (<100ms per prediction)
- Performance gap increases with higher missing data rates
- **No imputation bias** - masked ignore predictions based only on real observations
- Published results available in IEEE ICVES 2024 and IEEE ITSC 2024 papers (see Publications section)

---

## Data Format

**Training/Test CSV**: `[num_samples, num_sensors]` - traffic speed values
**Spatial Embedding**: `[num_sensors, embedding_dim]` - learned sensor embeddings
**Timestamps**: `[num_samples, 2]` - (day_of_week, time_of_day) for each sample

**Missing values**:
- Variant 1 & 2: Imputed before/during training
- Variant 3: Represented as `-1` (or custom placeholder)

## Publications (Cite if useful)

1. T. b. Zahid and B. Morris, "Using Deep Traffic Prediction for EMFAC Emission Estimation and Visualization," 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), Edmonton, AB, Canada, 2024, pp. 2488-2493, doi: 10.1109/ITSC58415.2024.10919675. keywords: {Solid modeling;Accuracy;Decision making;Transportation;Estimation;Data visualization;Predictive models;Transformers;Data models;Environmental factors},

2. T. Bin Zahid and B. T. Morris, "Benchmarking/Limitations of Traffic Prediction with Noisy Field Measurements," 2024 IEEE International Conference on Vehicular Electronics and Safety (ICVES), Ahmedabad, India, 2024, pp. 1-6, doi: 10.1109/ICVES61986.2024.10928136. keywords: {Training;Vehicular and wireless technologies;Accuracy;Roads;Urban planning;Predictive models;Transformers;Data models;Robustness;Noise measurement},


