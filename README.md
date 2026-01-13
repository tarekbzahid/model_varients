# Spatiotemporal Traffic Forecasting with Graph Neural Networks

Scalable graph neural network framework for real-time traffic speed prediction across Nevada DOT's 900-sensor network.

**Published at:** IEEE ICVES 2024, IEEE ITSC 2024

## Key Contributions

- Novel Masked Training: Preserves prediction uncertainty instead of imputing missing data
- Scalable Architecture: Scaled from 26 to 900 sensors while maintaining real-time performance
- Production Deployment: Sub-100ms inference latency
- Multiple GNN Variants: GCN, GraphSAGE, GAT, and AGCRN

## Problem Context

Real-world traffic sensor networks face 30-40% missing data rates from sensor failures. Traditional imputation approaches lose uncertainty information critical for reliable predictions.

## Repository Structure

- 1_basic: Baseline GNN implementations
- 2_masked_missingX_imputedY: Hybrid approach with masked inputs
- 3_masked_ignore_placeholder: Uncertainty-preserving approach (BEST RESULTS)

## Installation

Install dependencies:

    pip install torch torch-geometric pandas numpy scikit-learn

## Quick Start

    from masked_ignore_placeholder.model import MaskedGNN
    from masked_ignore_placeholder.train import train_model
    
    # Load data
    data = load_traffic_data('nevada_dot_sensors.csv')
    
    # Train model
    model = MaskedGNN(num_nodes=900, hidden_dim=64)
    train_model(model, data, mask_rate=0.3)

## Results

On Nevada DOT network with 35% missing data:
- Baseline (imputation): RMSE 8.42, MAE 5.73
- Masked approach (ours): RMSE 7.89, MAE 5.21
- Inference time: 87ms

## Publications (Cite if useful)

1. T. b. Zahid and B. Morris, "Using Deep Traffic Prediction for EMFAC Emission Estimation and Visualization," 2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC), Edmonton, AB, Canada, 2024, pp. 2488-2493, doi: 10.1109/ITSC58415.2024.10919675. keywords: {Solid modeling;Accuracy;Decision making;Transportation;Estimation;Data visualization;Predictive models;Transformers;Data models;Environmental factors},

2. T. Bin Zahid and B. T. Morris, "Benchmarking/Limitations of Traffic Prediction with Noisy Field Measurements," 2024 IEEE International Conference on Vehicular Electronics and Safety (ICVES), Ahmedabad, India, 2024, pp. 1-6, doi: 10.1109/ICVES61986.2024.10928136. keywords: {Training;Vehicular and wireless technologies;Accuracy;Roads;Urban planning;Predictive models;Transformers;Data models;Robustness;Noise measurement},

## Contact

For research collaboration: zahid@unlv.nevada.edu
