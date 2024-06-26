GMAN Training on Masked Data

This version of GMAN is trained on masked data using two types of datasets:

    Training Dataset: Contains both missing values and imputed values.
    Historical Data: Missing values are filled with 0 or -1, masking the data.
    Prediction Data: Contains imputed values, providing the closest approximation to real data.

The model aims to learn to predict imputed values from masked data.
