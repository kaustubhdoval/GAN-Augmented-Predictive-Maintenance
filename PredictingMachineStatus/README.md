# Predicting Machine Status

## Pipeline Overview

1. **Data Loading**: Raw CSV files (columns: T, X, Y, Z) are loaded from the dataset directory.
2. **Aggregation**: For each unique T, sensor readings are aggregated (mean, variance, min, max, etc.) per axis (X, Y, Z).
3. **Windowing**: The time series is split into overlapping windows. Features are computed for each window.
4. **Feature Extraction**: Multiple features are calculated per window and per axis (see table below).
5. **Normalization**: Dynamic features are normalized by angular velocity (ω) to ensure comparability across RPMs.
6. **Model Training**: XGBoost classifier is trained using file-level splits to prevent data leakage.
7. **Evaluation & Output**: Model performance is evaluated, and feature importances and processed features are saved.

#### Feature Normalization

Dynamic features (e.g., velocity, acceleration, jerk, energy, bandpower, peak-to-peak) are normalized by the angular velocity (ω = 0.10472 × RPM). This is crucial because vibration characteristics can scale with spindle speed. Normalization ensures the model learns patterns related to chatter, not just differences in RPM, making features RPM-invariant and improving generalization.

#### Model Training & File-Level Split

The model is trained using an XGBoost classifier. To avoid data leakage, the dataset is split at the file level: all windows from a given recording are kept together in either the training or validation set. This ensures the model is evaluated on truly unseen data. Class imbalance is handled by penalizing missed chatter detections more heavily (using XGBoost's scale_pos_weight parameter).

#### Output

- **Processed Features**: Saved as a Parquet file (`processed_features.parquet`) for efficient reuse.
- **Feature Importances**: Top features are visualized and saved as `feature_importance.png`.
- **Evaluation Metrics**: Classification reports and confusion matrices are printed to the console.

## Features

Most features are computed separately for each axis (X, Y, Z), resulting in columns like `X_mean`, `Y_mean`, `Z_mean`, etc. Some features (e.g., cross-axis correlations) are computed between axes.

| Feature                       | Description                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| mean                          | Mean value in the window                                                                          |
| std                           | Standard deviation in the window                                                                  |
| rms                           | Root mean square value in the window                                                              |
| max                           | Maximum absolute value in the window                                                              |
| ptp (peak-to-peak)            | Difference between max and min values in the window                                               |
| kurtosis                      | 4th moment, detects impulsive spikes characteristic of chatter                                    |
| crest_factor                  | Max/RMS ratio, spikes during chatter bursts                                                       |
| var_mean                      | Average per-timestep variance across the window, captures sensor spread at each moment            |
| vel_mean, vel_std, vel_max    | Magnitude of velocity vector (computed from derivatives)                                          |
| acc_mean, acc_std, acc_max    | Magnitude of acceleration vector                                                                  |
| jerk_mean, jerk_std, jerk_max | Magnitude of jerk vector                                                                          |
| cross_corr_XY, cross_corr_XZ  | Correlation between axes; chatter couples X and Z in a way normal cutting doesn't                 |
| dominant_freq                 | Dominant frequency in the window (from FFT)                                                       |
| chatter_freq_ratio            | Dominant frequency / spindle frequency; RPM-invariant; chatter has a stable ratio to spindle freq |
| entropy (spectral_entropy)    | Entropy of the frequency spectrum, measures signal complexity                                     |
| energy (spectral_energy)      | Total energy in the frequency spectrum                                                            |
| bandpower (high freq)         | Power in the high-frequency band (top 20% of spectrum), often elevated during chatter             |
