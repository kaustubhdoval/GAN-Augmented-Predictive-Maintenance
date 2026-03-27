# Predicting Machine Status

**NOTE:** The dataset used is Confidential and has not been uploaded to Github

## Approach

#### 1. Prepare Data

1. For each unique T entry compute -> AvgX, AvgY, AvgZ, VarX, VarY, VarZ, JerkX ... AccX, .. VelX, ...
2. Create Windowed Frames (variable window size) - For each window compute mean, std, rms, max-min, avgVel, avgAcc, avgJerk, and FFT (Dominant Frequency, Spectral Energy, Spectral Entropy)

- Remember to NORMALIZE vibration data with RPM

#### 2. Train Model

- XGBoost - Split Train/Test wrt RPM
- Data Discrepancy: chatter only at low RPM and good operation only at high RPM. So:
  - Split at FILE level
  - Since we also have a lot more not-chatter data we need to penalize missing chatter heavily

## Chatter Dataset:

All of the Data belongs to one End Mill
| Dataset | Chatter (Yes/No) |
|---|---|
|4500_1.csv|Y|
|5500_1.csv |Y|
|5500_2.csv |Y|
|5500_3.csv |Y|
|6000_1.csv |Y|
|6000_2.csv |Y|
|7500_1.csv |N|
|7500_2.csv |N|
|7500_3.csv |N|
|8000_1.csv |N|
|8000_2.csv |N|
|8000_3.csv |N|
|8000_4.csv |N|
|8500_1.csv |N|
|8500_2.csv |N|
|8500_3.csv |N|
|8500_4.csv |N|
|9000_1.csv |N|
|9000_2.csv |N|
|9500_1.csv |N|
|9500_2.csv |N|
