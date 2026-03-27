"""
Chatter Detection Model
"""
# Imports
# also 'pip install pyarrow' <- required dependency
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Training Datasets
dataWithChatter = ["4500_1.csv", "5500_1.csv", "5500_3.csv", "6000_1.csv"]
dataWithNoChatter = ["7500_1.csv", "7500_2.csv", "7500_3.csv", "8000_1.csv", "8000_4.csv", "8500_1.csv"]

# Testing Datasets
testDataChatter = ["5500_2.csv", "6000_2.csv"]
testDataNoChatter = ["8000_2.csv", "8000_3.csv", "8500_2.csv"]

# Helpers
def extract_rpm(filename):
    return int(os.path.basename(filename).split("_")[0])


def prepare_single(df, rpm):
    """
    Aggregate raw rows sharing the same T into one row per timestep.
    Returns a clean time series with derivatives.
    """
    grouped = df.groupby("T", sort=False)

    agg = grouped.agg(
        AvgX=("X", "mean"), AvgY=("Y", "mean"), AvgZ=("Z", "mean"),
        VarX=("X", "var"),  VarY=("Y", "var"),  VarZ=("Z", "var"),
        MinX=("X", "min"),  MinY=("Y", "min"),  MinZ=("Z", "min"),
        MaxX=("X", "max"),  MaxY=("Y", "max"),  MaxZ=("Z", "max"),
        Count=("X", "count")
    ).reset_index()

    agg["RangeX"] = agg["MaxX"] - agg["MinX"]
    agg["RangeY"] = agg["MaxY"] - agg["MinY"]
    agg["RangeZ"] = agg["MaxZ"] - agg["MinZ"]

    agg = agg.sort_values("T").reset_index(drop=True)
    dt  = agg["T"].diff().replace(0, np.nan)

    for axis in ["X", "Y", "Z"]:
        vel  = agg[f"Avg{axis}"].diff() / dt
        acc  = vel.diff() / dt
        jerk = acc.diff() / dt
        agg[f"Vel{axis}"]  = vel
        agg[f"Acc{axis}"]  = acc
        agg[f"Jerk{axis}"] = jerk

    agg = agg.fillna(0)
    agg["RPM"] = rpm
    return agg


# ── Feature Engineering ───────────────────────────────────────────────────────
def create_windows(df, label, window_size=200, overlap=0.5):
    """
    Slide a window over the time series and extract features per window.
    Some Important Features

      - peak-to-peak:         measures the change in peak-to-peak values in the window
      - kurtosis:             detects impulsive spikes characteristic of chatter
      - crest_factor:         max/rms ratio, spikes during chatter bursts
      - chatter_freq_ratio:   dominant_freq / spindle_freq
                              RPM-invariant; chatter has a stable ratio to spindle freq
      - var_mean (per axis):  average per-timestep variance across the window,
                              captures sensor spread at each moment
      - cross_corr_XY/XZ:    correlation between axes; chatter couples X and Z
                              in a way normal cutting doesn't
      - dominant_freq:          
      - spectral_entropy: 
      - spectral_energy: 
      - high frequency bandpower: 
    """
    step        = int(window_size * (1 - overlap))
    spindle_hz  = df["RPM"].iloc[0] / 60.0       # RPM → Hz, used for freq ratio
    feature_rows = []

    for start in range(0, len(df) - window_size, step):
        end    = start + window_size
        window = df.iloc[start:end]
        f      = {}

        # ── Time domain ───────────────────────────────────────────────────────
        for axis in ["X", "Y", "Z"]:
            s = window[f"Avg{axis}"].values

            f[f"{axis}_mean"]         = np.mean(s)
            f[f"{axis}_std"]          = np.std(s)
            f[f"{axis}_rms"]          = np.sqrt(np.mean(s**2))
            f[f"{axis}_max"]          = np.max(np.abs(s))
            f[f"{axis}_ptp"]          = np.ptp(s)

            # Kurtosis — 4th moment, large when there are sharp impulse spikes
            # Normal vibration ~3, chatter pushes this much higher
            mean_s = np.mean(s)
            std_s  = np.std(s) + 1e-12
            f[f"{axis}_kurtosis"]     = np.mean(((s - mean_s) / std_s) ** 4)

            # Crest factor — peak / rms, high during chatter bursts
            f[f"{axis}_crest_factor"] = np.max(np.abs(s)) / (np.sqrt(np.mean(s**2)) + 1e-12)

            # Average per-timestep variance within the window
            f[f"{axis}_var_mean"]     = window[f"Var{axis}"].mean()

        # ── Derivative magnitudes ─────────────────────────────────────────────
        for name, cols in [("vel", ["VelX","VelY","VelZ"]),
                            ("acc", ["AccX","AccY","AccZ"]),
                            ("jerk",["JerkX","JerkY","JerkZ"])]:
            mag = np.sqrt(sum(window[c].values**2 for c in cols))
            f[f"{name}_mean"] = np.mean(mag)
            f[f"{name}_std"]  = np.std(mag)
            f[f"{name}_max"]  = np.max(mag)

        # ── Cross-axis correlation ─────────────────────────────────────────────
        # Chatter forces X and Z to vibrate in sync; normal cutting doesn't
        f["cross_corr_XZ"] = np.corrcoef(window["AvgX"].values, window["AvgZ"].values)[0, 1]
        f["cross_corr_XY"] = np.corrcoef(window["AvgX"].values, window["AvgY"].values)[0, 1]

        # ── FFT features ──────────────────────────────────────────────────────
        for axis in ["X", "Y", "Z"]:
            s         = window[f"Avg{axis}"].values - np.mean(window[f"Avg{axis}"].values)
            fft_vals  = np.fft.rfft(s)
            fft_power = np.abs(fft_vals) ** 2
            freqs     = np.fft.rfftfreq(len(s))
            total_pow = np.sum(fft_power) + 1e-12

            dom_freq  = freqs[np.argmax(fft_power)]

            # Chatter frequency ratio — dominant freq normalized to spindle freq
            # This is RPM-invariant: if chatter always appears at 2x spindle,
            # this ratio stays ~2.0 regardless of RPM
            f[f"{axis}_chatter_freq_ratio"] = dom_freq / (spindle_hz + 1e-12)

            psd              = fft_power / total_pow
            f[f"{axis}_entropy"]   = -np.sum(psd * np.log(psd + 1e-12))
            f[f"{axis}_energy"]    = total_pow

            # High-frequency band power (top 20%) — chatter lives here
            cutoff = int(len(freqs) * 0.8)
            f[f"{axis}_bandpower"] = np.sum(fft_power[cutoff:]) / total_pow  # normalized

        f["label"] = label
        feature_rows.append(f)

    return pd.DataFrame(feature_rows)


def normalize_by_w(df):
    """
    Divide dynamic features by angular velocity (ω = 0.10472 * RPM).
    This makes features comparable across RPMs so the model doesn't
    just learn that low RPM = chatter.
    """
    df = df.copy()

    if "RPM" in df.columns:
        rpm_vals = df["RPM"]
    elif "rpm_group" in df.columns:
        rpm_vals = df["rpm_group"]
    else:
        raise KeyError("No RPM or rpm_group column found for normalization")

    omega = 0.10471975512 * rpm_vals

    dynamic_cols = [
        c for c in df.columns
        if any(k in c for k in ["vel", "acc", "jerk", "energy", "bandpower", "ptp"])
    ]
    for col in dynamic_cols:
        df[col] = df[col] / omega

    return df


# ── Saving/Loading Features ───────────────────────────────────────────────────────────────────────
def save_features(df, filename="PredictingMachineStatus/processed_features.parquet"):
    df.to_parquet(filename, index=False)
    print(f"Saved → {filename}")

def load_features(filename="PredictingMachineStatus/processed_features.parquet"):
    df = pd.read_parquet(filename)
    print(f"Loaded ← {filename}")
    return df


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(df):
    """
    Split at FILE level using GroupShuffleSplit so no recording leaks
    across train/val. RPM is dropped as a feature — the model must learn
    from vibration patterns, not spindle speed.
    """
    df = df.copy()

    # Drop RPM — we don't want the model learning "4500 = chatter"
    # All dynamic features are already normalized by ω anyway
    X      = df.drop(columns=["label", "rpm_group", "file_group"])
    y      = df["label"]
    groups = df["file_group"]

    # ── Class imbalance ───────────────────────────────────────────────────────
    # You have ~6 chatter files vs ~11 no-chatter files.
    # scale_pos_weight = count(negative) / count(positive) tells XGBoost
    # to penalize missing a chatter detection more heavily.
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    spw   = n_neg / n_pos
    print(f"Class balance — chatter: {n_pos}, no-chatter: {n_neg}, scale_pos_weight: {spw:.2f}")

    # ── File-level split ──────────────────────────────────────────────────────
    # GroupShuffleSplit ensures all windows from one recording stay together.
    # 20% of FILES go to validation (roughly 1-2 chatter, 2-3 no-chatter files)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Sanity checks
    train_files = df["file_group"].iloc[train_idx].unique()
    val_files   = df["file_group"].iloc[val_idx].unique()
    print(f"\nTrain files ({len(train_files)}): {sorted(train_files)}")
    print(f"Val files   ({len(val_files)}):   {sorted(val_files)}")
    print(f"Train label dist:\n{y_train.value_counts()}")
    print(f"Val label dist:\n{y_val.value_counts()}")
    assert not set(train_files) & set(val_files), "FILE LEAKAGE DETECTED"

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,            # slightly shallower to reduce overfitting on small dataset
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,   # handle class imbalance
        eval_metric="aucpr",    # area under precision-recall — better than logloss for imbalanced data
        early_stopping_rounds=30,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    preds  = model.predict(X_val)
    probas = model.predict_proba(X_val)[:, 1]

    print("\n── Validation Report ──")
    print(classification_report(y_val, preds, target_names=["no-chatter", "chatter"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))

    # ── Feature importance ────────────────────────────────────────────────────
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    top20    = feat_imp.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    top20.plot(kind="barh", ax=ax)
    ax.set_title("Top 20 Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("PredictingMachineStatus/feature_importance.png", dpi=150)
    print("\nSaved feature_importance.png")

    # ── Retrain on ALL data for final model ───────────────────────────────────
    print("\nRetraining on full dataset for final model...")
    model.fit(X, y, verbose=False)

    return model, X.columns.tolist()


# ── Final test set evaluation ─────────────────────────────────────────────────
def evaluate_on_test_set(model, feature_cols, test_chatter, test_no_chatter):
    """
    Run ONCE at the very end on files never seen during training or tuning.
    """
    if not test_chatter and not test_no_chatter:
        print("No test files defined — skipping final evaluation.")
        return

    print("\n══════ FINAL HELD-OUT TEST EVALUATION ══════")
    all_windows = []

    for filename in test_chatter + test_no_chatter:
        rpm   = extract_rpm(filename)
        label = 1 if filename in test_chatter else 0

        df  = pd.read_csv(f"dataset/chatter/{filename}", dtype=np.float32, usecols=["T","X","Y","Z"])
        agg = prepare_single(df, rpm)
        w   = create_windows(agg, label)
        w   = normalize_by_w(w)
        w["rpm_group"]  = rpm
        w["file_group"] = filename
        all_windows.append(w)

    test_df = pd.concat(all_windows, ignore_index=True)
    X_test  = test_df[feature_cols]
    y_test  = test_df["label"]

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["no-chatter", "chatter"]))
    print(confusion_matrix(y_test, preds))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_files     = dataWithChatter + dataWithNoChatter
    FORCE_REBUILD = False

    parquet_path = "PredictingMachineStatus/processed_features.parquet"

    if not FORCE_REBUILD and os.path.exists(parquet_path):
        training_df = load_features(parquet_path)
    else:
        # Build from scratch
        all_windows = []

        for filename in all_files:
            rpm   = extract_rpm(filename)
            label = 1 if filename in dataWithChatter else 0
            print(f"  {filename} | RPM={rpm} | label={'chatter' if label else 'no-chatter'}")

            df  = pd.read_csv(f"dataset/chatter/{filename}", dtype=np.float32, usecols=["T","X","Y","Z"])
            agg = prepare_single(df, rpm)
            w   = create_windows(agg, label)

            w["rpm_group"]  = rpm
            w["file_group"] = filename

            all_windows.append(w)

        training_df = pd.concat(all_windows, ignore_index=True)
        print(f"\nRaw windows shape: {training_df.shape}")
        print(f"Columns: {training_df.columns.tolist()}")

        training_df = normalize_by_w(training_df)
        save_features(training_df, parquet_path)

    print(f"\nTraining df shape: {training_df.shape}")
    print(f"Label distribution:\n{training_df['label'].value_counts()}")

    model, feature_cols = train_model(training_df)
    evaluate_on_test_set(model, feature_cols, testDataChatter, testDataNoChatter)