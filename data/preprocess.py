"""
ORION — Phase 1: Preprocessing Pipeline
========================================
Converts raw per-stay Parquet files into:
  - Normalized, windowed sequences (last N hours)
  - Train / validation / test splits (temporal, patient-level)
  - PyTorch-ready tensors saved as .pt files

Usage:
    python data/preprocess.py --ts_dir data/processed/timeseries \
                               --out_dir data/processed/tensors \
                               --window 24 --horizon 6
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

FEATURE_COLS = [
    "heart_rate", "sbp", "dbp", "spo2", "temp_f", "resp_rate",
    "lactate", "wbc", "creatinine", "bicarbonate", "glucose",
]

# Physiologically reasonable clipping ranges
CLIP_RANGES = {
    "heart_rate":   (20,  300),
    "sbp":          (40,  280),
    "dbp":          (20,  200),
    "spo2":         (50,  100),
    "temp_f":       (85,  115),
    "resp_rate":    (4,   60),
    "lactate":      (0,   30),
    "wbc":          (0,   150),
    "creatinine":   (0,   30),
    "bicarbonate":  (0,   50),
    "glucose":      (20,  1200),
}


def load_stay(path: Path, window: int) -> np.ndarray | None:
    """
    Load a stay Parquet, forward-fill missing values,
    return the LAST `window` hours as an array [window × features].
    """
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FEATURE_COLS].copy()

    # Clip physiological outliers
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Forward-fill then backward-fill, then fill remaining with column median
    df = df.ffill().bfill()
    df = df.fillna(df.median())

    if len(df) < window:
        # Pad with row-median at the front
        pad_size = window - len(df)
        pad_row = df.median().values
        padding = np.tile(pad_row, (pad_size, 1))
        arr = np.vstack([padding, df.values])
    else:
        arr = df.values[-window:]  # take LAST window hours

    return arr.astype(np.float32)


def build_dataset(ts_dir: Path, outcomes_path: Path,
                  window: int, horizon: int):
    outcomes = pd.read_parquet(outcomes_path)
    outcome_map = dict(zip(outcomes["stay_id"], outcomes["mortality"]))

    parquet_files = list(ts_dir.glob("*.parquet"))
    print(f"→ Found {len(parquet_files)} stay files")

    X, y, stay_ids = [], [], []
    for pf in tqdm(parquet_files):
        sid = int(pf.stem)
        if sid not in outcome_map:
            continue
        arr = load_stay(pf, window)
        if arr is None:
            continue
        X.append(arr)
        y.append(outcome_map[sid])
        stay_ids.append(sid)

    X = np.stack(X, axis=0)          # [N, window, features]
    y = np.array(y, dtype=np.float32)
    stay_ids = np.array(stay_ids)
    return X, y, stay_ids


def normalize(X_train, X_val, X_test, out_dir: Path):
    """Fit StandardScaler on train, apply to all splits. Save scaler."""
    N_tr, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, F)
    scaler.fit(X_train_flat)

    def transform(X):
        n, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, t, f)

    X_train = transform(X_train)
    X_val   = transform(X_val)
    X_test  = transform(X_test)
    joblib.dump(scaler, out_dir / "scaler.pkl")
    print(f"✓ Scaler saved → {out_dir}/scaler.pkl")
    return X_train, X_val, X_test


def main(ts_dir, out_dir, window, horizon):
    ts_dir   = Path(ts_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes_path = ts_dir.parent / "outcomes.parquet"
    X, y, stay_ids = build_dataset(ts_dir, outcomes_path, window, horizon)
    print(f"✓ Dataset shape: X={X.shape}, y={y.shape}  (pos rate={y.mean():.3f})")

    # Temporal split: first 70% train, next 15% val, last 15% test
    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.85)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:], y[n_val:]

    X_train, X_val, X_test = normalize(X_train, X_val, X_test, out_dir)

    # Save as PyTorch tensors
    splits = {"train": (X_train, y_train),
              "val":   (X_val,   y_val),
              "test":  (X_test,  y_test)}
    for split, (Xs, ys) in splits.items():
        torch.save({
            "X": torch.tensor(Xs, dtype=torch.float32),
            "y": torch.tensor(ys, dtype=torch.float32),
        }, out_dir / f"{split}.pt")
        print(f"✓ {split}.pt  — {Xs.shape[0]} samples")

    # Save metadata
    meta = {"window": window, "horizon": horizon,
            "n_features": X.shape[2], "feature_cols": FEATURE_COLS,
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)}
    import json
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\n✅ Preprocessing complete. Next: python models/temporal/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts_dir",  default="data/processed/timeseries")
    parser.add_argument("--out_dir", default="data/processed/tensors")
    parser.add_argument("--window",  type=int, default=24)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()
    main(args.ts_dir, args.out_dir, args.window, args.horizon)
