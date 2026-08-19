"""
Preprocessing for CIC-IDS2017/2018-style traffic CSVs.

- Drops identifier columns (would let the model "cheat" by memorizing IPs)
- Cleans inf/NaN values (known issue in CICFlowMeter output, e.g. Flow Bytes/s)
- Splits into a BENIGN-only training set (unsupervised training) and a
  mixed labeled test set (for honest evaluation against known attacks)
- Scales features and saves the scaler for reuse at inference time

Usage:
    python3 preprocess.py --input data/sample_traffic.csv --outdir data/processed
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IDENTIFIER_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port", "Destination Port"]
LABEL_COL = "Label"


def load_and_clean(path: str) -> pd.DataFrame:
    """Returns the full cleaned dataframe INCLUDING identifier columns.
    Callers should separate feature_cols (numeric, for the model) from
    identifier columns (kept only for dashboard display) themselves.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # CIC CSVs sometimes have leading spaces

    # replace inf with NaN, then drop rows with any NaN (CICFlowMeter divide-by-zero artifacts)
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with inf/NaN values")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_clean(args.input)
    identifier_cols_present = [c for c in IDENTIFIER_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c != LABEL_COL and c not in identifier_cols_present]
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features (excluding {len(identifier_cols_present)} identifier cols)")
    print(df[LABEL_COL].value_counts())

    benign = df[df[LABEL_COL] == "BENIGN"]
    attacks = df[df[LABEL_COL] != "BENIGN"]

    # Train Isolation Forest / Autoencoder ONLY on benign traffic (unsupervised anomaly detection:
    # the model learns what "normal" looks like, never sees attack examples during training)
    benign_train, benign_test = train_test_split(
        benign, test_size=args.test_size, random_state=42
    )

    # Test set = held-out benign + ALL attack rows, so evaluation reflects real-world mix
    test_df = pd.concat([benign_test, attacks], ignore_index=True).sample(
        frac=1, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(benign_train[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    # y_test: 1 = anomaly (attack), 0 = normal -- for evaluation only, never used in training
    y_test = (test_df[LABEL_COL] != "BENIGN").astype(int).values

    np.save(os.path.join(args.outdir, "X_train.npy"), X_train)
    np.save(os.path.join(args.outdir, "X_test.npy"), X_test)
    np.save(os.path.join(args.outdir, "y_test.npy"), y_test)
    test_df.to_csv(os.path.join(args.outdir, "test_raw.csv"), index=False)
    joblib.dump(scaler, os.path.join(args.outdir, "scaler.joblib"))

    with open(os.path.join(args.outdir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nTrain set (benign only): {X_train.shape}")
    print(f"Test set (mixed): {X_test.shape}  |  anomaly rate: {y_test.mean():.2%}")
    print(f"Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
