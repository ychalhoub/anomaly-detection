"""
Merges multiple CIC-IDS2017 day CSVs into a single file, which the
preprocess.py script expects (it splits benign vs. attack internally
based on the Label column).

Usage:
    python3 merge_days.py --files data/raw/Monday-WorkingHours.pcap_ISCX.csv data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv --out data/raw/merged.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="paths to day CSVs to merge")
    parser.add_argument("--out", default="data/raw/merged.csv")
    args = parser.parse_args()

    dfs = []
    for path in args.files:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]  # CIC CSVs often have leading spaces in headers
        print(f"{path}: {len(df)} rows, labels: {df['Label'].value_counts().to_dict()}")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    merged.to_csv(args.out, index=False)
    print(f"\nWrote {len(merged)} total rows to {args.out}")
    print(merged["Label"].value_counts())


if __name__ == "__main__":
    main()
