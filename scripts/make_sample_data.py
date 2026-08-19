"""
Generates a synthetic CSV that mimics the CIC-IDS2017 schema so the rest of the
pipeline can be built and tested before the real dataset is downloaded.

Real data: https://www.unb.ca/cic/datasets/ids-2017.html
Download the "MachineLearningCSV.zip" (already has CICFlowMeter features + labels).
Once downloaded, drop the real CSVs into data/raw/ and this script is no longer needed.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# A representative subset of the real CIC-IDS2017 columns.
# (The real files have ~80 columns; these are the ones that matter most for detection.)
FEATURE_COLS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Average Packet Size",
]

NON_FEATURE_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]


def make_benign(n):
    df = pd.DataFrame(
        {
            "Flow Duration": RNG.normal(500000, 150000, n).clip(1000),
            "Total Fwd Packets": RNG.poisson(12, n),
            "Total Backward Packets": RNG.poisson(10, n),
            "Total Length of Fwd Packets": RNG.normal(1200, 300, n).clip(50),
            "Total Length of Bwd Packets": RNG.normal(1400, 350, n).clip(50),
            "Fwd Packet Length Mean": RNG.normal(100, 20, n).clip(20),
            "Bwd Packet Length Mean": RNG.normal(110, 25, n).clip(20),
            "Flow Bytes/s": RNG.normal(5000, 1500, n).clip(0),
            "Flow Packets/s": RNG.normal(40, 12, n).clip(0),
            "Flow IAT Mean": RNG.normal(40000, 10000, n).clip(0),
            "Flow IAT Std": RNG.normal(15000, 5000, n).clip(0),
            "Fwd IAT Mean": RNG.normal(38000, 9000, n).clip(0),
            "Bwd IAT Mean": RNG.normal(42000, 9500, n).clip(0),
            "SYN Flag Count": RNG.poisson(1, n),
            "ACK Flag Count": RNG.poisson(8, n),
            "PSH Flag Count": RNG.poisson(3, n),
            "Average Packet Size": RNG.normal(105, 20, n).clip(20),
        }
    )
    df["Label"] = "BENIGN"
    return df


def make_ddos(n):
    # DDoS: tons of packets, very short duration, high flag counts, high pkt/s
    df = pd.DataFrame(
        {
            "Flow Duration": RNG.normal(2000, 800, n).clip(1),
            "Total Fwd Packets": RNG.poisson(300, n),
            "Total Backward Packets": RNG.poisson(2, n),
            "Total Length of Fwd Packets": RNG.normal(15000, 4000, n).clip(500),
            "Total Length of Bwd Packets": RNG.normal(100, 50, n).clip(0),
            "Fwd Packet Length Mean": RNG.normal(50, 10, n).clip(10),
            "Bwd Packet Length Mean": RNG.normal(20, 10, n).clip(0),
            "Flow Bytes/s": RNG.normal(500000, 120000, n).clip(0),
            "Flow Packets/s": RNG.normal(4000, 900, n).clip(0),
            "Flow IAT Mean": RNG.normal(10, 5, n).clip(0),
            "Flow IAT Std": RNG.normal(5, 2, n).clip(0),
            "Fwd IAT Mean": RNG.normal(9, 4, n).clip(0),
            "Bwd IAT Mean": RNG.normal(200, 80, n).clip(0),
            "SYN Flag Count": RNG.poisson(30, n),
            "ACK Flag Count": RNG.poisson(1, n),
            "PSH Flag Count": RNG.poisson(0.5, n),
            "Average Packet Size": RNG.normal(45, 12, n).clip(10),
        }
    )
    df["Label"] = "DDoS"
    return df


def make_portscan(n):
    # Port scan: many short flows, near-zero payload, high SYN, near-zero backward packets
    df = pd.DataFrame(
        {
            "Flow Duration": RNG.normal(500, 200, n).clip(1),
            "Total Fwd Packets": RNG.poisson(2, n).clip(1),
            "Total Backward Packets": RNG.poisson(0.3, n),
            "Total Length of Fwd Packets": RNG.normal(60, 20, n).clip(0),
            "Total Length of Bwd Packets": RNG.normal(5, 5, n).clip(0),
            "Fwd Packet Length Mean": RNG.normal(40, 10, n).clip(0),
            "Bwd Packet Length Mean": RNG.normal(5, 5, n).clip(0),
            "Flow Bytes/s": RNG.normal(3000, 1500, n).clip(0),
            "Flow Packets/s": RNG.normal(60, 25, n).clip(0),
            "Flow IAT Mean": RNG.normal(300, 150, n).clip(0),
            "Flow IAT Std": RNG.normal(50, 20, n).clip(0),
            "Fwd IAT Mean": RNG.normal(280, 140, n).clip(0),
            "Bwd IAT Mean": RNG.normal(0, 1, n).clip(0),
            "SYN Flag Count": RNG.poisson(1.8, n),
            "ACK Flag Count": RNG.poisson(0.2, n),
            "PSH Flag Count": RNG.poisson(0.1, n),
            "Average Packet Size": RNG.normal(35, 10, n).clip(0),
        }
    )
    df["Label"] = "PortScan"
    return df


def main():
    n_benign, n_ddos, n_portscan = 8000, 500, 400
    df = pd.concat(
        [make_benign(n_benign), make_ddos(n_ddos), make_portscan(n_portscan)],
        ignore_index=True,
    )

    # shuffle rows like a real capture would be time-interleaved
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # add fake identifier columns like the real dataset has (dropped during preprocessing)
    df.insert(0, "Flow ID", [f"10.0.0.{RNG.integers(2,254)}-192.168.1.{RNG.integers(2,254)}-{RNG.integers(1024,65535)}-{RNG.integers(1,1024)}-6" for _ in range(len(df))])
    df.insert(1, "Source IP", [f"10.0.0.{RNG.integers(2,254)}" for _ in range(len(df))])
    df.insert(2, "Destination IP", [f"192.168.1.{RNG.integers(2,254)}" for _ in range(len(df))])
    df.insert(3, "Timestamp", pd.date_range("2026-08-14 08:00:00", periods=len(df), freq="s"))

    out_path = "/home/claude/anomaly-detection/data/sample_traffic.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["Label"].value_counts())


if __name__ == "__main__":
    main()
