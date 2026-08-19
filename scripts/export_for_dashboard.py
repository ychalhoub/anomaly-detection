"""
Scores the test set with the trained model and exports a JSON file the
Node.js dashboard can read directly (no live Python inference needed for the demo).

Usage:
    python3 export_for_dashboard.py --data data/processed --model models/isolation_forest.joblib
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd


def severity_label(score: float, threshold: float) -> str:
    if score < threshold * 0.5:
        return "low"
    elif score < threshold:
        return "medium"
    elif score < threshold * 1.5:
        return "high"
    return "critical"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--model", default="models/isolation_forest.joblib")
    parser.add_argument("--metrics", default="models/metrics.json")
    parser.add_argument("--out", default="dashboard/public/flows.json")
    parser.add_argument("--limit", type=int, default=500, help="cap rows for a snappy demo UI")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = joblib.load(args.model)
    X_test = np.load(os.path.join(args.data, "X_test.npy"))
    y_test = np.load(os.path.join(args.data, "y_test.npy"))
    test_raw = pd.read_csv(os.path.join(args.data, "test_raw.csv"))

    with open(args.metrics) as f:
        metrics = json.load(f)
    threshold = metrics["best_f1_threshold"]

    anomaly_scores = -model.score_samples(X_test)
    predictions = (model.predict(X_test) == -1).astype(int)

    rows = []
    n = min(args.limit, len(test_raw))
    for i in range(n):
        rows.append(
            {
                "id": i,
                "sourceIp": test_raw.iloc[i].get("Source IP", "N/A"),
                "destIp": test_raw.iloc[i].get("Destination IP", "N/A"),
                "flowDuration": float(test_raw.iloc[i].get("Flow Duration", 0)),
                "packets": int(test_raw.iloc[i].get("Total Fwd Packets", 0)),
                "trueLabel": str(test_raw.iloc[i].get("Label", "UNKNOWN")),
                "anomalyScore": round(float(anomaly_scores[i]), 4),
                "predictedAnomaly": bool(predictions[i]),
                "severity": severity_label(float(anomaly_scores[i]), threshold),
            }
        )

    output = {
        "generatedAt": pd.Timestamp.now().isoformat(),
        "threshold": threshold,
        "metrics": {
            "rocAuc": metrics["roc_auc"],
            "testAnomalyRate": metrics["test_anomaly_rate"],
            "confusionMatrix": metrics["confusion_matrix"],
        },
        "flows": rows,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported {n} scored flows to {args.out}")
    print(f"Flagged as anomalous: {sum(r['predictedAnomaly'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
