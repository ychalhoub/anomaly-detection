"""
Trains an Isolation Forest on benign-only traffic, then evaluates against
a held-out mixed (benign + attack) test set.

Usage:
    python3 train_isolation_forest.py --data data/processed --outdir models
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected proportion of anomalies in TRAINING data (should be low since we train on benign-only)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X_train = np.load(os.path.join(args.data, "X_train.npy"))
    X_test = np.load(os.path.join(args.data, "X_test.npy"))
    y_test = np.load(os.path.join(args.data, "y_test.npy"))

    print(f"Training Isolation Forest on {X_train.shape[0]} benign flows...")
    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # score_samples: higher = more normal. We flip sign so higher = more anomalous,
    # which is more intuitive for a "severity score" in the dashboard.
    raw_scores = model.score_samples(X_test)
    anomaly_scores = -raw_scores  # higher = more anomalous
    predictions = model.predict(X_test)  # 1 = normal, -1 = anomaly
    y_pred = (predictions == -1).astype(int)

    print("\n=== Classification report (0=normal, 1=attack) ===")
    print(classification_report(y_test, y_pred, digits=3))

    auc = roc_auc_score(y_test, anomaly_scores)
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # find the threshold that maximizes F1, for reference / dashboard severity cutoffs
    precisions, recalls, thresholds = precision_recall_curve(y_test, anomaly_scores)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.0
    print(f"Best-F1 threshold on anomaly score: {best_threshold:.4f} (F1={f1_scores[best_idx]:.3f})")

    # save model + metrics
    joblib.dump(model, os.path.join(args.outdir, "isolation_forest.joblib"))
    metrics = {
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "best_f1_threshold": best_threshold,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "test_anomaly_rate": float(y_test.mean()),
        "params": {"n_estimators": args.n_estimators, "contamination": args.contamination},
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + metrics to {args.outdir}/")


if __name__ == "__main__":
    main()
