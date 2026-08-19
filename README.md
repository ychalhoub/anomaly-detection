# Network Anomaly Detection for Lebanese SMBs

LebNet Tech Fellows final project — Option 2 (AI for Lebanon).

## Problem

Lebanese SMBs (retail, clinics, small service providers) cannot afford enterprise-grade
security monitoring (CrowdStrike, Darktrace, Cisco ISE, etc.), which typically costs
thousands of dollars annually and requires dedicated IT staff. Since the economic crisis,
this gap has widened: shrinking IT budgets, more legacy/unpatched infrastructure, fewer
in-house security resources — leaving small businesses disproportionately exposed to
network-based attacks.

## Approach

An open-source, low-cost network anomaly detection pipeline:

1. **Model**: Isolation Forest (unsupervised), trained ONLY on benign traffic, so it
   never needs labeled attack examples to detect new/unknown threats.
2. **Data**: CIC-IDS2017/2018 benchmark datasets (flow-level features extracted via
   CICFlowMeter — duration, packet counts, byte rates, TCP flags, etc.)
3. **Dashboard**: Lightweight Node.js/Express + vanilla JS frontend. No security
   expertise required to read — flows are flagged with a plain severity label
   (low/medium/high/critical), not raw model internals.

Designed for the deployment constraint that actually matters for Lebanese SMBs:
runs on a single low-spec machine, no subscription, no dedicated analyst required.

## Project structure

```
anomaly-detection/
├── data/
│   ├── sample_traffic.csv       # synthetic data (schema matches CIC-IDS2017) for pipeline testing
│   └── processed/               # train/test splits, scaler (generated)
├── scripts/
│   ├── make_sample_data.py      # generates synthetic test data (delete once real CIC-IDS data is in use)
│   ├── preprocess.py            # cleans data, splits benign-only train / mixed test
│   ├── train_isolation_forest.py# trains model, evaluates, saves metrics
│   └── export_for_dashboard.py  # scores test set, exports JSON for the dashboard
├── models/                      # trained model + metrics.json (generated)
└── dashboard/
    ├── server.js                # Express API (summary, flows, simulated live feed)
    └── public/
        ├── index.html           # dashboard UI
        └── flows.json           # scored flow data consumed by the UI (generated)
```

## Setup with the REAL dataset

1. Download CIC-IDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html
   (get `MachineLearningCSV.zip` — already has CICFlowMeter features + labels)
2. Place the CSV(s) in `data/raw/` (pick 1-2 days, e.g. a normal day + the DDoS day,
   to keep training fast)
3. Run the pipeline:

```bash
python3 scripts/preprocess.py --input data/raw/YOUR_FILE.csv --outdir data/processed
python3 scripts/train_isolation_forest.py --data data/processed --outdir models
python3 scripts/export_for_dashboard.py --data data/processed --model models/isolation_forest.joblib --out dashboard/public/flows.json
```

4. Run the dashboard:

```bash
cd dashboard
npm install
node server.js
# open http://localhost:3000
```

## Optional: LLM-generated explanations

To add plain-English explanations for each flagged anomaly (shown in the dashboard's
"Why flagged" column):

1. Get a free API key at https://aistudio.google.com/apikey
2. Set it as an environment variable:
   ```bash
   set GEMINI_API_KEY=your-key-here          # Windows cmd
   $env:GEMINI_API_KEY="your-key-here"       # Windows PowerShell
   export GEMINI_API_KEY=your-key-here       # Mac/Linux
   ```
3. Run it AFTER exporting flows for the dashboard (it edits flows.json in place):
   ```bash
   pip install google-genai
   python3 scripts/explain_anomalies.py --flows dashboard/public/flows.json
   ```
   Use `--limit 30` to only generate explanations for the first 30 flagged flows
   (keeps it fast and stays comfortably within the free tier).
4. Restart the dashboard (`node server.js`) to see the explanations.

## Evaluation

Since CIC-IDS2017/2018 provides ground-truth attack labels, the model is evaluated with:
- Precision / Recall / F1 (attack = positive class)
- ROC-AUC on the anomaly score
- Confusion matrix

These are computed automatically by `train_isolation_forest.py` and saved to `models/metrics.json`.

## Limitations (be upfront about this in the report)

CIC-IDS2017/2018 is a general-purpose benchmark, not real Lebanese business traffic.
Future work: fine-tune / re-validate on real traffic captured from an actual Lebanese
SMB network once a partner site is available.
