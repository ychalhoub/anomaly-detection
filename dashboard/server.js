/**
 * Express backend for the network anomaly detection dashboard.
 *
 * Serves pre-scored flow data (produced by scripts/export_for_dashboard.py)
 * and simulates a "live" traffic feed by replaying flows on a timer -- no
 * live Python inference needed, which keeps the demo simple and fast.
 */

const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

const DATA_PATH = path.join(__dirname, "public", "flows.json");

function loadData() {
  const raw = fs.readFileSync(DATA_PATH, "utf-8");
  return JSON.parse(raw);
}

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json());

// Full dataset + model metrics (for summary cards / charts)
app.get("/api/summary", (req, res) => {
  const data = loadData();
  res.json({
    generatedAt: data.generatedAt,
    threshold: data.threshold,
    metrics: data.metrics,
    totalFlows: data.flows.length,
    flaggedCount: data.flows.filter((f) => f.predictedAnomaly).length,
  });
});

// All flows, optionally filtered by severity
app.get("/api/flows", (req, res) => {
  const data = loadData();
  let flows = data.flows;
  const { severity, anomaliesOnly } = req.query;

  if (severity) {
    flows = flows.filter((f) => f.severity === severity);
  }
  if (anomaliesOnly === "true") {
    flows = flows.filter((f) => f.predictedAnomaly);
  }
  res.json({ count: flows.length, flows });
});

// Simulated "live" feed: returns the next N flows starting at an offset,
// so the frontend can poll this to fake a streaming feed for the demo.
let liveCursor = 0;
app.get("/api/live", (req, res) => {
  const data = loadData();
  const batchSize = parseInt(req.query.batch || "5", 10);
  const batch = [];
  for (let i = 0; i < batchSize; i++) {
    batch.push(data.flows[liveCursor % data.flows.length]);
    liveCursor++;
  }
  res.json({ cursor: liveCursor, flows: batch });
});

app.listen(PORT, () => {
  console.log(`Anomaly detection dashboard running at http://localhost:${PORT}`);
});
