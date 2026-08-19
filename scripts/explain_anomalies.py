"""
Generates plain-English explanations for flagged anomalous flows using the
Google Gemini API (free tier), and adds them to the dashboard's flows.json.

Requires a free Gemini API key. Get one at https://aistudio.google.com/apikey
Then either:
    export GEMINI_API_KEY=your-key-here     (Mac/Linux)
    set GEMINI_API_KEY=your-key-here        (Windows cmd)
    $env:GEMINI_API_KEY="your-key-here"     (Windows PowerShell)

Or pass it directly with --api-key.

Usage:
    python3 explain_anomalies.py --flows dashboard/public/flows.json
"""

import argparse
import json
import os
import time

from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are a security analyst assistant for a small business owner
with no technical background. Given stats about a flagged network flow, write ONE
short sentence (under 25 words) explaining in plain English why it looks suspicious
and what kind of threat it might resemble. Avoid jargon like "flow", "packet rate",
or "anomaly score" -- describe it the way you'd explain it to a shop owner.
Do not use hedging phrases like "might be" more than once. Be direct and concrete."""


def is_valid_explanation(text: str) -> bool:
    """Checks whether an existing explanation looks complete, not truncated or an error."""
    if not text or text.startswith("(explanation unavailable"):
        return False
    if len(text) < 15:
        return False
    if text.rstrip()[-1] not in ".!?":
        return False  # looks cut off mid-sentence
    return True


def build_prompt(flow: dict) -> str:
    return (
        f"Severity: {flow['severity']}\n"
        f"Anomaly score: {flow['anomalyScore']}\n"
        f"Packets sent: {flow['packets']}\n"
        f"Duration (microseconds): {flow['flowDuration']}\n"
        f"Source: {flow['sourceIp']}  Destination: {flow['destIp']}\n\n"
        "Explain why this looks suspicious in one short sentence."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", default="dashboard/public/flows.json")
    parser.add_argument("--api-key", default=None, help="Overrides GEMINI_API_KEY env var")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--limit", type=int, default=None, help="cap how many flagged flows get explanations")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Set the GEMINI_API_KEY environment variable "
            "or pass --api-key. Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    with open(args.flows) as f:
        data = json.load(f)

    flagged = [fl for fl in data["flows"] if fl["predictedAnomaly"]]

    # Prioritize the most severe flows first, so a limited budget of API calls
    # covers the flows that matter most for the demo (critical > high > medium > low)
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    flagged.sort(key=lambda fl: severity_rank.get(fl["severity"], 4))

    if args.limit:
        flagged = flagged[: args.limit]

    print(f"Generating explanations for {len(flagged)} flagged flows...")
    print(f"(Using {args.model}, ~5s per flow)")

    for i, flow in enumerate(flagged):
        if "explanation" in flow and is_valid_explanation(flow["explanation"]):
            continue  # already has a good, complete explanation, skip to save API calls
        explanation = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=build_prompt(flow),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=200,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                explanation = response.text.strip()
                break
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    wait = 15 * (attempt + 1)  # back off: 15s, 30s, 45s
                    print(f"  Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                explanation = f"(explanation unavailable: {e})"
                break
        if explanation is None:
            explanation = "(explanation unavailable: rate limit exceeded after retries)"

        for fl in data["flows"]:
            if fl["id"] == flow["id"]:
                fl["explanation"] = explanation
                break

        if (i + 1) % 10 == 0 or (i + 1) == len(flagged):
            print(f"  {i + 1}/{len(flagged)} done")

        time.sleep(5)  # 15 req/min allowed on this model -> ~4s minimum, 5s for safety margin

    with open(args.flows, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved explanations back to {args.flows}")


if __name__ == "__main__":
    main()
