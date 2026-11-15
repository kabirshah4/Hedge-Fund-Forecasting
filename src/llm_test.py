"""Step 9 (setup test): run a local LLM (Llama 3.1 8B via Ollama) on headlines.

Checks that the LLM pipeline works before the Thanksgiving LLM experiment.
For each headline the model answers, as JSON:
  sentiment       positive / negative / neutral for the stock
  reports_stake   true if the headline reports an investor's own trade or
                  stake (e.g. "Pershing Square Reports Stake In Agilent").
                  Such headlines must be excluded as leakage: they reveal the
                  very trades we are trying to predict.

Model facts to record: Llama 3.1 8B, training cutoff December 2023 (Meta).

Usage:
  .venv/bin/python src/llm_test.py            # 20 headlines incl. known stake headlines
Output:
  experiments/llm_test.csv
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODEL = "llama3.1:8b"
URL = "http://localhost:11434/api/chat"
PROMPT = (
    "You label financial news headlines. Reply with JSON only, like "
    '{"sentiment": "positive|negative|neutral", "reports_stake": true|false}. '
    "sentiment: is the headline good or bad news for the stock named? "
    "reports_stake: does the headline report that a specific investor, fund or "
    "institution bought, sold, or holds a stake in the stock (for example a 13F, "
    "13D or 13G disclosure)?\nHeadline: "
)


def ask(headline: str) -> dict:
    body = json.dumps({
        "model": MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": PROMPT + headline}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(json.load(r)["message"]["content"])


def main() -> None:
    N = pd.read_parquet(PROC / "news_headlines.parquet", columns=["date", "ticker", "headline"])
    stake = N[N["headline"].str.contains(r"Stake|13F|13D|13G|Takes Position|Boosts Position", case=False, regex=True)]
    sample = pd.concat([stake.sample(8, random_state=1), N.sample(12, random_state=2)]).reset_index(drop=True)
    rows, t0 = [], time.time()
    for h in sample["headline"]:
        try:
            a = ask(h)
        except Exception as err:  # keep going; record the failure
            a = {"sentiment": None, "reports_stake": None, "error": str(err)[:80]}
        rows.append({"headline": h, **a})
    out = pd.DataFrame(rows)
    (ROOT / "experiments").mkdir(exist_ok=True)
    out.to_csv(ROOT / "experiments" / "llm_test.csv", index=False)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 80)
    print(out[["headline", "sentiment", "reports_stake"]].to_string(index=False))
    print(f"\n{len(out)} headlines in {time.time() - t0:.0f}s; "
          f"flagged as reporting a stake: {out['reports_stake'].eq(True).sum()} "
          f"(8 were chosen because they mention stakes/13F)")


if __name__ == "__main__":
    main()
