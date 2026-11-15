"""Step 4a: map stock IDs (CUSIPs) to tickers, and fetch S&P 500 membership history.

13F filings name stocks by CUSIP (a 9-character ID). Price data and index
lists use tickers (AAPL). This script:
  1. picks the CUSIPs worth mapping: held by at least MIN_HOLDERS 13F filers in
     some quarter (a superset of the S&P 500, which thousands of filers hold),
  2. asks OpenFIGI (free, from Bloomberg) for each CUSIP's US ticker, 10 per
     request, staying under the free limit of 25 requests a minute,
  3. downloads S&P 500 membership history (ticker, date added, date removed)
     from github.com/fja05680/sp500 (MIT license).

Saves progress as it goes, so it can be stopped and rerun.

Outputs:
  data/processed/cusip_map.parquet   cusip, ticker, name, security_type, figi
  data/raw/sp500/sp500_ticker_start_end.csv

Usage:
  .venv/bin/python src/map_tickers.py
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = PROC / "cusip_map.parquet"
MIN_HOLDERS = 300
URL = "https://api.openfigi.com/v3/mapping"
BATCH = 10  # max jobs per request without an API key
PAUSE = 2.6  # seconds between requests: ~23 a minute, under the 25 limit
SP500_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)


def post(jobs: list[dict]) -> list[dict]:
    req = urllib.request.Request(
        URL,
        data=json.dumps(jobs).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception as err:  # rate limit (429) or a network blip: wait, retry
            wait = 60 if "429" in str(err) else 5 * (attempt + 1)
            print(f"  retry in {wait}s ({err})")
            time.sleep(wait)
    raise RuntimeError("OpenFIGI kept failing")


def best_match(results: list[dict]) -> dict | None:
    """Prefer the US composite listing of a common stock."""
    if not results:
        return None
    us = [r for r in results if r.get("exchCode") == "US"] or results
    common = [r for r in us if "Common" in (r.get("securityType") or "")] or us
    return common[0]


def main() -> None:
    labels = pd.read_parquet(PROC / "labels.parquet", columns=["cusip"])
    prices = pd.read_parquet(PROC / "implied_prices.parquet")
    widely_held = prices.groupby("cusip")["n_holders"].max()
    wanted = sorted(set(widely_held[widely_held >= MIN_HOLDERS].index) & set(labels["cusip"]))

    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["cusip"])
    todo = [c for c in wanted if c not in set(done["cusip"])]
    print(f"{len(wanted)} CUSIPs to map, {len(todo)} left")

    rows = done.to_dict("records")
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        # CUSIPs that start with a letter are CINS codes (companies registered
        # outside the US, e.g. Accenture G1151C101): a different ID type.
        id_type = lambda c: "ID_CINS" if c[0].isalpha() else "ID_CUSIP"
        # Ask for the US listing first; OpenFIGI otherwise may return only
        # foreign listings (Electronic Arts came back as its Mexican "EA*").
        us = post([{"idType": id_type(c), "idValue": c, "exchCode": "US"} for c in batch])
        time.sleep(PAUSE)
        missing = [c for c, a in zip(batch, us) if not a.get("data")]
        anywhere = dict(zip(missing, post([{"idType": id_type(c), "idValue": c} for c in missing]))) if missing else {}
        if missing:
            time.sleep(PAUSE)
        for cusip, ans in zip(batch, us):
            if not ans.get("data"):
                ans = anywhere.get(cusip, {})
            m = best_match(ans.get("data", []))
            listed_us = bool(m) and m.get("exchCode") == "US"
            rows.append({
                "cusip": cusip,
                "listed_us": listed_us,
                # a foreign listing's ticker (e.g. "TWTRUSD") is not a US ticker
                "ticker": m.get("ticker") if listed_us else None,
                "name": m.get("name") if m else None,
                "security_type": m.get("securityType") if m else None,
                "figi": m.get("compositeFIGI") if m else None,
            })
        if (i // BATCH) % 20 == 0 or i + BATCH >= len(todo):
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")
        time.sleep(PAUSE)
    pd.DataFrame(rows).to_parquet(OUT, index=False)

    sp_dir = ROOT / "data" / "raw" / "sp500"
    sp_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SP500_URL, sp_dir / "sp500_ticker_start_end.csv")

    m = pd.DataFrame(rows)
    print(f"\nmapped {m['ticker'].notna().sum()}/{len(m)} CUSIPs to a ticker")
    print(m["security_type"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
