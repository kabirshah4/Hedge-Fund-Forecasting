"""Step 21: extend the price panel from 30 June to 31 October 2026 (review M3/Q4).

Re-downloading the whole history would subtly change every earlier result,
because Yahoo re-scales adjusted prices for later dividends. Instead we fetch
June-October 2026 only and splice it on: for each ticker, the new adjusted
series is rescaled so that it equals the existing value on the last common
trading day, and only dates after that day are appended. Labels and features
(which use prices up to quarter t-1 <= 2026Q1) are unaffected; RQ3 gains the
2026Q2 formation quarter. The original file is kept as prices_daily_to_2026-06.parquet.

Usage: .venv/bin/python src/extend_prices.py
"""

import shutil
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
START, END = "2026-06-01", "2026-11-01"


def main() -> None:
    src = PROC / "prices_daily.parquet"
    keep = PROC / "prices_daily_to_2026-06.parquet"
    if not keep.exists():
        shutil.copy(src, keep)
    P = pd.read_parquet(keep)
    P["date"] = pd.to_datetime(P["date"])
    tickers = sorted(P["ticker"].unique())
    new = []
    for i in range(0, len(tickers), 100):
        df = yf.download(tickers[i:i + 100], start=START, end=END, actions=False, auto_adjust=False,
                         group_by="column", threads=True, progress=False)
        if df.empty:
            continue
        long = df.stack(level=1, future_stack=True).reset_index()
        long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
        long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "ticker"}).dropna(subset=["close"])
        new.append(long[["date", "ticker", "close", "adj_close", "volume"]])
        print(f"  {min(i + 100, len(tickers))}/{len(tickers)}", flush=True)
    N = pd.concat(new, ignore_index=True)
    N["date"] = pd.to_datetime(N["date"])
    last = P.groupby("ticker")["date"].max().rename("last")
    out = []
    for t, g in N.groupby("ticker"):
        if t not in last.index:
            continue
        d0 = last[t]
        old = P[(P["ticker"] == t) & (P["date"] == d0)]["adj_close"]
        ref = g[g["date"] == d0]["adj_close"]
        if old.empty or ref.empty or ref.iloc[0] == 0:
            continue  # no common day to splice on
        g = g[g["date"] > d0].copy()
        g["adj_close"] = g["adj_close"] * old.iloc[0] / ref.iloc[0]
        out.append(g)
    E = pd.concat(out, ignore_index=True)
    pd.concat([P, E], ignore_index=True).sort_values(["ticker", "date"]).to_parquet(src, index=False)
    print(f"appended {len(E):,} rows for {E['ticker'].nunique():,} tickers; panel now ends {E['date'].max().date()}")


if __name__ == "__main__":
    main()
