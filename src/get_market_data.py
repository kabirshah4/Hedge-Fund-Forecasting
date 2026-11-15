"""Step 4b: download daily prices and stock-split history from Yahoo Finance.

Uses the tickers found by map_tickers.py (US listings only). For each ticker:
  - daily close, split- and dividend-adjusted close, and volume
  - every stock split (date and ratio), used to fix the labels exactly

Notes:
  - Yahoo has no data for many delisted companies. Their quarterly returns can
    come from the 13F typical prices instead (implied_prices.parquet).
  - OpenFIGI writes share classes as "BRK/B"; Yahoo wants "BRK-B".

Outputs:
  data/processed/prices_daily.parquet  date, ticker, close, adj_close, volume
  data/processed/splits.parquet        ticker, date, ratio

Usage:
  .venv/bin/python src/get_market_data.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
START, END = "2012-10-01", "2026-07-01"
CHUNK = 100


def yahoo(ticker: str) -> str:
    return ticker.replace("/", "-").replace(" ", "-")


def main() -> None:
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    tickers = sorted({yahoo(t) for t in m.loc[m["listed_us"] & m["ticker"].notna(), "ticker"]})
    print(f"{len(tickers)} tickers")

    prices, splits = [], []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        df = yf.download(
            chunk, start=START, end=END, actions=True, auto_adjust=False,
            group_by="column", threads=True, progress=False,
        )
        if df.empty:
            continue
        long = df.stack(level=1, future_stack=True).reset_index()
        long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
        long = long.rename(columns={"level_1": "ticker", "price": "date"})
        long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "ticker"})
        long = long.dropna(subset=["close"])
        prices.append(long[["date", "ticker", "close", "adj_close", "volume"]])
        if "stock_splits" in long:
            s = long.loc[long["stock_splits"].fillna(0) > 0, ["ticker", "date", "stock_splits"]]
            splits.append(s.rename(columns={"stock_splits": "ratio"}))
        print(f"  {min(i + CHUNK, len(tickers))}/{len(tickers)}")

    P = pd.concat(prices, ignore_index=True)
    S = pd.concat(splits, ignore_index=True) if splits else pd.DataFrame(columns=["ticker", "date", "ratio"])
    P.to_parquet(PROC / "prices_daily.parquet", index=False)
    S.to_parquet(PROC / "splits.parquet", index=False)
    got = P["ticker"].nunique()
    print(f"\nprices: {len(P):,} rows for {got}/{len(tickers)} tickers; splits: {len(S)}")
    missing = sorted(set(tickers) - set(P["ticker"]))
    print(f"no Yahoo data for {len(missing)} tickers, e.g. {missing[:15]}")


if __name__ == "__main__":
    main()
