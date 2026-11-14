"""Step 6a: keep the news headlines about stocks in our universe.

Source: FNSPID (Dong et al. 2024, arXiv 2402.06698), file All_external.csv:
Benzinga headlines with a date and the stock ticker they are about.

Keeps: headlines whose ticker belongs to a stock that is ever in our top-500
universe, dated 2013-01-01 or later. Drops exact duplicate headlines for the
same stock on the same day.

Output:
  data/processed/news_headlines.parquet  date (UTC), quarter, ticker, cusip, headline

Usage:
  .venv/bin/python src/prepare_news.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
NEWS = ROOT / "data" / "raw" / "news"
# Benzinga headlines (stock-tagged rows run to mid-2020) and Nasdaq.com articles
# (stock-tagged, mid-2020 to 2023). Only the date, headline and ticker are read.
FILES = ["All_external.csv", "nasdaq_exteral_data.csv"]


def main() -> None:
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    U = pd.read_parquet(PROC / "universe.parquet", columns=["cusip"]).drop_duplicates()
    m = m[m["listed_us"].astype(bool) & m["ticker"].notna() & m["cusip"].isin(U["cusip"])]
    # FNSPID writes share classes like "BRK.B"; OpenFIGI uses "BRK/B".
    m = m.assign(t=m["ticker"].str.replace("/", ".").str.upper())
    to_cusip = m.drop_duplicates("t").set_index("t")["cusip"]

    parts = []
    for name in FILES:
        path = NEWS / name
        if not path.exists():
            print(f"missing {name}, skipped")
            continue
        reader = pd.read_csv(
            path, usecols=["Date", "Article_title", "Stock_symbol"], dtype=str,
            chunksize=500_000, on_bad_lines="skip", engine="c",
        )
        for i, chunk in enumerate(reader):
            chunk = chunk.dropna()
            chunk["Stock_symbol"] = chunk["Stock_symbol"].str.upper().str.strip()
            chunk = chunk[chunk["Stock_symbol"].isin(to_cusip.index)]
            chunk["source"] = "benzinga" if name.startswith("All") else "nasdaq"
            parts.append(chunk)
        print(f"  {name}: kept {sum(len(p) for p in parts if p['source'].iloc[0:1].eq('benzinga' if name.startswith('All') else 'nasdaq').all()):,} so far")
    N = pd.concat(parts, ignore_index=True)
    N["date"] = pd.to_datetime(N["Date"].str.replace(" UTC", ""), errors="coerce", utc=True)
    N = N.dropna(subset=["date"])
    N = N[N["date"] >= pd.Timestamp("2013-01-01", tz="UTC")]
    N = N.rename(columns={"Stock_symbol": "ticker", "Article_title": "headline"})
    N["cusip"] = N["ticker"].map(to_cusip)
    N["headline"] = N["headline"].str.strip()
    N = N.drop_duplicates(["cusip", "headline"])
    N["quarter"] = N["date"].dt.year * 4 + (N["date"].dt.month - 1) // 3
    N[["date", "quarter", "ticker", "cusip", "headline", "source"]].to_parquet(PROC / "news_headlines.parquet", index=False)

    print(f"\nheadlines kept: {len(N):,} for {N['cusip'].nunique()} stocks")
    print(f"date range: {N['date'].min().date()} to {N['date'].max().date()}")
    print("headlines per year and source:")
    print(N.groupby([N["date"].dt.year, "source"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
