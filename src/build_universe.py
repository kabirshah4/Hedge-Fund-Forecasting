"""Step 4c: the stock universe: each quarter's 500 largest stocks.

"Largest" = total institutional holdings value across ALL 13F filers that
quarter, measured robustly (see build_holdings.py): each filer's shares x
the stock's typical price, leaving out filers whose own price is more than 5x
off (a unit mistake in one of their columns). Summing reported values
directly fails: a few filers reported dollars where the SEC expected
thousands, putting Google at "$263 trillion"; others typed dollars into the
shares column, putting Berkshire's Class A at "$2.7 trillion".

What counts: equities only.
  - Bonds are dropped by their ID: a CUSIP's characters 7-8 are digits for
    stocks and contain letters for debt (e.g. 43713WAB3).
  - Stocks KNOWN to be ETFs/funds (e.g. SPY), ADRs, closed-end funds or MLPs
    are dropped.
  - Stocks with NO known type are KEPT: many are companies later acquired or
    delisted (Google's pre-2015 share, EMC, Lorillard), and dropping them
    would bring back survivorship bias.
  - Every stock must be held by at least MIN_HOLDERS filers that quarter,
    which removes mutual-fund shares and odd IDs held by a single filer.

Because it is built from SEC data alone, companies that were later renamed,
acquired or delisted are still in it: no survivorship bias.

The labels for quarter t use the universe from quarter t-1 (see build_labels.py),
so membership never depends on information from later.

Output:
  data/processed/universe.parquet  period, cusip, rank, inst_value_usd, ticker, name, security_type

Usage:
  .venv/bin/python src/build_universe.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
SIZE = 500
EXCLUDED_TYPES = {"ETP", "Closed-End Fund", "ADR", "MLP", "Mutual Fund", "Open-End Fund",
                  "Fund of Funds", "Unit", "Right", "Warrant", "Preferred"}
# Real large companies are held by hundreds or thousands of 13F filers (95% of
# known stocks in the top 500 have 561+). Mutual-fund shares and odd IDs that
# slip through are typically held by ONE filer reporting a big value.
MIN_HOLDERS = 100


def main() -> None:
    P = pd.read_parquet(PROC / "implied_prices.parquet")
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    P = P[P["period"] >= "2013-03-31"]
    P = P.merge(m[["cusip", "ticker", "name", "security_type"]], on="cusip", how="left")

    # Quarter-end periods only (a few filers report odd dates).
    P = P[P["period"].dt.is_quarter_end]

    is_equity_id = P["cusip"].str[6:8].str.isdigit()
    widely_held = P["n_holders"] >= MIN_HOLDERS
    U = P[is_equity_id & widely_held & ~P["security_type"].isin(EXCLUDED_TYPES)].copy()
    U["rank"] = U.groupby("period")["inst_value_usd"].rank(ascending=False, method="first")
    U = U[U["rank"] <= SIZE].sort_values(["period", "rank"])
    cols = ["period", "cusip", "rank", "inst_value_usd", "ticker", "name", "security_type"]
    U[cols].to_parquet(PROC / "universe.parquet", index=False)

    per_q = U.groupby("period").size()
    print(f"universe: {U['period'].nunique()} quarters, {per_q.min()}-{per_q.max()} stocks each, "
          f"{U['cusip'].nunique()} distinct stocks over time")
    print(f"universe stocks with no known security type (kept): {U['security_type'].isna().mean():.1%}")
    print(f"stocks in the universe without a US ticker (e.g. delisted): {U['ticker'].isna().mean():.1%}")
    last = U[U["period"] == U["period"].max()]
    print("\nlargest 10 at the latest quarter:")
    print(last.head(10)[["rank", "ticker", "name", "inst_value_usd"]].to_string(index=False))
    turnover = (
        U.groupby("period")["cusip"].apply(set).pipe(
            lambda s: pd.Series([len(a - b) for a, b in zip(s.iloc[1:], s.iloc[:-1])], index=s.index[1:])
        )
    )
    print(f"\nstocks entering the top {SIZE} per quarter: median {turnover.median():.0f}")


if __name__ == "__main__":
    main()
