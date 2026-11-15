"""Step 2b: turn quarterly holdings into BUY / HOLD / SELL labels.

For each fund, each stock and each pair of back-to-back quarters (t-1, t) in
which the fund filed BOTH reports, compare the shares held:

  BUY   shares went up by more than TOL (new positions included)
  SELL  shares went down by more than TOL (full exits included)
  HOLD  otherwise

Stock splits: a 4-for-1 split makes a fund's share count jump 4x without any
trade. If the share count changed by a split-like ratio k AND the stock's
typical price (from build_holdings.py) moved by about 1/k, we treat it as a
split: the previous quarter's shares are multiplied by k before comparing.

Outputs:
  data/processed/labels.parquet
  experiments/labels_summary.txt

Usage:
  .venv/bin/python src/build_labels.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
PROC = SMP.PROC
TOL = 0.01  # changes smaller than 1% count as HOLD (rounding, tiny rebalances)
FIRST_PERIOD = pd.Timestamp("2013-06-30")  # first quarter in the SEC data sets
SPLIT_RATIOS = [1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50]
CANDIDATE_KS = np.array(SPLIT_RATIOS + [1 / k for k in SPLIT_RATIOS])  # splits and reverse splits


def quarter_number(d: pd.Series) -> pd.Series:
    return d.dt.year * 4 + (d.dt.month - 1) // 3


WHOLE_KS = np.array([k for k in CANDIDATE_KS if k >= 2 or k <= 0.5])  # no 3-for-2


def nearest_k(x: pd.Series) -> pd.Series:
    """The candidate split ratio closest to x (compared on a log scale)."""
    logs = np.log(CANDIDATE_KS)
    lx = np.log(x.clip(lower=1e-6).to_numpy())[:, None]
    return pd.Series(CANDIDATE_KS[np.abs(lx - logs).argmin(axis=1)], index=x.index)


def split_factors(both_rows: pd.DataFrame) -> pd.Series:
    """Split factor per (cusip, q) where a split is detected.

    Two independent signals must agree on the SAME split ratio k:
      1. shares: the typical (median) share ratio across our funds that held
         the stock in both quarters points to k, and
      2. price: the typical price per share (value / shares, across all 13F
         filers) moved by about 1/k.
    Lessons from testing:
      - Funds that rebalance to a constant dollar amount trim shares when the
        price rises: that fakes ONE signal but not both pointing at a split k.
      - Real splits are rarely exact in our data, because funds also trade in
        the split quarter (NVIDIA 2024: typical ratio 9.2, not 10).
      - Prices move a lot in a quarter (Tesla doubled in its split quarter),
        so the price window must be wide.
    Small 3-for-2 splits (k = 1.5 or 2/3) look like ordinary rebalancing, so
    they additionally need 70% of holders at exactly the same ratio.
    """
    g = both_rows.groupby(["cusip", "q"])
    n = g.size()
    r_med = g["r"].median()
    q_med = g["q_ratio"].median()
    # Pick the ratio that fits BOTH signals best (rounding each separately
    # fails: Google 2022's price signal alone rounds to 25, its shares to 20).
    lr = np.log(r_med.clip(lower=1e-6).to_numpy())[:, None]
    lq = np.log(q_med.clip(lower=1e-6).fillna(1e-6).to_numpy())[:, None]
    logs = np.log(CANDIDATE_KS)[None, :]
    score = np.abs(lr - logs) + np.abs(lq + logs)
    k = pd.Series(CANDIDATE_KS[score.argmin(axis=1)], index=r_med.index)
    agree = q_med.gt(0)
    big = (k >= 2) | (k <= 0.5)
    loose = (np.abs(np.log(r_med / k)) < np.log(1.5)) & (np.abs(np.log(q_med * k)) < np.log(2.5))
    tight = (np.abs(np.log(r_med / k)) < np.log(1.05)) & (np.abs(np.log(q_med * k)) < np.log(1.6))
    exact = (
        (np.abs(both_rows["r"] / both_rows.set_index(["cusip", "q"]).index.map(k).to_numpy() - 1) < 0.01)
        .groupby([both_rows["cusip"], both_rows["q"]])
        .mean()
    )
    is_split = agree & (
        (big & (n >= 2) & loose)
        | (big & (n == 1) & tight)
        | (~big & (n >= 3) & tight & (exact >= 0.7))
    )
    return k[is_split]


def yahoo(ticker: str) -> str:
    return ticker.replace("/", "-").replace(" ", "-")


def real_tickered_cusips() -> list[str]:
    """CUSIPs whose split history we take from Yahoo (US ticker with price data)."""
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    m = m[m["listed_us"] & m["ticker"].notna()]
    have = set(pd.read_parquet(PROC / "prices_daily.parquet", columns=["ticker"])["ticker"].unique())
    return m.loc[m["ticker"].map(yahoo).isin(have), "cusip"].tolist()


def real_split_factors() -> pd.Series:
    """(cusip, q) -> product of Yahoo split ratios dated in quarter q."""
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    m = m[m["listed_us"] & m["ticker"].notna()].assign(yt=lambda d: d["ticker"].map(yahoo))
    S = pd.read_parquet(PROC / "splits.parquet")
    S["date"] = pd.to_datetime(S["date"])
    S["q"] = quarter_number(S["date"])
    S = S.merge(m[["cusip", "yt"]], left_on="ticker", right_on="yt")
    return S.groupby(["cusip", "q"])["ratio"].prod()


def main() -> None:
    H = pd.read_parquet(PROC / "holdings.parquet")
    P = pd.read_parquet(PROC / "implied_prices.parquet")
    funds = pd.read_csv(SMP.FUND_LIST, dtype=str)

    H = H[H["period"] >= FIRST_PERIOD].copy()
    # Only hedge funds ("yes") are in the sample. Controls such as Berkshire
    # Hathaway are kept only when INCLUDE_CONTROL=1 (a separate sanity run).
    keep = ["yes", "control"] if os.environ.get("INCLUDE_CONTROL") == "1" else ["yes"]
    H = H[H["fund"].isin(funds.loc[funds["include"].isin(keep), "fund"])]

    # A fund with two SEC IDs (old + new) may file under both in an overlap
    # quarter: keep the newer ID's report for that quarter.
    order = {cik: i for row in funds.itertuples() for i, cik in enumerate(row.ciks.split(";"))}
    H["id_order"] = H["cik"].map(order).fillna(0)
    newest = H.groupby(["fund", "period"])["id_order"].transform("max")
    H = H[H["id_order"] == newest].drop(columns="id_order")

    H["q"] = quarter_number(H["period"])
    P["q"] = quarter_number(P["period"])
    # Odd period-end dates can put two reports in one calendar quarter: keep the latest.
    H = H.sort_values("period").drop_duplicates(["fund", "q", "cusip"], keep="last")
    P = P.sort_values("n_holders").drop_duplicates(["cusip", "q"], keep="last")
    price = P.set_index(["cusip", "q"])["implied_price"]

    # Quarters each fund filed; a pair (t-1, t) is usable only if both were filed.
    filed = H[["fund", "q"]].drop_duplicates()
    pairs = filed.merge(filed.assign(q=filed["q"] + 1), on=["fund", "q"])  # q = t, both t and t-1 filed
    gaps = len(filed) - len(pairs) - filed["fund"].nunique()

    now = H[["fund", "cik", "q", "period", "cusip", "issuer", "shares"]]
    before = H[["fund", "q", "cusip", "issuer", "shares"]].assign(q=H["q"] + 1)
    before = before.rename(columns={"shares": "shares_prev", "issuer": "issuer_prev"})
    L = now.merge(before, on=["fund", "q", "cusip"], how="outer")
    L = L.merge(pairs, on=["fund", "q"])  # drop rows whose quarter pair isn't complete
    L["shares"] = L["shares"].fillna(0.0)
    L["shares_prev"] = L["shares_prev"].fillna(0.0)
    L["issuer"] = L["issuer"].fillna(L["issuer_prev"])
    L["period"] = L.groupby(["fund", "q"])["period"].transform("first")
    L["cik"] = L.groupby(["fund", "q"])["cik"].transform("first")

    # Split adjustment, only where the fund held the stock in both quarters.
    both = (L["shares"] > 0) & (L["shares_prev"] > 0)
    r = np.where(both, L["shares"] / L["shares_prev"].where(both, 1), 1.0)
    p_now = price.reindex(list(zip(L["cusip"], L["q"]))).to_numpy()
    p_before = price.reindex(list(zip(L["cusip"], L["q"] - 1))).to_numpy()
    L["r"] = r
    L["q_ratio"] = np.where(both & (p_before > 0), p_now / p_before, np.nan)
    # Split factor per stock-quarter, shared by every fund that held it.
    # Stocks with a US ticker: REAL split history from Yahoo (product of every
    # split dated after the previous quarter-end, up to this quarter-end).
    # Stocks without one (mostly delisted): the holdings-based heuristic.
    real = real_split_factors()
    tickered = set(real_tickered_cusips())
    heuristic = split_factors(L.loc[both & ~L["cusip"].isin(tickered), ["cusip", "q", "r", "q_ratio"]])
    k_map = pd.concat([real, heuristic])
    L["split_factor"] = 1.0
    hit = both & pd.MultiIndex.from_arrays([L["cusip"], L["q"]]).isin(k_map.index)
    L.loc[hit, "split_factor"] = k_map.reindex(
        pd.MultiIndex.from_arrays([L.loc[hit, "cusip"], L.loc[hit, "q"]])
    ).to_numpy()
    L["split_source"] = np.where(
        L["cusip"].isin(tickered), "yahoo", np.where(L["split_factor"] != 1, "heuristic", "none")
    )
    L = L.drop(columns=["r", "q_ratio"])

    # Universe: keep quarter-t labels only for stocks in the top 500 at the
    # end of quarter t-1 (known before any of quarter t's trades).
    U = pd.read_parquet(PROC / "universe.parquet", columns=["period", "cusip"])
    U["q"] = quarter_number(U["period"]) + 1  # membership known at t-1 applies to t
    before_universe = len(L)
    L = L.merge(U[["cusip", "q"]], on=["cusip", "q"])
    kept_share = len(L) / before_universe
    L["shares_prev_adj"] = L["shares_prev"] * L["split_factor"]

    # The label.
    prev = L["shares_prev_adj"]
    L["change_pct"] = np.where(prev > 0, L["shares"] / prev.where(prev > 0, 1) - 1, np.inf)
    L["label"] = np.select(
        [L["change_pct"] > TOL, L["change_pct"] < -TOL],
        ["BUY", "SELL"],
        default="HOLD",
    )
    L["kind"] = np.select(
        [prev == 0, L["shares"] == 0],
        ["new position", "full exit"],
        default="change",
    )

    cols = ["fund", "cik", "period", "cusip", "issuer", "shares_prev", "split_factor", "split_source",
            "shares_prev_adj", "shares", "change_pct", "label", "kind"]
    L = L[cols].sort_values(["fund", "period", "cusip"]).reset_index(drop=True)
    L.to_parquet(PROC / "labels.parquet", index=False)

    # Summary for the decisions log and for checking by eye.
    lines = []
    lines.append(f"labels: {len(L):,} rows, {L['fund'].nunique()} funds, "
                 f"{L['period'].min().date()} to {L['period'].max().date()}")
    lines.append(f"quarter pairs skipped because a fund missed a filing: {gaps}")
    lines.append(f"rows kept by the top-500 universe filter: {kept_share:.1%}")
    lines.append(f"stocks per quarter (mean): {L.groupby('period')['cusip'].nunique().mean():.0f}")
    lines.append("\nlabel share:")
    lines.append(L["label"].value_counts(normalize=True).round(3).to_string())
    lines.append("\nkind share:")
    lines.append(L["kind"].value_counts(normalize=True).round(3).to_string())
    lines.append(f"\nrows treated as stock splits: {(L['split_factor'] != 1).sum():,}")
    lines.append("most common split factors:")
    lines.append(L.loc[L["split_factor"] != 1, "split_factor"].round(3).value_counts().head(8).to_string())
    lines.append("\nlabels per year:")
    lines.append(L.groupby(L["period"].dt.year)["label"].value_counts().unstack().to_string())
    text = "\n".join(lines)
    SMP.EXP.mkdir(exist_ok=True)
    (SMP.EXP / "labels_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
