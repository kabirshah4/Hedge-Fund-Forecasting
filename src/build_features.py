"""Step 5a: features for predicting each fund's next-quarter action.

One row per (fund, stock, quarter t) where the fund HELD the stock at the end
of quarter t-1 (like Mimicking Finance: "positions with continuous presence").
Target: label at t (BUY / HOLD / SELL, from build_labels.py).

Every feature uses only information from the end of quarter t-1 or earlier,
so nothing from quarter t leaks into the prediction.

Feature groups
  fund history   the fund's last 3 actions on this stock, holding age,
                 position weight and rank in the portfolio
  fund overall   number of stocks held, share of positions it changed last
                 quarter, share it bought
  stock          return last quarter, 4-quarter momentum, size rank
  peers          how many of our other funds held it, and what share of them
                 bought / sold it last quarter (herding)
  macro          average policy uncertainty (EPU index) over quarter t-1

Output:
  data/processed/features.parquet

Usage:
  .venv/bin/python src/build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
PROC = SMP.PROC
ACTION_CODE = {"SELL": 0, "HOLD": 1, "BUY": 2}


def qn(d: pd.Series) -> pd.Series:
    return d.dt.year * 4 + (d.dt.month - 1) // 3


def stock_returns() -> pd.DataFrame:
    """Quarterly stock returns per (cusip, q). Yahoo adjusted prices where a
    ticker exists; otherwise the 13F typical price (no split correction)."""
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    m = m[m["listed_us"].astype(bool) & m["ticker"].notna()].copy()
    m["yt"] = m["ticker"].str.replace("/", "-").str.replace(" ", "-")
    px = pd.read_parquet(PROC / "prices_daily.parquet", columns=["date", "ticker", "adj_close"])
    px["date"] = pd.to_datetime(px["date"])
    px["q"] = qn(px["date"])
    qend = px.sort_values("date").groupby(["ticker", "q"])["adj_close"].last().reset_index()
    qend["ret"] = qend.groupby("ticker")["adj_close"].pct_change()
    yahoo = qend.merge(m[["cusip", "yt"]], left_on="ticker", right_on="yt")[["cusip", "q", "ret"]]

    ip = pd.read_parquet(PROC / "implied_prices.parquet", columns=["cusip", "period", "implied_price"])
    ip = ip[ip["period"].dt.is_quarter_end]
    ip["q"] = qn(ip["period"])
    ip = ip.sort_values(["cusip", "q"])
    ip["ret13f"] = ip.groupby("cusip")["implied_price"].pct_change()
    ip.loc[ip.groupby("cusip")["q"].diff() != 1, "ret13f"] = np.nan  # only back-to-back quarters
    r = ip[["cusip", "q", "ret13f"]].merge(yahoo, on=["cusip", "q"], how="left")
    r["ret"] = r["ret"].fillna(r["ret13f"]).clip(-0.95, 5)
    r = r.sort_values(["cusip", "q"])
    r["mom4"] = (
        r.groupby("cusip")["ret"].transform(lambda s: (1 + s).rolling(4, min_periods=3).apply(np.prod, raw=True) - 1)
    )
    return r[["cusip", "q", "ret", "mom4"]]


def main() -> None:
    L = pd.read_parquet(PROC / "labels.parquet")
    L["q"] = qn(L["period"])
    L["y"] = L["label"].map(ACTION_CODE)

    # Rows to predict: positions held at the end of t-1.
    T = L[L["shares_prev"] > 0][["fund", "cusip", "q", "period", "y", "label"]].copy()

    # --- fund history on this stock: actions at t-1, t-2, t-3 (all labels,
    #     including new positions, which count as BUY) ---
    act = L[["fund", "cusip", "q", "y"]]
    for lag in (1, 2, 3):
        a = act.assign(q=act["q"] + lag).rename(columns={"y": f"act_lag{lag}"})
        T = T.merge(a, on=["fund", "cusip", "q"], how="left")

    # --- holding context from the full holdings (all stocks, not just top 500) ---
    H = pd.read_parquet(PROC / "holdings.parquet", columns=["fund", "period", "cusip", "value_usd"])
    H = H[H["period"].dt.is_quarter_end].copy()
    H["q"] = qn(H["period"])
    H = H.groupby(["fund", "q", "cusip"], as_index=False)["value_usd"].sum()
    H["weight"] = H["value_usd"] / H.groupby(["fund", "q"])["value_usd"].transform("sum")
    H["rank_in_fund"] = H.groupby(["fund", "q"])["value_usd"].rank(ascending=False)
    H["n_holdings"] = H.groupby(["fund", "q"])["cusip"].transform("size")
    # consecutive quarters held, up to and including q
    H = H.sort_values(["fund", "cusip", "q"])
    new_run = H.groupby(["fund", "cusip"])["q"].diff().ne(1)
    H["holding_age"] = H.groupby([H["fund"], H["cusip"], new_run.cumsum()]).cumcount() + 1
    prev = H.assign(q=H["q"] + 1)[["fund", "cusip", "q", "weight", "rank_in_fund", "n_holdings", "holding_age"]]
    T = T.merge(prev, on=["fund", "cusip", "q"], how="left")
    T["log_weight"] = np.log(T["weight"].clip(lower=1e-7))

    # --- fund-wide activity last quarter (from all its labels) ---
    fa = L.groupby(["fund", "q"]).agg(
        fund_changed=("label", lambda s: (s != "HOLD").mean()),
        fund_buy_share=("label", lambda s: (s == "BUY").mean()),
    ).reset_index()
    T = T.merge(fa.assign(q=fa["q"] + 1), on=["fund", "q"], how="left")

    # --- the fund's own long-run tendency, from quarters before t only ---
    fh = L.sort_values("q").groupby(["fund", "q"])["y"].agg(["sum", "size"]).groupby(level=0).cumsum()
    fh = fh.reset_index()
    fh["fund_hist_mean_action"] = fh["sum"] / fh["size"]
    T = T.merge(fh.assign(q=fh["q"] + 1)[["fund", "q", "fund_hist_mean_action"]], on=["fund", "q"], how="left")

    # --- stock returns up to t-1 ---
    R = stock_returns()
    T = T.merge(R.assign(q=R["q"] + 1).rename(columns={"ret": "ret_lag1", "mom4": "mom4_lag1"}),
                on=["cusip", "q"], how="left")

    # --- stock size rank at t-1 (universe) ---
    U = pd.read_parquet(PROC / "universe.parquet", columns=["period", "cusip", "rank"])
    U["q"] = qn(U["period"]) + 1
    T = T.merge(U[["cusip", "q", "rank"]].rename(columns={"rank": "size_rank"}), on=["cusip", "q"], how="left")

    # --- peers: what our OTHER funds did with this stock at t-1 ---
    pe = L.groupby(["cusip", "q"]).agg(
        peer_n=("fund", "size"),
        peer_buy=("label", lambda s: (s == "BUY").sum()),
        peer_sell=("label", lambda s: (s == "SELL").sum()),
    ).reset_index()
    pe = pe.assign(q=pe["q"] + 1)
    T = T.merge(pe, on=["cusip", "q"], how="left")
    own = T["act_lag1"]
    T["peer_n"] = T["peer_n"].fillna(0) - own.notna()
    T["peer_buy_share"] = (T["peer_buy"].fillna(0) - (own == 2)) / T["peer_n"].where(T["peer_n"] > 0)
    T["peer_sell_share"] = (T["peer_sell"].fillna(0) - (own == 0)) / T["peer_n"].where(T["peer_n"] > 0)
    T = T.drop(columns=["peer_buy", "peer_sell"])

    # --- policy uncertainty, averaged over quarter t-1 ---
    e = pd.read_excel(ROOT / "data" / "raw" / "epu" / "US_Policy_Uncertainty_Data.xlsx")
    e["Year"] = pd.to_numeric(e["Year"], errors="coerce")
    e = e.dropna(subset=["Year", "Month"])
    e = e[e["Year"] >= 2010]
    e["q"] = e["Year"].astype(int) * 4 + (e["Month"].astype(int) - 1) // 3
    epu = e.groupby("q")["News_Based_Policy_Uncert_Index"].mean().rename("epu_lag1").reset_index()
    T = T.merge(epu.assign(q=epu["q"] + 1), on="q", how="left")

    T["fund_id"] = T["fund"].astype("category").cat.codes
    T.to_parquet(PROC / "features.parquet", index=False)
    print(f"features: {len(T):,} rows x {T.shape[1]} columns, {T['fund'].nunique()} funds")
    print("missing share per feature:")
    print(T.isna().mean().round(3)[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
