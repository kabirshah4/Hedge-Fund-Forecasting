"""Step 19: data-quality numbers requested by the second blind review (W8, M17).

  split_like_pct      prediction cases whose share ratio is within 0.2% of a
                      split ratio (2, 3, 4, 5, 10, 20, 3:2 and inverses) with no
                      split applied: an UPPER bound on missed splits, since many
                      are real trades that exactly double or halve a position
  cusip_change_pct    test-period exits matched, in the same fund and quarter,
                      to a new position in a different CUSIP with the same issuer
                      name (CUSIP changes masquerading as SELLs)
  universe filters    of the 500 largest stocks by value before filtering, the
                      average number removed per quarter by each rule
  filing lag          share of hedge fund test-period reports filed within 31
                      and 45 days of quarter-end (first filing)

Output: experiments/data_quality.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

import build_universe as BU

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def main():
    out = {}
    L = pd.read_parquet(PROC / "labels.parquet")
    held = L[L["shares_prev"] > 0]
    ratio = held["shares"] / held["shares_prev_adj"]
    near = np.zeros(len(held), bool)
    for k in [2, 3, 4, 5, 10, 20, 1.5, 1 / 1.5, 0.5, 1 / 3, 0.25, 0.2, 0.1, 0.05]:
        near |= np.isclose(ratio, k, rtol=0.002)
    out["split_like_pct"] = 100 * (near & (held["split_factor"] == 1.0)).mean()
    key = lambda s: s.fillna("").str.upper().str.replace(r"[^A-Z ]", "", regex=True).str.split().str[:2].str.join(" ")
    ex = L[L["kind"] == "full exit"][["fund", "period", "cusip", "issuer"]].assign(k=lambda d: key(d["issuer"]))
    nw = L[L["kind"] == "new position"][["fund", "period", "cusip", "issuer"]].assign(k=lambda d: key(d["issuer"]))
    mm = ex.merge(nw, on=["fund", "period", "k"], suffixes=("", "_new"))
    mm = mm[(mm["cusip"] != mm["cusip_new"]) & (mm["period"] > "2021-12-31")].drop_duplicates(["fund", "period", "cusip"])
    test_sells = ((L["period"] > "2021-12-31") & (L["label"] == "SELL") & (L["shares_prev"] > 0)).sum()
    out["cusip_change_n"] = len(mm)
    out["cusip_change_pct"] = 100 * len(mm) / test_sells
    P = pd.read_parquet(PROC / "implied_prices.parquet")
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    P = P[(P["period"] >= "2013-03-31") & P["period"].dt.is_quarter_end].merge(m[["cusip", "security_type"]], on="cusip", how="left")
    P["r0"] = P.groupby("period")["inst_value_usd"].rank(ascending=False, method="first")
    top = P[P["r0"] <= BU.SIZE]
    n = top["period"].nunique()
    out["univ_removed_type"] = top["security_type"].isin(BU.EXCLUDED_TYPES).sum() / n
    out["univ_removed_holders"] = (top["n_holders"] < BU.MIN_HOLDERS).sum() / n
    out["univ_removed_nonequity"] = (~top["cusip"].str[6:8].str.isdigit()).sum() / n
    H = pd.read_parquet(PROC / "holdings.parquet", columns=["fund", "period", "filed"])
    H = H[H["period"].dt.is_quarter_end]
    H["filed"] = pd.to_datetime(H["filed"])
    f = H.groupby(["fund", "period"])["filed"].min().reset_index()
    f = f[f["period"] >= "2021-12-31"]
    lag = (f["filed"] - f["period"]).dt.days
    out["filed_within31_pct"] = 100 * (lag <= 31).mean()
    out["filed_within45_pct"] = 100 * (lag <= 45).mean()
    out["filing_lag_median"] = lag.median()
    pd.DataFrame([out]).to_csv(ROOT / "experiments" / "data_quality.csv", index=False)
    for k, v in out.items():
        print(f"{k:24s} {v:.2f}")


if __name__ == "__main__":
    main()
