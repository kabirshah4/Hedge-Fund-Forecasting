"""Step 27: firm characteristics as extra features (fifth review, W1).

Builds, for every universe stock and prediction quarter t, characteristics that
were public at the end of quarter t-1 (the cutoff for all other stock features):

Fundamentals, from SEC XBRL "frames" (one request returns one concept for every
filer and period; https://data.sec.gov/api/xbrl/frames/...). A value is used
only if its period ended at least 120 days before the cutoff, so it was filed
(10-K deadline <= 90 days, 10-Q <= 45 days) -- no lookahead.
    book-to-market, earnings yield, ROA, gross profitability, operating margin,
    leverage, cash/assets, asset growth (1y), sales growth (1y)
Prices (data/processed/prices_daily.parquet), up to the cutoff:
    1-month return (reversal), 12-1 momentum, 60-day volatility, 252-day beta
    (vs. the equal-weighted panel), Amihud illiquidity, turnover

Stocks are matched to SEC CIKs by ticker (SEC company_tickers.json) and, for
companies no longer listed, by exact normalized name against the frames' entity
names, so delisted companies are covered as well as possible.

SEC requests need EDGAR_IDENTITY in the environment (see edgar.py); raw JSON is
cached in data/raw/frames/.

Output: data/processed/firm_chars.parquet (cusip, q, <features>)
Usage:  EDGAR_IDENTITY="..." .venv/bin/python src/firm_chars.py
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import edgar

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "frames"
PROC = ROOT / "data" / "processed"
LAG = pd.Timedelta(days=120)
STALE = pd.Timedelta(days=400)

INSTANT = {"Assets": "assets", "StockholdersEquity": "equity", "Liabilities": "liabilities",
           "CashAndCashEquivalentsAtCarryingValue": "cash"}
SHARES = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")]
ANNUAL = {"NetIncomeLoss": "ni", "GrossProfit": "gp", "OperatingIncomeLoss": "opinc",
          "Revenues": "rev1", "RevenueFromContractWithCustomerExcludingAssessedTax": "rev2", "SalesRevenueNet": "rev3"}
PRICE_FEATS = ["ret_1m", "mom_12_1", "vol_60d", "beta_252", "amihud", "turnover"]
FUND_FEATS = ["book_to_market", "earnings_yield", "roa", "gross_prof", "op_margin", "leverage", "cash_assets",
              "asset_growth", "sales_growth"]
FEATURES = FUND_FEATS + PRICE_FEATS


def norm(name) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    s = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|the|class [a-z]|cl [a-z]|holdings?|group|lp|llc|sa|nv)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch(session, tax, tag, unit, period) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{tax}_{tag}_{unit}_{period}.json"
    if not f.exists():
        try:  # edgar.get spaces requests (SEC fair-access limit) and retries
            text = edgar.get(session, f"https://data.sec.gov/api/xbrl/frames/{tax}/{tag}/{unit}/{period}.json").text
        except Exception:
            text = "{}"   # concept/period not published
        f.write_text(text)
    d = json.loads(f.read_text() or "{}")
    if not d.get("data"):
        return pd.DataFrame(columns=["cik", "end", "val", "name"])
    D = pd.DataFrame(d["data"])
    return pd.DataFrame({"cik": D["cik"].astype(int), "end": pd.to_datetime(D["end"]), "val": D["val"].astype(float),
                         "name": D["entityName"]})


def download():
    s = edgar.session()
    inst_periods = [f"CY{y}Q{k}I" for y in range(2010, 2026) for k in range(1, 5)]
    ann_periods = [f"CY{y}" for y in range(2009, 2025)]
    tabs, names = {}, []
    for tag, col in INSTANT.items():
        tabs[col] = pd.concat([fetch(s, "us-gaap", tag, "USD", p) for p in inst_periods])
        names.append(tabs[col][["cik", "name"]])
    sh = []
    for tax, tag in SHARES:
        sh.append(pd.concat([fetch(s, tax, tag, "shares", p) for p in inst_periods]))
    tabs["shares"] = pd.concat(sh).sort_values("end").drop_duplicates(["cik", "end"], keep="first")
    for tag, col in ANNUAL.items():
        tabs[col] = pd.concat([fetch(s, "us-gaap", tag, "USD", p) for p in ann_periods])
    r = edgar.get(s, "https://www.sec.gov/files/company_tickers.json")
    tick = pd.DataFrame(r.json()).T
    tick = pd.DataFrame({"ticker": tick["ticker"].str.upper(), "cik": tick["cik_str"].astype(int)})
    return tabs, pd.concat(names).drop_duplicates(), tick


def asof(left: pd.DataFrame, table: pd.DataFrame, key: str, name: str, shift=pd.Timedelta(0)) -> pd.DataFrame:
    """For each (cik, key date - shift), the latest value with end <= that date (and not older than STALE)."""
    L = left.assign(_d=left[key] - shift).sort_values("_d")
    R = table[["cik", "end", "val"]].dropna().copy()
    R["end"], R["cik"] = pd.to_datetime(R["end"]), R["cik"].astype(int)
    R = R.sort_values("end").rename(columns={"val": name})
    out = pd.merge_asof(L, R, left_on="_d", right_on="end", by="cik", direction="backward", tolerance=STALE)
    return out.drop(columns=["_d", "end"])


def price_features(cutoffs: pd.DatetimeIndex) -> pd.DataFrame:
    P = pd.read_parquet(PROC / "prices_daily.parquet")
    adj = P.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    close = P.pivot(index="date", columns="ticker", values="close").reindex(adj.index)
    vol = P.pivot(index="date", columns="ticker", values="volume").reindex(adj.index)
    ret = adj.pct_change(fill_method=None)
    mkt = ret.clip(-0.5, 0.5).mean(axis=1)
    dvol = close * vol
    out = []
    for c in cutoffs:
        i = adj.index.searchsorted(c, side="right") - 1
        if i < 260:
            continue
        a = adj.iloc[i]
        r60, r252, m252 = ret.iloc[i - 59:i + 1], ret.iloc[i - 251:i + 1], mkt.iloc[i - 251:i + 1]
        beta = r252.apply(lambda x: x.cov(m252) / m252.var() if x.notna().sum() > 150 else np.nan)
        am = (r60.abs() / dvol.iloc[i - 59:i + 1]).mean() * 1e9
        out.append(pd.DataFrame({
            "cutoff": c, "ticker": adj.columns,
            "ret_1m": (a / adj.iloc[i - 21] - 1).to_numpy(),
            "mom_12_1": (adj.iloc[i - 21] / adj.iloc[i - 252] - 1).to_numpy(),
            "vol_60d": r60.std().to_numpy(), "beta_252": beta.to_numpy(),
            "amihud": np.log1p(am).to_numpy(), "dollar_vol": dvol.iloc[i - 59:i + 1].mean().to_numpy(),
            "adj_close": a.to_numpy()}))
    return pd.concat(out), adj, close


def main() -> None:
    tabs, names, tick = download()
    U = pd.read_parquet(PROC / "universe.parquet", columns=["period", "cusip"])
    C = pd.read_parquet(PROC / "cusip_map.parquet")[["cusip", "ticker", "name"]]
    U = U.merge(C, on="cusip", how="left")
    U["q"] = U["period"].dt.year * 4 + (U["period"].dt.month - 1) // 3 + 1   # predicting quarter t from universe at t-1
    U["cutoff"] = U["period"]                                                 # end of quarter t-1
    # CIK: by ticker, then by exact normalized name (covers delisted companies)
    U["tkey"] = U["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    U = U.merge(tick.rename(columns={"ticker": "tkey"}).drop_duplicates("tkey"), on="tkey", how="left")
    nm = names.assign(n=names["name"].map(norm)).drop_duplicates("n").set_index("n")["cik"]
    by_name = U["name"].map(norm).map(nm)
    U["cik"] = U["cik"].fillna(by_name)
    print(f"CIK matched: {U['cik'].notna().mean():.1%} of universe stock-quarters", flush=True)
    F = U.dropna(subset=["cik"]).copy()
    F["cik"] = F["cik"].astype(int)

    for col in ["assets", "equity", "liabilities", "cash", "shares"]:
        F = asof(F, tabs[col], "cutoff", col, LAG)
    F = asof(F, tabs["assets"], "cutoff", "assets_1y", LAG + pd.Timedelta(days=365))
    rev = pd.concat([tabs["rev1"], tabs["rev2"], tabs["rev3"]]).sort_values("val").drop_duplicates(["cik", "end"], keep="last")
    for col, tab in [("ni", tabs["ni"]), ("gp", tabs["gp"]), ("opinc", tabs["opinc"]), ("rev", rev)]:
        F = asof(F, tab, "cutoff", col, LAG)
    F = asof(F, rev, "cutoff", "rev_1y", LAG + pd.Timedelta(days=365))
    # date of the shares figure, to scale its price to the cutoff (adjusted prices absorb splits)
    sh = tabs["shares"][["cik", "end", "val"]].dropna().rename(columns={"val": "sh2", "end": "sh_end"})
    sh["sh_end"], sh["cik"] = pd.to_datetime(sh["sh_end"]), sh["cik"].astype(int)
    sh = sh.sort_values("sh_end")
    F = pd.merge_asof(F.assign(_d=F["cutoff"] - LAG).sort_values("_d"), sh, left_on="_d", right_on="sh_end",
                      by="cik", direction="backward", tolerance=STALE).drop(columns=["_d", "sh2"])

    cut = pd.DatetimeIndex(sorted(F["cutoff"].unique()))
    PF, adj, close = price_features(cut)
    F["ticker_u"] = F["ticker"].astype(str).str.upper()
    F = F.merge(PF.rename(columns={"ticker": "ticker_u"}), on=["cutoff", "ticker_u"], how="left")
    # market cap at the cutoff = shares (dated sh_end) x unadjusted close at sh_end x adj-price growth since
    idx = adj.index
    def px(t, d, frame):
        if pd.isna(d) or t not in frame.columns:
            return np.nan
        j = idx.searchsorted(d, side="right") - 1
        return frame[t].iloc[j] if j >= 0 else np.nan
    keys = F[["ticker_u", "sh_end"]].drop_duplicates()
    keys["c0"] = [px(t, d, close) for t, d in zip(keys["ticker_u"], keys["sh_end"])]
    keys["a0"] = [px(t, d, adj) for t, d in zip(keys["ticker_u"], keys["sh_end"])]
    F = F.merge(keys, on=["ticker_u", "sh_end"], how="left")
    mcap = F["shares"] * F["c0"] * F["adj_close"] / F["a0"]
    F["mcap"] = mcap.where(mcap > 1e6)

    pos = lambda x: x.where(x > 0)
    F["book_to_market"] = F["equity"] / F["mcap"]
    F["earnings_yield"] = F["ni"] / F["mcap"]
    F["roa"] = F["ni"] / pos(F["assets"])
    F["gross_prof"] = F["gp"] / pos(F["assets"])
    F["op_margin"] = F["opinc"] / pos(F["rev"])
    F["leverage"] = F["liabilities"] / pos(F["assets"])
    F["cash_assets"] = F["cash"] / pos(F["assets"])
    F["asset_growth"] = F["assets"] / pos(F["assets_1y"]) - 1
    F["sales_growth"] = F["rev"] / pos(F["rev_1y"]) - 1
    F["turnover"] = F["dollar_vol"] / F["mcap"]
    for c in FEATURES:  # winsorize ratios at 1/99% per quarter so bad XBRL values cannot dominate
        lo, hi = F.groupby("q")[c].transform(lambda s: s.quantile(0.01)), F.groupby("q")[c].transform(lambda s: s.quantile(0.99))
        F[c] = F[c].clip(lo, hi)
    out = F[["cusip", "q"] + FEATURES].drop_duplicates(["cusip", "q"])
    out.to_parquet(PROC / "firm_chars.parquet", index=False)
    print(out[FEATURES].notna().mean().round(3).to_string())
    print(f"saved {len(out):,} stock-quarters")


if __name__ == "__main__":
    main()
