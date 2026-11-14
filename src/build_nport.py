"""Step 20: fund-level mutual fund holdings from SEC Form N-PORT (review W1).

Builds two samples that run through the identical pipeline:
  nport        individual active U.S. equity mutual funds (one N-PORT series =
               one fund), the unit Cohen, Lu & Nguyen study
  nportparent  the same funds added up to their registrant (the fund trust),
               mimicking the firm-level aggregation of a 13F report

From each quarterly N-PORT data set we keep reports whose period ends on a
calendar quarter-end, the latest filing per (series, period), and long
positions in common equity (ASSET_CAT = EC, units = shares) in stocks that
are ever in our 500-stock universe.

Fund selection, as of the first N-PORT quarter (2019-12-31), mirroring the
long-only 13F rule:
  - at least 20 universe stocks, and universe stocks >= 60% of net assets
    (a U.S. large-cap equity fund);
  - active, not index-like: active share >= 0.6 against the value-weighted
    universe (Cremers & Petajisto);
  - not an ETF (series name contains "ETF"): N-PORT has no ETF flag;
  - the 300 largest by net assets.
Parents: registrants with at least two selected funds.

Outputs:
  data/processed_nport/holdings.parquet, data/fund_list_nport.csv
  data/processed_nportparent/holdings.parquet, data/fund_list_nportparent.csv

Usage: .venv/bin/python src/build_nport.py
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nport"
SELECT_AT = pd.Timestamp("2019-12-31")
TOP_N = 300


def read(zf, name, cols, chunks=False):
    f = zf.open(name)
    kw = dict(sep="\t", usecols=lambda c: c in cols, dtype=str, on_bad_lines="skip", quoting=3)
    if chunks:
        return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), chunksize=2_000_000, **kw)
    return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), **kw)


def scan(zips, univ, series=None, periods=None):
    """Quarter-end N-PORT reports -> long common-equity positions in universe stocks.
    series: keep only these SERIES_IDs; periods: keep only these report dates."""
    parts, infos = [], []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            sub = read(zf, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "REPORT_DATE"])
            sub["period"] = pd.to_datetime(sub["REPORT_DATE"], format="%d-%b-%Y", errors="coerce")
            sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
            sub = sub[sub["period"].dt.is_quarter_end & (sub["period"] >= SELECT_AT)]
            if periods is not None:
                sub = sub[sub["period"].isin(periods)]
            info = read(zf, "FUND_REPORTED_INFO.tsv", ["ACCESSION_NUMBER", "SERIES_ID", "SERIES_NAME", "NET_ASSETS"])
            reg = read(zf, "REGISTRANT.tsv", ["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME"])
            info = info.merge(sub, on="ACCESSION_NUMBER").merge(reg, on="ACCESSION_NUMBER", how="left")
            info = info[info["SERIES_ID"].notna()]
            if series is not None:
                info = info[info["SERIES_ID"].isin(series)]
            keep_acc = set(info["ACCESSION_NUMBER"])
            n = 0
            for ch in read(zf, "FUND_REPORTED_HOLDING.tsv",
                           ["ACCESSION_NUMBER", "ISSUER_NAME", "ISSUER_CUSIP", "BALANCE", "UNIT", "CURRENCY_VALUE",
                            "PAYOFF_PROFILE", "ASSET_CAT"], chunks=True):
                ch = ch[ch["ACCESSION_NUMBER"].isin(keep_acc) & (ch["ASSET_CAT"] == "EC") & (ch["UNIT"] == "NS")
                        & (ch["PAYOFF_PROFILE"] == "Long") & ch["ISSUER_CUSIP"].isin(univ)]
                n += len(ch)
                parts.append(ch[["ACCESSION_NUMBER", "ISSUER_NAME", "ISSUER_CUSIP", "BALANCE", "CURRENCY_VALUE"]])
            infos.append(info)
            print(f"{z.name}: {len(info):,} quarter-end reports kept, {n:,} universe equity positions", flush=True)
    I = pd.concat(infos).sort_values("filed").drop_duplicates(["SERIES_ID", "period"], keep="last")
    Hh = pd.concat(parts).merge(I[["ACCESSION_NUMBER", "SERIES_ID", "SERIES_NAME", "CIK", "REGISTRANT_NAME", "period",
                                   "filed", "NET_ASSETS"]], on="ACCESSION_NUMBER")  # latest filing per series-period only
    Hh["shares"] = pd.to_numeric(Hh["BALANCE"], errors="coerce")
    Hh["value_usd"] = pd.to_numeric(Hh["CURRENCY_VALUE"], errors="coerce")
    Hh = Hh.groupby(["SERIES_ID", "period", "ISSUER_CUSIP"], as_index=False).agg(
        shares=("shares", "sum"), value_usd=("value_usd", "sum"), issuer=("ISSUER_NAME", "first"),
        name=("SERIES_NAME", "first"), cik=("CIK", "first"), registrant=("REGISTRANT_NAME", "first"),
        filed=("filed", "first"), net_assets=("NET_ASSETS", "first"))
    return Hh.rename(columns={"ISSUER_CUSIP": "cusip"})


def main() -> None:
    univ = set(pd.read_parquet(ROOT / "data/processed/universe.parquet", columns=["cusip"])["cusip"])
    zips = sorted(RAW.glob("*_nport.zip"))
    # pass 1: the files that can contain 2019-12-31 reports (filed within 60 days)
    Hh = scan([z for z in zips if z.name[:6] in ("2019q4", "2020q1")], univ, periods=[SELECT_AT])
    # selection as of 2019-12-31
    U = pd.read_parquet(ROOT / "data/processed/universe.parquet", columns=["period", "cusip", "inst_value_usd"])
    u0 = U[U["period"] == SELECT_AT]
    mkt = u0.set_index("cusip")["inst_value_usd"] / u0["inst_value_usd"].sum()
    h0 = Hh[(Hh["period"] == SELECT_AT) & Hh["cusip"].isin(mkt.index)]
    g = h0.groupby("SERIES_ID")
    stats = pd.DataFrame({"n": g.size(), "uval": g["value_usd"].sum(),
                          "na": pd.to_numeric(g["net_assets"].first(), errors="coerce"), "name": g["name"].first(),
                          "cik": g["cik"].first(), "registrant": g["registrant"].first()})
    def act_share(d):
        w = d.groupby("cusip")["value_usd"].sum()
        w = w / w.sum()
        return 0.5 * (w.reindex(mkt.index, fill_value=0) - mkt).abs().sum()
    stats["active_share"] = h0.groupby("SERIES_ID").apply(act_share, include_groups=False)
    ok = (stats["n"] >= 20) & (stats["uval"] >= 0.6 * stats["na"]) & (stats["active_share"] >= 0.6) \
        & ~stats["name"].str.upper().str.contains(r"\bETF\b", na=False)
    sel = stats[ok].sort_values("na", ascending=False).head(TOP_N)
    print(f"series at {SELECT_AT.date()}: {len(stats):,}; pass the equity / active / non-ETF rules: {int(ok.sum())}; keep {len(sel)}")
    print(f"dropped as index-like (active share < 0.6) among large-cap equity funds: "
          f"{int(((stats['n'] >= 20) & (stats['uval'] >= 0.6 * stats['na']) & (stats['active_share'] < 0.6)).sum())}")

    # pass 2: every file, selected funds only
    H = scan(zips, univ, series=set(sel.index))
    # one stable name per fund: the series name at selection (funds get renamed)
    H["fund"] = H["SERIES_ID"] + " " + H["SERIES_ID"].map(sel["name"]).str.slice(0, 40)
    out = H[["fund", "cik", "period", "cusip", "issuer", "shares", "value_usd", "filed"]].copy()
    out["cik"] = out["cik"].fillna("0").astype(str)
    for sample, df in [("nport", out)]:
        d = ROOT / "data" / f"processed_{sample}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "holdings.parquet", index=False)
        fl = df.groupby("fund")["cik"].first().reset_index()
        pd.DataFrame({"fund": fl["fund"], "style": "mutual fund (N-PORT series)", "ciks": fl["cik"],
                      "include": "yes", "note": ""}).to_csv(ROOT / "data" / f"fund_list_{sample}.csv", index=False)

    # parents: registrants with >= 2 selected funds; shares summed across their funds
    multi = sel.groupby("cik").size()
    multi = multi[multi >= 2].index
    H["cik"] = H["SERIES_ID"].map(sel["cik"])  # registrant at selection (a fund can move trusts)
    P = H[H["cik"].isin(multi)].copy()
    reg_name = sel.groupby("cik")["registrant"].first()
    P["fund"] = "REG " + P["cik"] + " " + P["cik"].map(reg_name).str.slice(0, 40)
    agg = P.groupby(["fund", "cik", "period", "cusip"], as_index=False).agg(
        issuer=("issuer", "first"), shares=("shares", "sum"), value_usd=("value_usd", "sum"), filed=("filed", "max"))
    # a parent-quarter only exists if every one of its funds reported that quarter
    nf_sel = P.groupby("fund")["SERIES_ID"].nunique()
    nf_q = P.groupby(["fund", "period"])["SERIES_ID"].nunique()
    complete = nf_q[nf_q == nf_sel.reindex(nf_q.index.get_level_values(0)).to_numpy()].index
    agg = agg.set_index(["fund", "period"]).loc[lambda d: d.index.isin(complete)].reset_index()
    d = ROOT / "data" / "processed_nportparent"
    d.mkdir(parents=True, exist_ok=True)
    agg[["fund", "cik", "period", "cusip", "issuer", "shares", "value_usd", "filed"]].to_parquet(d / "holdings.parquet", index=False)
    fl = agg.groupby("fund")["cik"].first().reset_index()
    pd.DataFrame({"fund": fl["fund"], "style": "mutual fund family (sum of N-PORT series)", "ciks": fl["cik"],
                  "include": "yes", "note": ""}).to_csv(ROOT / "data" / "fund_list_nportparent.csv", index=False)
    # which funds belong to a parent (for the fund-vs-parent comparison)
    H[H["cik"].isin(multi)][["fund", "cik"]].drop_duplicates().to_csv(ROOT / "data" / "nport_fund_parent.csv", index=False)
    sel.to_csv(ROOT / "data" / "processed_nport" / "selection.csv")
    print(f"funds: {out['fund'].nunique()}, quarters {out['period'].nunique()}; parents: {agg['fund'].nunique()} "
          f"covering {H[H['cik'].isin(multi)]['fund'].nunique()} funds")


if __name__ == "__main__":
    main()
