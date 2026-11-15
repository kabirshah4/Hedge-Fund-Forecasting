"""Step 15: a matched long-only comparison sample (mutual-fund managers) from Form ADV.

Why: the paper's headline compares hedge funds with Cohen, Lu & Nguyen's
mutual funds, which used different data, a different model and a different
unit (a single fund, not a whole firm). Running OUR identical pipeline on
long-only, mutual-fund-style 13F filers makes the comparison one study.
Like hedge fund 13Fs, these filings cover a whole firm, so aggregation across
many portfolios is held constant.

Rules (as of July 2012 - June 2013, before the sample starts, like the hedge
fund list):
  1. Each adviser's latest Form ADV in that window.
  2. More than 75% of its assets under management come from investment
     companies, i.e. mutual funds (Item 5.D, 2012 form version: field 5D2d).
  3. Essentially no hedge funds: hedge fund gross assets < 5% of regulatory
     assets (Schedule D 7.B.(1)).
  4. Regulatory assets >= $1 billion; linked to a 13F filer with >= 10
     holdings for June 2013 (same linking as build_fund_list_adv.py).
  5. Active, not index-like: active share >= 0.6 against the value-weighted
     500-stock universe at June 2013 (Cremers & Petajisto's threshold), using
     only positions in universe stocks.
  6. The 100 largest by regulatory assets.

Outputs:
  data/processed/adv_longonly_advisers.csv   every candidate with its numbers
  data/fund_list_longonly.csv                the list (same columns as fund_list.csv)

Usage: .venv/bin/python src/build_fund_list_longonly.py
"""

import zipfile

import numpy as np
import pandas as pd

import build_fund_list_adv as BA
from build_holdings import read

MF_SHARE = ">75%"
MAX_HEDGE_OF_RAUM = 0.05
MIN_RAUM = 1e9
MIN_ACTIVE_SHARE = 0.6
TOP_N = 100


def candidates() -> pd.DataFrame:
    B = pd.read_csv(BA.ADV / "IA_ADV_Base_A_20111105_20241231.csv",
                    usecols=["FilingID", "DateSubmitted", "1A", "1E1", "5F2c"], encoding="latin-1", low_memory=False)
    B["date"] = pd.to_datetime(B["DateSubmitted"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    B = B[(B["date"] >= BA.ASOF_FROM) & (B["date"] <= BA.ASOF_TO)]
    B = B.sort_values("date").groupby("1E1").tail(1)
    V = pd.read_csv(BA.ADV / "IA_ADV_5D_Versions_2011_2012.csv", usecols=["FilingID", "5D2d"],
                    encoding="latin-1", low_memory=False)
    B = B.merge(V, on="FilingID", how="inner")
    B = B[(B["5D2d"] == MF_SHARE) & (pd.to_numeric(B["5F2c"], errors="coerce") >= MIN_RAUM)]
    D = pd.read_csv(BA.ADV / "IA_Schedule_D_7B1_20111105_20241231.csv",
                    usecols=["FilingID", "Feeder Fund", "Fund of Funds", "Fund Type", "Gross Asset Value"],
                    encoding="latin-1", low_memory=False)
    D = D[D["FilingID"].isin(B["FilingID"]) & (D["Feeder Fund"] != "Y") & (D["Fund of Funds"] != "Y")
          & (D["Fund Type"] == "Hedge Fund")]
    hedge = D.assign(g=pd.to_numeric(D["Gross Asset Value"], errors="coerce").fillna(0)).groupby("FilingID")["g"].sum()
    B["hedge_gav"] = B["FilingID"].map(hedge).fillna(0)
    B["raum"] = pd.to_numeric(B["5F2c"], errors="coerce")
    B = B[B["hedge_gav"] / B["raum"] < MAX_HEDGE_OF_RAUM]
    C = pd.read_csv(BA.ADV / "IA_1D3_CIK_20111105_20241231.csv")
    allB = pd.read_csv(BA.ADV / "IA_ADV_Base_A_20111105_20241231.csv", usecols=["FilingID", "1E1"],
                       encoding="latin-1", low_memory=False)
    ciks = C.merge(allB, on="FilingID").groupby("1E1")["CIK"].apply(lambda s: sorted(set(s.astype(int))))
    B = B.rename(columns={"1A": "adviser", "1E1": "crd"})
    B["adv_ciks"] = B["crd"].map(ciks)
    return B.sort_values("raum", ascending=False)


def active_share(ciks: list[int]) -> dict[int, float]:
    """Active share vs. the value-weighted universe, June 2013, universe stocks only."""
    U = pd.read_parquet(BA.PROC / "universe.parquet", columns=["period", "cusip", "inst_value_usd"])
    U = U[U["period"] == pd.Timestamp("2013-06-30")]
    mkt = (U.set_index("cusip")["inst_value_usd"] / U["inst_value_usd"].sum())
    with zipfile.ZipFile(BA.RAW13F / "2013q3_form13f.zip") as zf:
        sub = read(zf, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT", "SUBMISSIONTYPE"])
        sub = sub[sub["PERIODOFREPORT"].str.upper().str.contains("JUN-2013") & sub["SUBMISSIONTYPE"].str.startswith("13F-HR")]
        sub["CIK"] = sub["CIK"].astype(int)
        sub = sub[sub["CIK"].isin(ciks)]
        info = read(zf, "INFOTABLE.TSV", ["ACCESSION_NUMBER", "CUSIP", "VALUE", "PUTCALL"])
    info = info[info["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"]) & info["PUTCALL"].isna()]
    info = info.merge(sub[["ACCESSION_NUMBER", "CIK"]], on="ACCESSION_NUMBER")
    info["v"] = pd.to_numeric(info["VALUE"], errors="coerce")
    info = info[info["CUSIP"].isin(mkt.index)]
    out = {}
    for cik, g in info.groupby("CIK"):
        w = g.groupby("CUSIP")["v"].sum()
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        out[cik] = 0.5 * (w.reindex(mkt.index, fill_value=0) - mkt).abs().sum()
    return out


def main() -> None:
    A = candidates()
    print(f"advisers: >75% of AUM in mutual funds, < 5% hedge funds, >= $1B: {len(A)}")
    A = BA.link(A, BA.filers_2013())
    A = A[A["cik_13f"].notna()].drop_duplicates("cik_13f")
    print(f"linked to a 13F filer with >= {BA.MIN_HOLDINGS} holdings: {len(A)}")
    AS = active_share(A["cik_13f"].astype(int).tolist())
    A["active_share"] = A["cik_13f"].astype(int).map(AS)
    A.to_csv(BA.PROC / "adv_longonly_advisers.csv", index=False)
    L = A[A["active_share"] >= MIN_ACTIVE_SHARE].head(TOP_N)
    print(f"active share >= {MIN_ACTIVE_SHARE}: {(A['active_share'] >= MIN_ACTIVE_SHARE).sum()} -> keeping {len(L)}; "
          f"dropped as index-like: {(A['active_share'] < MIN_ACTIVE_SHARE).sum()}")
    out = pd.DataFrame({
        "fund": L["name_13f"].str.title(),
        "style": "long-only (mutual fund manager)",
        "ciks": L["cik_13f"].astype(int).astype(str).str.zfill(10),
        "include": "yes",
        "note": "ADV " + L["adviser"].str.slice(0, 50) + "; RAUM $" + (L["raum"] / 1e9).round(1).astype(str)
                + "B; active share " + L["active_share"].round(2).astype(str) + "; " + L["link"],
    })
    out.to_csv(BA.ROOT / "data" / "fund_list_longonly.csv", index=False)
    pd.set_option("display.width", 200)
    print(L[["adviser", "raum", "active_share", "link", "name_13f", "holdings_13f"]].head(40).to_string(index=False))
    print("\nlargest index-like (dropped):")
    print(A[A["active_share"] < MIN_ACTIVE_SHARE][["adviser", "raum", "active_share"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
