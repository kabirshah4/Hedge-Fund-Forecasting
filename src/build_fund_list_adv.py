"""Step 11 (Phase 2): a rule-based hedge fund list from SEC Form ADV.

Why: our first list (data/fund_list.csv) was hand-picked in 2026 from funds
we had heard of, which is hindsight: famous survivors. This builds a list
from rules applied to data that existed BEFORE the sample starts.

Rules (as of mid-2013, before the first labels in 2013Q3):
  1. Take each registered adviser's latest Form ADV filed between
     2012-07-01 and 2013-06-30.
  2. From Schedule D 7.B.(1) (the adviser's private funds), sum the gross
     asset value (GAV) of funds it marks as "Hedge Fund", skipping feeder
     funds and funds of funds (their money is already counted in a master
     fund, or belongs to other managers).
  3. Hedge fund adviser = hedge fund GAV >= $1 billion, hedge funds are at
     least half of its private-fund GAV, AND hedge fund GAV is at least 25%
     of its total regulatory assets (drops giant traditional managers such as
     PIMCO or Wellington that also run some hedge funds).
  4. Link the adviser to a 13F filer: first by any CIK it reports in ADV
     Item 1.D.3 (this item starts in 2017; a CIK is only an identifier, so
     using a later filing to look it up reveals nothing about performance),
     else by its legal name: exact after removing punctuation and legal
     suffixes, or else a looser key (common words removed) that must have at
     least two words and be unique among filers.
     Keep it if that CIK filed a 13F for 2013-06-30 with >= 10 holdings.
  5. Keep the 100 largest by hedge fund GAV.

Outputs:
  data/processed/adv_hedge_advisers.csv   every adviser passing rule 3, with match info
  data/fund_list_adv.csv                  the final list, same columns as fund_list.csv

Usage: .venv/bin/python src/build_fund_list_adv.py
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

from build_holdings import read

ROOT = Path(__file__).resolve().parent.parent
ADV = ROOT / "data" / "raw" / "adv" / "extracted"
RAW13F = ROOT / "data" / "raw" / "13f"
PROC = ROOT / "data" / "processed"

ASOF_FROM, ASOF_TO = pd.Timestamp("2012-07-01"), pd.Timestamp("2013-06-30")
MIN_HEDGE_GAV = 1e9
MIN_HEDGE_SHARE = 0.5
MIN_HEDGE_OF_RAUM = 0.25
TOP_N = 100
MIN_HOLDINGS = 10

STOP = r"\b(LLC|L\.?L\.?C|LP|L\.?P|LTD|LIMITED|INC|CORP|CORPORATION|CO|COMPANY|MANAGEMENT|MGMT|CAPITAL|ADVISORS|ADVISERS|ADVISORY|PARTNERS|GROUP|HOLDINGS|INVESTMENTS?|ASSET|ASSETS|GLOBAL|FUND|FUNDS|THE|AND|&)\b"


LEGAL = {"LLC", "L L C", "LP", "L P", "LLP", "LTD", "LIMITED", "INC", "CORP", "CORPORATION",
         "CO", "COMPANY", "PARTNERSHIP", "THE"}


def legal_norm(name: str) -> str:
    """Remove punctuation, EDGAR state tags ("/MA"), "ET AL" and trailing
    legal suffixes; spaces are dropped at the end so "S A C" == "SAC"."""
    s = str(name).upper()
    s = re.sub(r"\s*/\s*[A-Z]{2,3}\s*/?\s*$", "", s)  # BAUPOST GROUP LLC/MA
    s = re.sub(r"\bET AL\b\.?", "", s)
    s = re.sub(r"[^A-Z0-9& ]", " ", s)
    s = re.sub(r"\bL L C\b", "LLC", re.sub(r"\bL P\b", "LP", re.sub(r"\s+", " ", s))).strip()
    toks = s.split()
    while toks and (toks[-1] in LEGAL or toks[-1] == "&"):
        toks.pop()
    if toks and toks[0] == "THE":
        toks = toks[1:]
    return "".join(toks)


def norm(name: str) -> str:
    s = re.sub(r"[^A-Z0-9& ]", " ", str(name).upper())
    s = re.sub(STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def adv_advisers() -> pd.DataFrame:
    B = pd.read_csv(ADV / "IA_ADV_Base_A_20111105_20241231.csv",
                    usecols=["FilingID", "DateSubmitted", "1A", "1E1", "5F2c"],
                    encoding="latin-1", low_memory=False)
    B["date"] = pd.to_datetime(B["DateSubmitted"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    B = B[(B["date"] >= ASOF_FROM) & (B["date"] <= ASOF_TO)]
    B = B.sort_values("date").groupby("1E1").tail(1)  # latest filing per adviser (CRD)
    D = pd.read_csv(ADV / "IA_Schedule_D_7B1_20111105_20241231.csv",
                    usecols=["FilingID", "Feeder Fund", "Fund of Funds", "Fund Type", "Gross Asset Value"],
                    encoding="latin-1", low_memory=False)
    D = D[D["FilingID"].isin(B["FilingID"])]
    D = D[(D["Feeder Fund"] != "Y") & (D["Fund of Funds"] != "Y")]
    D["gav"] = pd.to_numeric(D["Gross Asset Value"], errors="coerce").fillna(0)
    g = D.groupby("FilingID").apply(
        lambda d: pd.Series({"hedge_gav": d.loc[d["Fund Type"] == "Hedge Fund", "gav"].sum(),
                             "private_gav": d["gav"].sum(),
                             "n_hedge_funds": int((d["Fund Type"] == "Hedge Fund").sum())}),
        include_groups=False).reset_index()
    A = B.merge(g, on="FilingID")
    A["hedge_share"] = A["hedge_gav"] / A["private_gav"]
    A["hedge_of_raum"] = A["hedge_gav"] / A["5F2c"]
    A = A[(A["hedge_gav"] >= MIN_HEDGE_GAV) & (A["hedge_share"] >= MIN_HEDGE_SHARE)
          & (A["hedge_of_raum"] >= MIN_HEDGE_OF_RAUM)]
    A = A.rename(columns={"1A": "adviser", "1E1": "crd", "5F2c": "raum"})
    # CIKs reported in Item 1.D.3 on any of the adviser's filings up to the as-of date
    C = pd.read_csv(ADV / "IA_1D3_CIK_20111105_20241231.csv")
    allB = pd.read_csv(ADV / "IA_ADV_Base_A_20111105_20241231.csv", usecols=["FilingID", "1E1", "DateSubmitted"],
                       encoding="latin-1", low_memory=False)
    allB["date"] = pd.to_datetime(allB["DateSubmitted"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    C = C.merge(allB, on="FilingID")
    ciks = C.groupby("1E1")["CIK"].apply(lambda s: sorted(set(s.astype(int))))
    A["adv_ciks"] = A["crd"].map(ciks)
    return A.sort_values("hedge_gav", ascending=False)


def filers_2013() -> pd.DataFrame:
    """13F filers for period 2013-06-30: CIK, name, number of holdings."""
    with zipfile.ZipFile(RAW13F / "2013q3_form13f.zip") as zf:
        sub = read(zf, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT", "SUBMISSIONTYPE"])
        cov = read(zf, "COVERPAGE.TSV", ["ACCESSION_NUMBER", "FILINGMANAGER_NAME"])
        summ = read(zf, "SUMMARYPAGE.TSV", ["ACCESSION_NUMBER", "TABLEENTRYTOTAL"])
    f = sub.merge(cov, on="ACCESSION_NUMBER").merge(summ, on="ACCESSION_NUMBER", how="left")
    f = f[f["PERIODOFREPORT"].str.upper().str.contains("JUN-2013") & f["SUBMISSIONTYPE"].str.startswith("13F-HR")]
    f["n"] = pd.to_numeric(f["TABLEENTRYTOTAL"], errors="coerce").fillna(0)
    f = f.groupby("CIK").agg(name=("FILINGMANAGER_NAME", "last"), n=("n", "max")).reset_index()
    f["CIK"] = f["CIK"].astype(int)
    f["key"] = f["name"].map(norm)
    f["legal"] = f["name"].map(legal_norm)
    return f


def link(A: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    F = F[F["n"] >= MIN_HOLDINGS]
    byc = F.set_index("CIK")
    by_legal = F.sort_values("n").groupby("legal").tail(1).set_index("legal")
    key_counts = F["key"].value_counts()
    by_key = F[F["key"].map(key_counts) == 1].set_index("key")
    out = []
    for r in A.itertuples():
        cik, how = None, None
        cands = [c for c in (r.adv_ciks if isinstance(r.adv_ciks, list) else []) if c in byc.index]
        if cands:  # several entities may file: take the one with the most holdings
            cik, how = max(cands, key=lambda c: byc.loc[c, "n"]), "ADV CIK"
        else:
            lg, k = legal_norm(r.adviser), norm(r.adviser)
            if lg in by_legal.index:
                cik, how = int(by_legal.loc[lg, "CIK"]), "exact name"
            elif len(k.split()) >= 2 and k in by_key.index:
                cik, how = int(by_key.loc[k, "CIK"]), "loose name"
        out.append((cik, how, byc.loc[cik, "name"] if cik is not None else None,
                    byc.loc[cik, "n"] if cik is not None else None))
    A = A.copy()
    A[["cik_13f", "link", "name_13f", "holdings_13f"]] = pd.DataFrame(out, index=A.index)
    return A


def main() -> None:
    A = adv_advisers()
    print(f"advisers with hedge fund GAV >= ${MIN_HEDGE_GAV/1e9:.0f}B and >= {MIN_HEDGE_SHARE:.0%} of private-fund GAV: {len(A)}")
    F = filers_2013()
    A = link(A, F)
    A.to_csv(PROC / "adv_hedge_advisers.csv", index=False)
    print("linked to a 13F filer:", A["link"].value_counts().to_dict(), f"of {len(A)}")
    L = A[A["cik_13f"].notna()].drop_duplicates("cik_13f").head(TOP_N)
    print(f"final list: {len(L)} advisers; hedge fund GAV from ${L['hedge_gav'].min()/1e9:.1f}B to ${L['hedge_gav'].max()/1e9:.0f}B")
    out = pd.DataFrame({
        "fund": L["name_13f"].str.title(),
        "style": "",
        "ciks": L["cik_13f"].astype(int).astype(str).str.zfill(10),
        "include": "yes",
        "note": "ADV " + L["adviser"].str.slice(0, 50) + "; hedge GAV $" + (L["hedge_gav"] / 1e9).round(1).astype(str) + "B; " + L["link"],
    })
    out.to_csv(ROOT / "data" / "fund_list_adv.csv", index=False)
    old = pd.read_csv(ROOT / "data" / "fund_list.csv", dtype=str)
    old_ciks = {c.lstrip("0") for cs in old.loc[old["include"] == "yes", "ciks"] for c in cs.split(";")}
    overlap = out["ciks"].str.lstrip("0").isin(old_ciks).sum()
    print(f"overlap with the hand-picked list: {overlap} of {len(out)}")
    pd.set_option("display.width", 200)
    print(L[["adviser", "hedge_gav", "hedge_of_raum", "link", "name_13f", "holdings_13f"]].to_string(index=False))


if __name__ == "__main__":
    main()
