"""Step 2a: pull our hedge funds' quarterly stock holdings out of the 13F data.

For every quarterly SEC file this script:
  1. finds the filings made by the funds in data/fund_list.csv (by SEC ID / CIK),
  2. picks the right filing per fund and quarter (handling amendments),
  3. keeps only plain shares of stock (no options, no bonds),
  4. adds up rows for the same stock (one fund can report a stock on several rows),
  5. also computes, from ALL filers, a typical price per stock per quarter
     (value / shares). Used later to spot stock splits.

Outputs:
  data/processed/holdings.parquet        fund, cik, period, cusip, issuer, shares, value_usd
  data/processed/implied_prices.parquet  cusip, period, implied_price, n_holders, inst_value_usd

Usage:
  .venv/bin/python src/build_holdings.py
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "13f"
OUT = SMP.PROC

# The SEC changed VALUE from thousands of dollars to whole dollars for filings
# made on or after 3 Jan 2023.
DOLLARS_FROM = pd.Timestamp("2023-01-03")


def read(zf: zipfile.ZipFile, table: str, cols: list[str]) -> pd.DataFrame:
    name = next(n for n in zf.namelist() if n.upper().endswith(table))
    with zf.open(name) as f:
        return pd.read_csv(
            io.TextIOWrapper(f, encoding="latin-1"),
            sep="\t",
            usecols=lambda c: c in cols,
            dtype=str,
            on_bad_lines="skip",
            quoting=3,  # the files are plain tab-separated; quotes are literal text
        )


def fund_ids() -> dict[str, tuple[str, str]]:
    """CIK -> (fund name, include flag), for every included or control fund."""
    funds = pd.read_csv(SMP.FUND_LIST, dtype=str)
    funds = funds[funds["include"].isin(["yes", "control"])]
    out = {}
    for row in funds.itertuples():
        for cik in row.ciks.split(";"):
            out[cik] = (row.fund, row.include)
    return out


def pick_filings(sub: pd.DataFrame) -> pd.DataFrame:
    """One filing set per fund and quarter.

    Amendments come in two kinds:
      RESTATEMENT   replaces the earlier report: keep only the latest one
      NEW HOLDINGS  adds positions left out before: keep these as well
    """
    sub = sub.copy()
    sub["kind"] = sub["AMENDMENTTYPE"].fillna("").str.upper()
    base = sub[~sub["kind"].str.contains("NEW HOLDINGS")]
    base = base.sort_values("FILING_DATE").groupby(["CIK", "PERIOD"]).tail(1)
    extra = sub[sub["kind"].str.contains("NEW HOLDINGS")]
    return pd.concat([base, extra])


def process(zip_path: Path, ids: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        sub = read(zf, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
        cov = read(zf, "COVERPAGE.TSV", ["ACCESSION_NUMBER", "AMENDMENTTYPE", "REPORTTYPE"])
        info = read(
            zf,
            "INFOTABLE.TSV",
            ["ACCESSION_NUMBER", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"],
        )

    sub = sub.merge(cov, on="ACCESSION_NUMBER", how="left")
    sub = sub[sub["SUBMISSIONTYPE"].str.startswith("13F-HR", na=False)]  # holdings reports only
    sub["PERIOD"] = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    sub["FILING_DATE"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    sub = sub.dropna(subset=["PERIOD", "FILING_DATE"])

    # Plain shares only: no options (PUTCALL set) and no bonds (PRN = principal amount).
    info = info[(info["SSHPRNAMTTYPE"] == "SH") & info["PUTCALL"].isna()]
    info = info.assign(
        shares=pd.to_numeric(info["SSHPRNAMT"], errors="coerce"),
        value=pd.to_numeric(info["VALUE"], errors="coerce"),
        CUSIP=info["CUSIP"].str.upper().str.strip(),
    ).dropna(subset=["shares", "value"])
    info = info[(info["shares"] > 0) & (info["CUSIP"].str.len() == 9)]

    # Value in dollars, whichever unit the filing used.
    unit = sub.set_index("ACCESSION_NUMBER")["FILING_DATE"].map(lambda d: 1 if d >= DOLLARS_FROM else 1000)
    info["value_usd"] = info["value"] * info["ACCESSION_NUMBER"].map(unit).fillna(1)

    # Typical price per stock per quarter, from every filer (median is robust to typos).
    priced = info.merge(sub[["ACCESSION_NUMBER", "PERIOD"]], on="ACCESSION_NUMBER")
    priced = priced.assign(price=priced["value_usd"] / priced["shares"])
    priced = priced[(priced["price"] > 0) & (priced["price"] < 1e6)]
    prices = (
        priced.groupby(["CUSIP", "PERIOD"])
        .agg(implied_price=("price", "median"), n_holders=("price", "size"))
        .reset_index()
    )
    # Robust institutional value: a filer whose own price (value / shares) is
    # more than 5x away from the typical price has a unit mistake in one of
    # its columns (e.g. dollars typed where shares belong); leave that row out.
    # Sum the rest as shares x typical price.
    priced = priced.merge(prices[["CUSIP", "PERIOD", "implied_price"]], on=["CUSIP", "PERIOD"])
    ratio = priced["price"] / priced["implied_price"]
    clean = priced[(ratio > 0.2) & (ratio < 5)]
    inst = (
        clean.assign(v=clean["shares"] * clean["implied_price"])
        .groupby(["CUSIP", "PERIOD"])["v"].sum()
        .rename("inst_value_usd")
        .reset_index()
    )
    prices = prices.merge(inst, on=["CUSIP", "PERIOD"], how="left")

    # Our funds' holdings.
    ours = pick_filings(sub[sub["CIK"].isin(ids)])
    held = info.merge(ours[["ACCESSION_NUMBER", "CIK", "PERIOD", "FILING_DATE"]], on="ACCESSION_NUMBER")
    held = (
        held.groupby(["CIK", "PERIOD", "CUSIP"])
        .agg(
            issuer=("NAMEOFISSUER", "first"),
            shares=("shares", "sum"),
            value_usd=("value_usd", "sum"),
            filed=("FILING_DATE", "max"),
        )
        .reset_index()
    )
    held.insert(0, "fund", held["CIK"].map(lambda c: ids[c][0]))
    return held, prices


def main() -> None:
    ids = fund_ids()
    zips = sorted(RAW.glob("*.zip"))
    holdings, prices = [], []
    for i, z in enumerate(zips, 1):
        h, p = process(z, ids)
        holdings.append(h)
        prices.append(p)
        print(f"[{i}/{len(zips)}] {z.name}: {len(h):,} holdings, {h['fund'].nunique()} funds")

    # The same quarter can appear in two files (late or amended filings):
    # keep the most recently filed copy.
    H = (
        pd.concat(holdings)
        .sort_values("filed")
        .drop_duplicates(["CIK", "PERIOD", "CUSIP"], keep="last")
    )
    P = (
        pd.concat(prices)
        .sort_values("n_holders")
        .drop_duplicates(["CUSIP", "PERIOD"], keep="last")
    )
    H = H.rename(columns={"CIK": "cik", "PERIOD": "period", "CUSIP": "cusip"})
    P = P.rename(columns={"CUSIP": "cusip", "PERIOD": "period"})
    OUT.mkdir(parents=True, exist_ok=True)
    H.to_parquet(OUT / "holdings.parquet", index=False)
    if SMP.IS_MAIN:  # implied prices use all filers: same for every sample
        P.to_parquet(OUT / "implied_prices.parquet", index=False)
    print(f"\nholdings: {len(H):,} rows, {H['fund'].nunique()} funds, "
          f"{H['period'].min().date()} to {H['period'].max().date()}")
    print(f"implied prices: {len(P):,} stock-quarters")


if __name__ == "__main__":
    main()
