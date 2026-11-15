"""Find the SEC ID numbers (CIKs) of candidate hedge funds in the 13F data.

We do NOT trust CIKs typed from memory. Instead, we search the filer names
inside the downloaded 13F data for each candidate fund and report every
match, with how many quarters it filed and its typical portfolio size.
A human (you) then checks the matches on EDGAR and picks the right one.

Usage (after download_13f.py --go):
    .venv/bin/python src/find_fund_ciks.py
Output:
    data/processed/fund_candidates.csv   every name match, for you to review
"""

import io
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "13f"
OUT = ROOT / "data" / "processed" / "fund_candidates.csv"

# Well-known hedge fund managers that file 13Fs. These are SEARCH TERMS only;
# the CIK and exact legal name come from the data, and every match is checked
# by hand. Mix of styles: multi-strategy, quant, long/short equity, activist,
# macro, event-driven, tiger cubs.
CANDIDATES = [
    "CITADEL ADVISORS", "MILLENNIUM MANAGEMENT", "RENAISSANCE TECHNOLOGIES",
    "TWO SIGMA", "D. E. SHAW", "AQR CAPITAL", "BRIDGEWATER ASSOCIATES",
    "POINT72", "BALYASNY", "SCHONFELD", "EXODUSPOINT", "MARSHALL WACE",
    "TIGER GLOBAL", "COATUE", "VIKING GLOBAL", "LONE PINE", "MAVERICK CAPITAL",
    "D1 CAPITAL", "SUSQUEHANNA", "WINTON", "MAN GROUP", "BLUECREST",
    "PERSHING SQUARE", "ICAHN", "ELLIOTT", "THIRD POINT", "TRIAN",
    "STARBOARD VALUE", "JANA PARTNERS", "VALUEACT", "BAUPOST", "APPALOOSA",
    "GREENLIGHT CAPITAL", "SOROS FUND", "DUQUESNE FAMILY", "PAULSON & CO",
    "GLENVIEW CAPITAL", "SCION ASSET", "BERKSHIRE HATHAWAY",  # Berkshire = control, not a hedge fund
    "LANSDOWNE", "SLATE PATH", "LIGHT STREET", "WHALE ROCK", "DRAGONEER",
    "ALTIMETER", "HOLOCENE", "TCI FUND", "SIR CAPITAL", "CANYON CAPITAL",
    "SANDS CAPITAL", "AKO CAPITAL", "EMINENCE CAPITAL", "SAMLYN",
    "CANDLESTICK", "TOURADJI", "JERICHO CAPITAL", "DORSAL CAPITAL",
]


def read_table(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    member = next((n for n in zf.namelist() if n.upper().endswith(name)), None)
    if member is None:
        return pd.DataFrame(columns=usecols)
    with zf.open(member) as f:
        return pd.read_csv(
            io.TextIOWrapper(f, encoding="latin-1"), sep="\t", usecols=usecols,
            dtype=str, on_bad_lines="skip",
        )


def main() -> None:
    zips = sorted(RAW.glob("*.zip"))
    if not zips:
        raise SystemExit("No 13F zips found. Run: .venv/bin/python src/download_13f.py --go")

    frames = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            sub = read_table(zf, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT"])
            cov = read_table(zf, "COVERPAGE.TSV", ["ACCESSION_NUMBER", "FILINGMANAGER_NAME"])
            summ = read_table(zf, "SUMMARYPAGE.TSV", ["ACCESSION_NUMBER", "TABLEVALUETOTAL"])
        frames.append(sub.merge(cov, on="ACCESSION_NUMBER").merge(summ, on="ACCESSION_NUMBER", how="left"))
    filings = pd.concat(frames, ignore_index=True)
    filings["NAME"] = filings["FILINGMANAGER_NAME"].str.upper().str.strip()
    filings["TABLEVALUETOTAL"] = pd.to_numeric(filings["TABLEVALUETOTAL"], errors="coerce")
    # Dates look like 31-MAR-2026; parse them so min/max are by date, not text.
    filings["PERIOD"] = pd.to_datetime(filings["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")

    rows = []
    for term in CANDIDATES:
        hits = filings[filings["NAME"].str.contains(term, regex=False, na=False)]
        for (cik, name), g in hits.groupby(["CIK", "NAME"]):
            rows.append({
                "search_term": term,
                "cik": cik,
                "filer_name": name,
                "quarters_filed": g["PERIOD"].nunique(),
                "first_period": g["PERIOD"].min().date(),
                "last_period": g["PERIOD"].max().date(),
                "median_reported_value": g["TABLEVALUETOTAL"].median(),
                "keep": "",  # you fill in: yes / no
            })
    out = pd.DataFrame(rows).sort_values(["search_term", "quarters_filed"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    found = set(out["search_term"])
    print(f"{len(out)} matches for {len(found)}/{len(CANDIDATES)} search terms -> {OUT}")
    missing = [t for t in CANDIDATES if t not in found]
    if missing:
        print("No match (check spelling on EDGAR):", ", ".join(missing))
    print("\nNote: the units of TABLEVALUETOTAL changed in 2023 (thousands -> dollars);"
          " compare sizes within a period, not across.")


if __name__ == "__main__":
    main()
