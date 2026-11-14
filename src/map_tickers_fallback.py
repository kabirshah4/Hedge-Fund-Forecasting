"""Step 4a (fallback): find US tickers for universe stocks the first pass missed.

For some stocks OpenFIGI's CUSIP lookup only returns foreign listings (Exxon came
back only as "EXMOC" on Chilean exchanges), or the CUSIP was retired (GE's
pre-2021 CUSIP). Every listing carries a share-class ID (shareClassFIGI) that is
the same in every country, so we:
  1. look the CUSIP up again (any exchange) to get its share-class ID,
  2. ask OpenFIGI for the US listing of that share class.

Only universe stocks without a US ticker are retried. Updates
data/processed/cusip_map.parquet in place (marks rows with fallback=True).

Usage:
  .venv/bin/python src/map_tickers_fallback.py
"""

import time
from pathlib import Path

import pandas as pd

from map_tickers import PAUSE, post

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
BATCH = 10


def main() -> None:
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    U = pd.read_parquet(PROC / "universe.parquet", columns=["cusip"])
    if "fallback" not in m:
        m["fallback"] = False
    todo = sorted(set(U["cusip"]) & set(m.loc[~m["listed_us"].astype(bool), "cusip"]) | (set(U["cusip"]) - set(m["cusip"])))
    print(f"{len(todo)} universe stocks without a US ticker to retry")

    id_type = lambda c: "ID_CINS" if c[0].isalpha() else "ID_CUSIP"
    found = {}
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        anywhere = post([{"idType": id_type(c), "idValue": c} for c in batch])
        time.sleep(PAUSE)
        share_class = {}
        for c, a in zip(batch, anywhere):
            data = a.get("data") or []
            sc = next((d.get("shareClassFIGI") for d in data if d.get("shareClassFIGI")), None)
            if sc:
                share_class[c] = (sc, data[0])
        if share_class:
            jobs = [{"idType": "ID_BB_GLOBAL_SHARE_CLASS_LEVEL", "idValue": sc, "exchCode": "US"} for sc, _ in share_class.values()]
            us = post(jobs)
            time.sleep(PAUSE)
            for (c, (sc, first)), a in zip(share_class.items(), us):
                hit = next((d for d in (a.get("data") or []) if d.get("exchCode") == "US"), None)
                if hit:
                    found[c] = {
                        "ticker": hit.get("ticker"),
                        "name": hit.get("name") or first.get("name"),
                        "security_type": hit.get("securityType") or first.get("securityType"),
                        "figi": hit.get("compositeFIGI"),
                    }
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}  found so far: {len(found)}")

    new_rows = []
    for c, f in found.items():
        if c in set(m["cusip"]):
            idx = m.index[m["cusip"] == c]
            for k, v in f.items():
                m.loc[idx, k] = v
            m.loc[idx, "listed_us"] = True
            m.loc[idx, "fallback"] = True
        else:
            new_rows.append({"cusip": c, "listed_us": True, "fallback": True, **f})
    if new_rows:
        m = pd.concat([m, pd.DataFrame(new_rows)], ignore_index=True)
    m.to_parquet(PROC / "cusip_map.parquet", index=False)
    print(f"\nfound US tickers for {len(found)}/{len(todo)}")
    print(pd.DataFrame(found).T.head(12)[["ticker", "name"]].to_string())


if __name__ == "__main__":
    main()
