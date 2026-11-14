"""Download the SEC's Form N-PORT data sets (fund-level holdings of registered
funds, quarterly files since 2019Q4) into data/raw/nport/.

Why: the second blind review asked for a fund-level anchor. N-PORT gives
holdings for individual mutual funds (series), so we can run the identical
pipeline at the fund level and again after adding funds up to their parent,
which separates "aggregation" from "our pipeline" as the reason predictability
is low.

Resumable (skips finished files) and rate-limited via edgar.py; requires the
EDGAR_IDENTITY environment variable like download_13f.py.

Usage: .venv/bin/python src/download_nport.py
"""

import re
from pathlib import Path

import edgar

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "nport"
PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = edgar.session()
    links = sorted(set(re.findall(r'href="([^"]*nport[^"]*\.zip)"', edgar.get(s, PAGE).text, re.I)))
    for i, u in enumerate(links, 1):
        url = u if u.startswith("http") else "https://www.sec.gov" + u
        dest = OUT / url.rsplit("/", 1)[-1]
        if dest.exists() and dest.stat().st_size > 1_000_000:
            continue
        tmp = dest.with_suffix(".part")
        with edgar.get(s, url, stream=True) as r, open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        print(f"[{i}/{len(links)}] {dest.name} {dest.stat().st_size / 1e9:.2f} GB", flush=True)
    print("done")


if __name__ == "__main__":
    main()
