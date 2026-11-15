"""Download the SEC's official Form 13F data sets.

Source: https://www.sec.gov/dera/data/form-13f
Each zip holds tab-separated tables, the main ones being:
  SUBMISSION.tsv  one row per filing (who filed, which quarter, when)
  COVERPAGE.tsv   filer name and whether the filing is an amendment
  INFOTABLE.tsv   one row per holding (stock CUSIP, number of shares, value)

Files already downloaded are skipped, so it is safe to re-run.

Usage:
    .venv/bin/python src/download_13f.py            # list what would download
    .venv/bin/python src/download_13f.py --go       # actually download
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from tqdm import tqdm

from edgar import get, session

LISTING_URL = "https://www.sec.gov/dera/data/form-13f"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "13f"


def find_zip_links(s) -> list[str]:
    html = get(s, LISTING_URL).text
    soup = BeautifulSoup(html, "html.parser")
    links = {
        urljoin(LISTING_URL, a["href"])
        for a in soup.find_all("a", href=True)
        if re.search(r"form13f.*\.zip$", a["href"], re.IGNORECASE)
    }
    return sorted(links)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true", help="download (default: list only)")
    args = parser.parse_args()

    s = session()
    links = find_zip_links(s)
    print(f"Found {len(links)} 13F data set files on the SEC page.")
    for url in links:
        print("  ", url.rsplit("/", 1)[-1])
    if not args.go:
        print("\nDry run. Re-run with --go to download.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for url in tqdm(links, desc="Downloading"):
        dest = OUT_DIR / url.rsplit("/", 1)[-1]
        if dest.exists() and dest.stat().st_size > 0:
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(get(s, url).content)
        tmp.rename(dest)
    total_gb = sum(p.stat().st_size for p in OUT_DIR.glob("*.zip")) / 1e9
    print(f"Done. {len(list(OUT_DIR.glob('*.zip')))} files, {total_gb:.1f} GB in {OUT_DIR}")


if __name__ == "__main__":
    main()
