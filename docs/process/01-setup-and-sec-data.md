# Step 1: Project setup and SEC 13F data

**Date:** 2026-11-12 · **Scripts:** `src/edgar.py`, `src/download_13f.py`

## What we did

1. Created the project folders: `data/` (data), `src/` (code), `experiments/` (results), `paper/` (the paper), `docs/` (plans and notes).
2. Set up a Python environment (Python 3.12, managed by `uv`) with the libraries we need.
3. Downloaded every SEC Form 13F data set: **54 zip files, 3.1 GB**, covering holdings reports from mid-2013 to June 2026. All 54 passed an integrity check.

## Why

Our research question is about predicting hedge fund trades. The only free, official record of what hedge funds own is **Form 13F**:

- Every investment manager with over $100 million in US stocks must file one each quarter.
- It lists every US stock they hold at quarter end: the stock's ID (CUSIP), number of shares and dollar value.
- It's due **45 days after the quarter ends**. That delay matters a lot for avoiding "leakage" later.

The SEC publishes all 13F filings as clean tables ("Form 13F Data Sets"), so we don't have to read thousands of individual filings.

## How

- `src/edgar.py` makes polite requests to the SEC:
  - **It identifies us.** The SEC requires a name and email on automated requests, read from the `EDGAR_IDENTITY` setting so it never goes into the code.
  - **It waits between requests** to stay under the SEC's limit of 10 per second.
  - **It retries** if blocked.
- `src/download_13f.py` reads the SEC's data page, finds every 13F zip file and downloads the ones we don't have yet, so it's safe to re-run.

## What each zip contains

| Table | One row per | Key columns |
|---|---|---|
| `SUBMISSION.tsv` | filing | who filed (CIK), which quarter (PERIODOFREPORT), when filed (FILING_DATE) |
| `COVERPAGE.tsv` | filing | manager name, whether it's an amendment |
| `INFOTABLE.tsv` | holding | stock ID (CUSIP), shares, value, whether it's an option |
| `SUMMARYPAGE.tsv` | filing | total value reported |

`ACCESSION_NUMBER` is the filing's unique ID and links the tables together.

## Things that can go wrong (and how we handle them)

- **Units changed in 2023.** Before January 3, 2023, `VALUE` was in **thousands** of dollars; after that, in **dollars**. Handled in Step 3.
- **Amendments.** Funds sometimes re-file a quarter, either replacing it (a "restatement") or adding positions left out before ("new holdings"). Handled in Step 3.
- **13F shows only long US stock positions,** with no short sales and no foreign stocks. That's a limitation to state in the paper.

## Check yourself

1. Who has to file a Form 13F, and how often?
2. Why does the 45-day filing delay matter for a prediction study?
3. What does `ACCESSION_NUMBER` do?
4. Name one thing hedge funds do that 13F filings don't show.
