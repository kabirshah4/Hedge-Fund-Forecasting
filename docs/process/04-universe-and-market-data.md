# Step 4: Stock universe, tickers, prices and real splits

**Date:** 2026-11-12/13
**Scripts:** `src/map_tickers.py`, `src/map_tickers_fallback.py`, `src/build_universe.py`, `src/get_market_data.py`, `src/build_labels.py` (final version)
**Outputs:** `data/processed/cusip_map.parquet`, `universe.parquet`, `prices_daily.parquet`, `splits.parquet`, final `labels.parquet`

## What we did

1. **Mapped stock IDs to tickers.** 13F filings name stocks by CUSIP; prices are looked up by ticker (AAPL). We used **OpenFIGI**, a free Bloomberg service, for the 4,400 stocks held by 300 or more filers.
2. **Chose the stock universe:** each quarter's **500 largest stocks by institutional holdings value**, built from the SEC data. This was your decision, option A over the S&P 500.
3. **Downloaded daily prices and real split history** from Yahoo Finance: 8.8 million rows, 3,050 tickers, 919 splits.
4. **Rebuilt the labels** with exact split adjustments and the universe filter.

**Final labels: 291,053**, for 50 funds and 500 stocks a quarter, Q3 2013 to Q2 2026: **48.8% SELL, 48.5% BUY, 2.7% HOLD.**

## Why not the S&P 500?

Free data can't reliably link the S&P's *old* tickers to the IDs in SEC filings:
- **Renames:** BNY Mellon was BK, now BNY.
- **Acquisitions:** Aetna, Express Scripts.
- **Retired IDs:** GE's ID changed after its 2021 reverse split.

10–28% of early S&P members would silently disappear. Studying only the companies that survived to today biases results (**survivorship bias**), and reviewers look for it.

Our universe comes from the SEC data itself, so acquired and delisted companies stay in. **Membership for quarter *t* uses the ranking at the end of *t−1*,** so nothing about the future decides which stocks are in the study.

## Things that went wrong, and the fixes (good to understand)

| Problem | How we noticed | Fix |
|---|---|---|
| OpenFIGI returned a foreign listing (Electronic Arts as its Mexican "EA*") | Big US companies had odd tickers | Ask for the US listing first |
| Companies registered abroad (Accenture, ID G1151C101) weren't found | Missing big names | IDs starting with a letter are "CINS" codes, a different lookup type |
| Some stocks only had foreign listings in OpenFIGI (Exxon as Chilean "EXMOC") | Exxon had no ticker | Fallback through the share-class ID, which is the same in every country. Found 72 more; Exxon is still missing. |
| Google at "$263 trillion" | Impossible values | Some filers report dollars where the SEC expects thousands. Rank by shares × *typical* price instead of summing reported values |
| Berkshire Class A at "$2.7 trillion", Micron at #4 | Implausible ranking | Some filers typed dollars into the *shares* column. Drop any filer whose own price is more than 5× off the typical price |
| Bonds in the universe (ID 43713WAB3) | Odd IDs | A CUSIP's characters 7–8 are digits for stocks and contain letters for bonds |
| Mutual funds in the universe (Invesco Convertible, American Funds) | The fallback named them | Real companies have hundreds of holders and these had **one**. Require 100+ holders. |
| Acquired companies (old Google, EMC, Express Scripts) being dropped | Checked what "unknown type" really was | Keep unknown-type stocks, and exclude only *known* non-companies |

**The lesson:** every data source has errors. You find them by checking results against things you know, like "Apple can't be worth $263 trillion" or "Exxon obviously has a ticker".

## Stock splits, finally done right

For stocks with a ticker (85% of the universe), the split factor is the **product of every real split recorded by Yahoo** between the two quarter-ends. The remaining 15% (mostly delisted) use the holdings-based heuristic from Step 3.

**Test results: 9/9 pass.**
- Apple 4-for-1 (2020)
- Tesla 5-for-1 (2020) and 3-for-1 (2022)
- NVIDIA 4-for-1 (2021) and 10-for-1 (2024)
- Amazon 20-for-1 (2022)
- Google 20-for-1 (2022)
- No split for AMD (2024) or CrowdStrike (2020), correctly left alone

## Known limitations (to state in the paper)

- **About 15% of universe stocks have no US ticker,** mostly delisted, plus a few gaps like Exxon. They still have labels and 13F typical prices, but less precise split handling and no Yahoo daily prices.
- **Yahoo's free data** isn't an official research database. A stronger version would use CRSP through a university.
- **HOLD is only 2.7%,** because hedge funds adjust almost every position. Mimicking Finance uses the same ±1% band; we may also report a 2-way BUY/SELL version.

## Check yourself

1. What is survivorship bias, and why would using today's S&P 500 list cause it?
2. Why does the universe for quarter *t* come from quarter *t−1*?
3. How did we tell real companies apart from mutual-fund shares that slipped into the universe?
4. Why do we multiply shares by the *typical* price instead of adding up reported values?
5. Our labels kept only 17% of all trades. Why? (Hint: what else do hedge funds trade?)
