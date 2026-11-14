# Step 3: Holdings and BUY / HOLD / SELL labels

**Date:** 2026-11-12 · **Scripts:** `src/build_holdings.py`, `src/build_labels.py`
**Outputs:** `data/processed/holdings.parquet`, `implied_prices.parquet`, `labels.parquet`, `experiments/labels_summary.txt`
**Status:** superseded. The final labels use real split data and the top-500 universe; see [04-universe-and-market-data.md](04-universe-and-market-data.md). This file still explains the labeling rule and why splits are hard.

## What we did

### 3a. Holdings (`build_holdings.py`)

For each of the 54 SEC files, we pulled out our 50 funds' reports (49 hedge funds plus Berkshire) and kept:

- **Only plain shares of stock.** Options (puts and calls) and bonds were dropped.
- **One report per fund per quarter.** Amendments come in two kinds:
  - a "restatement" *replaces* the earlier report, so we keep the latest;
  - "new holdings" *adds* positions left out before, so we keep those too.
- **One row per stock.** A fund can list the same stock on several rows (for example, held for different clients), so we add them up.
- **Dollars in one unit.** Before January 3, 2023 `VALUE` was in thousands of dollars, after that in dollars. We converted everything to dollars.
- **The newest filed copy** when the same quarter appears in two SEC files.

We also computed a **typical price per share** for every stock and quarter: value ÷ shares, across *all* 13F filers, taking the median. The median ignores the occasional typo.

**Result:** 1.46 million rows (fund × quarter × stock), 50 funds, 2013 to mid-2026.

### 3b. Labels (`build_labels.py`)

For each fund, each stock and each pair of back-to-back quarters where the fund filed **both** reports:

| Label | Rule |
|---|---|
| **BUY** | shares rose by more than 1%, including brand-new positions |
| **SELL** | shares fell by more than 1%, including full exits |
| **HOLD** | anything smaller (rounding, tiny rebalances) |

If a fund skipped a quarter's filing, we don't make labels across the gap. A missing report doesn't mean it sold everything. This happened only twice.

**Result:** 1.75 million labels, 2013 Q3 to 2026 Q2: **49% BUY, 47% SELL, 4% HOLD.**

## Why HOLD is so rare

Many of our funds are quant or multi-strategy funds (Renaissance, Two Sigma, Citadel, Millennium) that adjust almost every position every quarter, so true "no change" is rare. That's a real feature of hedge funds, and different from mutual funds. Two choices for the paper to discuss:

- a wider HOLD band, for example ±5%;
- reporting a 2-way BUY/SELL task alongside the 3-way one.

## The stock-split problem (worth understanding well)

When a company does a **4-for-1 split**, every shareholder's share count multiplies by 4 and the price per share falls to about ¼, with no trade at all. Comparing raw share counts would wrongly call that a BUY. So splits must be detected and adjusted for.

We first tried to detect splits **from the holdings data itself**, and learned a lot from it failing:

| Attempt | Rule | What went wrong |
|---|---|---|
| 1 | One fund's share ratio is near a split ratio (2, 3, 1.5…) and price moved within a wide window | 24,000 fake "3-for-2 splits". For a 1.5 ratio, a stock whose price didn't move still fit the window. |
| 2 | Same, with a narrower price window | Still thousands of fakes, from funds adding 50% while the price fell 20–45% |
| 3 | Decide per **stock**, using the typical ratio across all funds holding it | Fakes again (e.g., AMD 2024, CrowdStrike 2020). Many funds **rebalance to a constant dollar amount**: when a price rises 50% they trim a third of their shares, and that looks exactly like a split. |
| 4 | Require 70% of holders at *exactly* the same ratio | Fakes gone, but real splits were missed too. Funds also trade in the split quarter, so NVIDIA's 2024 10-for-1 showed a typical ratio of 9.2, not exactly 10. |
| 5 | Require the share signal and the price signal to *agree*, choosing the ratio that best fits both | Catches every famous split (Amazon, Google, NVIDIA, Apple, Tesla) and leaves AMD and CrowdStrike alone. But the **ratio** is sometimes wrong (NVIDIA 8 instead of 10, Tesla 3 instead of 5), and there are too many 2-for-1 flags. |

**Lesson:** holdings data can't reliably tell a split apart from heavy trading or rebalancing in the same quarter. The proper fix is **real split history** (date and ratio) from a market data source. We get that in the next step, when prices are downloaded by ticker. The labels will then be rebuilt with exact split adjustments.

**Known test cases** for checking the final version:

| Stock | Split | Quarter |
|---|---|---|
| Apple | 4-for-1 | Q3 2020 |
| Tesla | 5-for-1 | Q3 2020 |
| Tesla | 3-for-1 | Q3 2022 |
| NVIDIA | 4-for-1 | Q3 2021 |
| NVIDIA | 10-for-1 | Q2 2024 |
| Amazon | 20-for-1 | Q2 2022 |
| Google | 20-for-1 | Q3 2022 |
| AMD | none | Q1 2024 (no split: must stay unadjusted) |

## Other known simplifications (to revisit)

- **All common stocks are included for now.** We restrict to the S&P 500 once stocks are mapped to tickers.
- **When a company changes its CUSIP** (some mergers, reverse splits), the old ID shows as SELL and the new one as BUY. Ticker mapping will catch these.
- **A restatement filed later** replaces holdings stock by stock, so a position the restatement *removed* could linger. This is rare and minor.

## Check yourself

1. Why do we drop options and bonds from the holdings?
2. What's the difference between a "restatement" amendment and a "new holdings" amendment?
3. Why do we compute a "typical price" from *all* 13F filers, not just our 50 funds?
4. Explain why a 4-for-1 split would look like a BUY if we didn't adjust for it.
5. Why does a fund that rebalances to a constant dollar amount look like it's doing stock splits?
6. Why is HOLD so rare for hedge funds, and what could we do about it in the paper?
