# Step 7: RQ3, do the trades nobody can predict earn more? (rough version)

**Date:** 2026-11-13 · **Script:** `src/rq3_returns.py`
**Outputs:** `experiments/rq3_trade_level.csv`, `rq3_stock_level.csv`, `fig_rq3.png`

## The idea

Mimicking Finance found that mutual fund managers' *unpredictable* trades are where their skill shows up. We test the same for hedge funds:
- Using LightGBM's test-period predictions (2022 to early 2026, 17 quarters), we call a trade **"predicted"** if the model got it right and a **"surprise"** if it didn't.
- Then we ask: which trades were followed by better stock returns?

## Two return windows

| Window | Who could trade on it |
|---|---|
| **Next quarter:** the 3 months right after quarter-end | Only the fund. The 13F filing isn't public yet. |
| **After filing:** 3 months starting 45 days after quarter-end | Anyone. The 13F is public by then. |

Returns are **market-adjusted:** each stock's return minus the average universe stock's return in the same window. Phase 2 replaces this with Fama-French risk adjustment.

> **⚠️ Superseded on 2026-11-13.** The numbers below had a lookahead bug. A stock only kept its next-quarter return if it was *still in the top 500 next quarter*, which you can only know afterwards. Fixing it removes most of the effect. See [08-rq3-factors.md](08-rq3-factors.md). The text below is kept as a record of the mistake.

## Tests and results (rough, buggy: see the warning above)

**1. Within each action (the fair comparison), next quarter:**

| | Surprise minus predicted | t-stat |
|---|---|---|
| Stocks funds **bought** | **+0.54%** (surprise buys rose more) | 1.86 |
| Stocks funds **sold** | **−0.63%** (surprise sells fell more, so those sells were well-timed) | −1.90 |

Both point the same way as Mimicking Finance: **the trades our model couldn't predict were the better-timed ones.**

**2. All trades pooled together (misleading, kept for honesty):** predicted trades look better (+0.40% vs +0.13%). That's a **mix effect**. The model mostly predicts SELL, so "predicted" and "surprise" contain very different shares of buys and sells. Always compare like with like.

**3. Stock level:** each quarter, stocks are ranked into fifths by how predictable their holders' trades were.
- **Next quarter:** least predictable minus most predictable = **+0.79% a quarter** (t = 1.95). That's similar in size to Mimicking Finance's ~1% a quarter (4.24% a year).
- **After the filing is public:** **+0.23%** (t = 0.46, not significant). Once the 13F is out, the edge is gone, which is what an efficient market predicts.

## How sure are we?

**Not very yet.** With 17 quarters, t-statistics around 1.9 are *suggestive*. The usual bar is about 2.0 for "significant". Phase 2:
1. Fama-French 5-factor + momentum alphas instead of "minus the average stock".
2. Robustness checks: other windows, value-weighting, dropping the biggest funds.
3. Using Chronos-2 and hybrid predictions as well as LightGBM.

## Check yourself

1. What's the difference between a "predicted" and a "surprise" trade?
2. Why do we look at returns *after* the 13F filing separately?
3. Explain the "mix effect": why can pooling buys and sells give the wrong answer?
4. Why is a t-statistic of 1.9 from 17 quarters not yet convincing?
5. If surprise sells are followed by bigger drops, what does that say about the fund?
