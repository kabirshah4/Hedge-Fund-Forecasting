# Step 8: RQ3 again, with factor alphas (and a bug found)

**Phase 2. Script:** [src/rq3_factors.py](../../src/rq3_factors.py) (plus the fix in [src/rq3_returns.py](../../src/rq3_returns.py))
**Outputs:** `experiments/rq3_factor_alphas.csv`, `experiments/rq3_monthly_spreads.csv`, `paper/figures/fig_rq3_alphas.pdf`

## What we did

1. **Calendar-time portfolios.** At each quarter-end we make six baskets of stocks: surprise buys, predicted buys, surprise sells, predicted sells, the least-predictable fifth of stocks (Q1) and the most-predictable fifth (Q5). We track each basket's **monthly** return over the next 3 months. That gives 51 months of data instead of 17 quarters.
2. **Spreads.** Long one basket, short another:
   - **Buy spread:** surprise buys − predicted buys.
   - **Sell spread:** predicted sells − surprise sells. Positive means surprise sells fell further.
   - **Q1 − Q5.**
3. **Factor regressions.** We regress each spread's monthly returns on known return drivers:
   - **CAPM:** the market.
   - **FF3:** + size (SMB) and value (HML).
   - **FF5 + momentum:** + profitability (RMW), investment (CMA) and momentum (Mom).

   The **intercept (alpha)** is the return those factors *can't* explain. If a spread's edge comes only from, say, holding small or trending stocks, alpha goes to ~0.
4. **Newey-West t-stats** handle the fact that one month's returns can be correlated with the next.

## The bug we found

The new monthly numbers didn't match Phase 1 (Q1 − Q5: −0.05%/month vs +0.79%/quarter). Tracking down why exposed a **lookahead bug** in the Phase 1 code. A stock's next-quarter return was only kept if the stock was **still in the top 500 next quarter**.

- That's information from the future: about 12% of stocks drop out each quarter, often because they crashed.
- **Fix:** every stock in the universe at quarter-end t keeps its t+1 return, wherever it ends up. We also now use only Yahoo prices for returns, because 13F-estimated prices for untickered stocks were too noisy.

## Results (next quarter, before the filing is public)

| Spread | Raw (per month) | After FF5 + momentum |
|---|---|---|
| Sells: surprise vs predicted | **+0.17% (t = 2.33)** | +0.08% (t = 1.15) |
| Buys: surprise vs predicted | +0.06% (t = 0.75) | +0.08% (t = 0.98) |
| Stocks: least vs most predictable | −0.05% (t = −0.28) | +0.14% (t = 0.85) |

- **Only sells show a raw edge, and the factors explain most of it.** The sell spread loads on the market (t = 4.5) and momentum (t = 3.9).
- **After the filing is public:** nothing is significant.
- **Quarterly check (fixed code):** buys +0.25% (t = 0.87), sells −0.56% (t = −1.89), Q1 − Q5 +0.17% (t = 0.29).

## Robustness: value weighting and longer holding periods

`experiments/rq3_robustness.csv` holds 24 versions: 2 weightings × 4 holding periods × 3 spreads.
- **Value-weighted:** big stocks count more. Weight = institutional value at quarter-end t.
- **Longer holds:** 6 and 12 months. Cohorts formed in different quarters overlap, so we average them month by month.

**Result:** **all 24** factor-adjusted alphas are **positive**, but only **4** have |t| ≥ 1.96:
- equal-weighted, 6 months, Q1−Q5: t = 2.00
- equal-weighted, 12 months, buys: t = 2.19
- equal-weighted, 12 months, Q1−Q5: t = 1.97
- value-weighted, 3 months, sells: t = 2.23

**Careful:** the 24 aren't independent. They use the same months and overlapping portfolios, so "24 of 24 positive" is weaker than it sounds. The honest reading: *the edge always points the same way as for mutual funds, but it's small and 51 months can't confirm it.*

## What it means for the paper

The RQ3 story changed. **Unlike mutual funds, hedge funds' unpredicted trades do not reliably earn more.** That's still a finding, and arguably a cleaner one: hedge funds are hard to predict, *and* their unpredictability doesn't show up as visible skill in 13F data. Possible reasons: trades driven by risk management, redemptions or hidden hedges (shorts and options aren't in 13F), or 51 months being too few to detect a small edge.

**Lesson:** a result that matches what you hoped for still needs checking. The bug was caught only because two methods disagreed.

## Check yourself

1. Why does requiring "still in the top 500 next quarter" bias returns?
2. What does a factor alpha measure, and why might a raw return difference shrink after adding factors?
3. Why do monthly portfolios give more statistical power than quarterly averages?
