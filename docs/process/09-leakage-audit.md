# Step 9: leakage audit

**Phase 2. Script:** [src/leakage_audit.py](../../src/leakage_audit.py) **Output:** `experiments/leakage_audit.txt`

## What "leakage" means

**Leakage** means using information you wouldn't have had at the time you made the prediction. It makes results look better than they really are. We already found one case in RQ3 (step 8), so we checked the rest of the pipeline systematically.

## Test 1: the future-scramble test

**Idea:** if a feature for quarter T only uses data from before T, then *destroying* all data from T onward can't change it.

**How:**
1. Pick a quarter T.
2. Replace everything dated T or later with random noise: fund holdings, fund actions, daily prices, 13F-implied prices, universe ranks and the EPU index.
3. Rebuild all the features.
4. Compare the features for quarter T with the real ones.

**Control:** the features for T+1 *should* change, since they're allowed to use quarter T. If they didn't change, the test wasn't really scrambling anything.

| Quarter T | Rows | Result | Control (T+1 features changed) |
|---|---|---|---|
| 2022Q1 | 5,162 | PASS, 0 of 19 features changed | 13 of 19 changed ✓ |
| 2024Q2 | 5,409 | PASS | 13 changed ✓ |
| 2026Q1 | 5,528 | PASS | 13 changed ✓ |

(The other 6 features are things like fund ID or holdings counts that don't depend on quarter T at all.)

## Test 2: were the filings public in time?

A feature for quarter t uses the fund's 13F for t−1. That filing has to be public before we predict, i.e. before quarter t's own 13F is due (about 135 days after t−1 ends). We also keep a fund's *newest* version of each filing, so an amendment filed much later would be a "data vintage" leak.

- **Test-period inputs:** 84.9% filed within 45 days, 15.0% within 46–135 days, **0.1% late** (1 fund-quarter).
- **All years:** 0.5% late, mostly Citadel, Two Sigma and Viking amendments. Too small to matter.

## Checked by reading the code

- **Training:** 2021 is used only for early stopping and for tuning Chronos's cut-offs. The test years are never used for any choice.
- **Chronos inputs:** cut at quarter t−1 by a `searchsorted` on quarter (line 106 of run_chronos.py).
- **News forecast:** quarter t−1 news, shifted one quarter. The news *nowcast* uses quarter t on purpose, with stake/13F headlines removed.

## Leaks still open (in the paper as limitations)

1. **The fund list was chosen in 2026.** We picked well-known funds that survived, which is hindsight. The fix is the Form ADV fund list (Thanksgiving, task tg5).
2. **Pretrained models may have seen the future.** Chronos-2 and the LLM were trained on recent data. That's unlikely to matter for share-count histories, but it's a real risk for the LLM reading news, which is why the memorization check is on the plan.
3. **Delisted stocks.** About 5% of positions have no Yahoo price, so RQ3 can't include their (often bad) final returns.

## Check yourself

1. Why does the control (T+1 *should* change) matter? What would a PASS mean without it?
2. Why can keeping the newest amendment of a filing be a leak?
3. Which open leak worries you most, and why?
