# Step 11: a rule-based fund list from SEC Form ADV

**Phase 2. Script:** [src/build_fund_list_adv.py](../../src/build_fund_list_adv.py)
**Outputs:** `data/fund_list_adv.csv` (100 funds), `data/processed/adv_hedge_advisers.csv` (every adviser that passed the rules, with how it was matched)

## Why

Our first list was hand-picked in 2026 from funds we'd heard of. That's **hindsight bias**: famous funds are mostly the ones that *survived and did well*. A reviewer would say "you only studied winners." The leakage audit (step 9) flagged this as the biggest open risk.

## What Form ADV is

Every investment adviser registered with the SEC files **Form ADV**, at least once a year. Its Schedule D section 7.B.(1) lists each **private fund** the adviser runs, with:
- the fund's **type**, which the adviser ticks itself: hedge fund, private equity, real estate…
- its **gross asset value (GAV)**: everything the fund holds, before subtracting borrowing.

## The rules (all applied to data from July 2012 to June 2013, before our sample starts)

1. Take each adviser's latest ADV filed in that window.
2. Add up the GAV of its funds marked "Hedge Fund". Skip **feeder funds**, which just pass money into a "master" fund that's already counted, and **funds of funds**, which hold other managers' funds.
3. Keep an adviser if:
   - its hedge funds hold at least $1 billion;
   - they're at least 50% of its private funds;
   - they're at least 25% of its total assets under management.

   The last rule drops giants like PIMCO and Wellington, which run some hedge funds but are mainly traditional managers. **437 advisers pass.**
4. **Link each adviser to its 13F filings.**
   - First, by the SEC ID (CIK) it reports in ADV Item 1.D.3. This item only exists from 2017, but a CIK is just an ID number and says nothing about performance.
   - Otherwise, by exact legal name. We clean up EDGAR quirks first: "BAUPOST GROUP LLC/MA", "TUDOR INVESTMENT CORP ET AL", "S.A.C." vs "SAC".
   - Otherwise, by a looser name match that must be unique and at least two words long.
   - The filer must have reported at least 10 holdings for June 2013. **269 advisers link.**
5. Keep the **100 largest by hedge fund GAV**: $4.6B to $198B each, $1.7 trillion in total.

## What we found along the way

- **Loose name matching is dangerous.** "Pacific Investment Management" (PIMCO) first matched an unrelated "Pacific Global Investment Management". Fix: only exact matches after removing legal words, plus a strict fallback.
- **The list contains funds that later shut down or changed names:** Eton Park, Claren Road, Highbridge, SAC Capital (which became Point72). That's what we want. Under a rule set in 2013, a fund that dies simply stops having data, so we can't cherry-pick survivors.
- **Only 25 of the 100 overlap with our hand-picked list.** Many big hedge funds are credit or macro funds with few U.S. stocks, which is why we require 10+ holdings.
- **Big names missing for honest reasons:**
  - Bridgewater fails the 25% rule (mostly separate accounts).
  - Adage files its 13F under a different entity name.
  - Point72, D1 and others didn't exist yet in 2013.

## Rerunning everything on the ADV sample

The scripts now take a switch: `SAMPLE=adv` makes them read `data/fund_list_adv.csv` and write to `data/processed_adv/` and `experiments_adv/`. Files that don't depend on which funds we study (stock universe, prices, ticker map) are shared, so both samples use exactly the same stocks.

```
cd src; export SAMPLE=adv
../.venv/bin/python build_holdings.py && ../.venv/bin/python build_labels.py && ../.venv/bin/python build_features.py
../.venv/bin/python baselines.py && ../.venv/bin/python rq1_bootstrap.py && ../.venv/bin/python robustness.py
../.venv/bin/python rq3_returns.py && ../.venv/bin/python rq3_factors.py
```

## Results

| | Hand-picked (49) | Form ADV (99) |
|---|---|---|
| Majority baseline | 56.7% | 59.4% |
| LightGBM | 61.0% | 62.9% |
| **Gain [95% CI]** | **+4.3 [2.0, 7.0]** | **+3.5 [1.6, 5.6]** |
| Their protocol, per-fund model | −1.1 | −1.1 |
| Their protocol, pooled | +2.5 | +2.4 |
| BUY vs SELL only | +2.8 | +0.7 |
| RQ3: FF5 + momentum alphas positive (of 24) | 24 | 13 |
| RQ3: alphas with t ≥ 1.96 | 4 | 0 |

- **RQ1 replicates almost exactly.** Hedge funds are hard to predict, whichever funds you pick.
- **RQ3 doesn't.** The faint "unpredicted trades earn a bit more" tilt in the hand-picked sample disappears completely in the rule-based sample. The likely reason is that **we picked famous, successful funds with hindsight**, which is exactly the bias this step was built to catch.

So the paper's RQ3 conclusion is now: *in an unbiased sample, hedge funds' unpredicted trades earn nothing extra.*

## Check yourself

1. Why does choosing funds with 2013 data avoid survivorship bias, while choosing them in 2026 doesn't?
2. Why skip feeder funds when adding up GAV?
3. Give one reason a big hedge fund might not appear in our ADV list even though it's real.
