# Step 14: answering a mock review (five critiques)

You pasted five likely reviewer critiques. Here's what we did about each, and why. Scripts: [src/rq1_metrics.py](../../src/rq1_metrics.py), [src/rq3_by_style.py](../../src/rq3_by_style.py), and changes in [src/rq2_models.py](../../src/rq2_models.py).

## 1. "Apples to oranges": 4 points vs 19 isn't comparable

**The critique's premise was partly wrong.** It said mutual funds have "a massive HOLD class". But Mimicking Finance's majority baseline is **52%**, which means no class dominates their data. If HOLD dominated, guessing the usual action would score much higher. Our baseline is 56.7%, so their problem is, if anything, *more* balanced than ours. Always check a critique's facts before acting on it.

**The fair part:** accuracy gains do depend on the class mix. So we added measures that don't:

| | Mutual funds | Hand-picked | Form ADV |
|---|---|---|---|
| Share of baseline errors fixed | 40% | **10%** | **9%** |
| Log-loss skill vs each fund's class frequencies | n/a | 4.6% | 4.4% |
| AUC, BUY vs SELL | n/a | 0.66 | 0.67 |

- **Error reduction** = (model − baseline) / (100% − baseline): the share of the naive guess's mistakes the model fixes.
- **Log-loss** scores the predicted *probabilities*, not just the final guess.
- **AUC** = the chance the model ranks a random BUY above a random SELL. 0.5 is a coin flip.

The reviewer also wanted an LSTM. Our "their protocol" check (step 10) already copies their per-fund rolling setup. An LSTM remains a possible extra.

## 2. New positions are missing

**True, and on purpose.** Like Mimicking Finance, we predict only positions held at t−1. New positions are 14% of labels. Predicting *which* of 500 stocks a fund will start buying is a different and much more lopsided task. The paper now says so in Method and Limitations, and calls it future work.

## 3. Pooling very different funds

**Partly right.** Multi-strategy and quant funds supply **79% of test positions**. That's why pooled accuracy (62.3%) is above mean per-fund accuracy (61.0%), and why we report the per-fund average throughout. We already showed per-fund models do *worse* (step 10). New: a **by-style table** (paper Table "Results by fund style"):

| Style | Funds | RQ1 gain | RQ3 sells α (t) | RQ3 buys α (t) |
|---|---|---|---|---|
| Long/short equity, growth | 21 | +2.1 | −0.11 (−0.3) | −0.09 (−0.2) |
| Activist / event-driven | 13 | +7.0 | +0.47 (1.3) | **+2.33 (2.3)** |
| Multi-strategy | 6 | +7.8 | **+0.22 (2.8)** | +0.10 (0.9) |
| Quantitative | 5 | +5.0 | −0.08 (−0.7) | −0.03 (−0.4) |
| Macro / value | 3 | −1.9 | +0.39 (0.7) | +0.64 (1.3) |

Two of ten RQ3 spreads are significant. That's more than chance alone would give (about 0.5), but with few funds per style it's a lead, not a finding.

## 4. 51 months is short

**Fair.** We reworded RQ3 everywhere. The claim isn't "hedge funds have no skill". It's "any skill in their unpredicted 13F trades is **statistically indistinguishable from noise** over this 4-year window." The paper also names the period's quirks (rate hikes, mega-cap tech) and adds statistical power as a limitation.

## 5. Quarterly-averaged FinBERT is too blunt

**Fair.** We now also test news from only the **last 15 days** of the quarter. Results go into RQ2 (see decisions.md). The LLM experiment is the stronger version of this test.

## Lesson

A good critique tells you where a reader will push. Answer the fair parts with evidence, and check the facts in the critique itself (critique 1's premise was wrong).
