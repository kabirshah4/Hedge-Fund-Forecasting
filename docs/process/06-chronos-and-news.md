# Step 6: Chronos-2 (zero-shot) and news, RQ2

**Date:** 2026-11-13
**Scripts:** `src/run_chronos.py`, `src/chronos_scores_all.py`, `src/prepare_news.py`, `src/score_news.py`, `src/rq2_models.py`
**Outputs:** `experiments/rq2_chronos.csv` (sets A, B, D in `rq2_chronos_ABD.csv`), `experiments/rq2_lightgbm.csv`, `data/processed/news_scores/`

## Part 1: Chronos-2 on its own (zero-shot)

**What it is:** Chronos-2 is a *time-series foundation model*, pre-trained on huge numbers of number series. "Zero-shot" means we **never train it on our data**. We hand it a series and it forecasts the next value.

**What we gave it:** for each position, the fund's share count over the quarters it has held the stock (up to 32 quarters), adjusted for splits and scaled so the last quarter = 1.0. It returns 9 forecast quantiles for next quarter, from pessimistic to optimistic.

**Turning the forecast into BUY/HOLD/SELL:** from those quantiles we estimate P(up more than 1%) and P(down more than 1%). The score P(up) − P(down) becomes a label with two cut-offs. The cut-offs are the **only** thing tuned, using 2021 (validation) only.

**Results (test 2022 to mid-2026):**

| Input set | Accuracy | Macro-F1 | Mean per-fund accuracy |
|---|---|---|---|
| A: own share history | 57.8% | 0.30 | 55.6% |
| B: + stock returns | 57.9% | 0.30 | 55.5% |
| D: + returns + policy uncertainty | 58.1% | 0.30 | 55.6% |
| *for comparison:* majority baseline | 56.0% | 0.34 | 56.7% |
| *for comparison:* LightGBM | 62.3% | 0.52 | 61.0% |

**What it means:**

- **Zero-shot Chronos-2 is no better than always guessing a fund's usual action.** Its cut-offs came out equal, so it never predicts HOLD, and its low macro-F1 shows it mostly predicts one class.
- **Extra inputs (stock returns, policy uncertainty) add almost nothing.**
- **This matches the research.** Rahimikia, Ni & Wang found that off-the-shelf time-series foundation models forecast stock returns poorly without training on financial data. Share-count histories are short (median 14 quarters), jumpy and driven by decisions, not smooth patterns, which is hard for a model pre-trained on things like weather or sales.
- **It's a legitimate finding, not a failure:** "foundation models don't help zero-shot here" is useful to know. Next we test whether its forecast helps *as one input* to LightGBM (the hybrid).

## Part 2: News

- **Headlines:** 1.4 million from FNSPID for 676 universe stocks, 2013–2023, scored with **FinBERT**. Sentiment = P(positive) − P(negative), from −1 to +1.
- **Leakage filter:** headlines that report an investor's stake or filing are removed, 0.7% of all headlines. The filter's sample caught *our own funds' trades in the news*: "Point72 Lowers Tyson Food Stake", "Tiger Global Cuts Stakes In Facebook, Netflix", "13F From David Tepper's Appaloosa". Without the filter, the model could read the answers.
- **Two news settings:**
  - **Forecast:** news from quarter *t−1*, a prediction made at the start of the quarter.
  - **Nowcast:** news during quarter *t*, a prediction made at quarter-end, before the 13F reveals the trades 45 days later.
- **Scored on 2022–2023 only,** the test years with news (your decision).

## Part 3: RQ2 results, LightGBM with and without each addition

All five versions use the same split. The base model is the RQ1 LightGBM. "Change" is the difference in mean per-fund accuracy versus base; the 95% interval comes from redrawing whole funds 1,000 times.

| Added to LightGBM | Scored on | Change (points) | 95% interval |
|---|---|---|---|
| Chronos-2 forecast (the hybrid) | 2022–2026 | +0.14 | −0.44 to +0.90 |
| Chronos-2 forecast | 2022–2023 | −0.09 | −0.58 to +0.39 |
| News, forecast (quarter *t−1*) | 2022–2023 | +0.04 | −0.50 to +0.56 |
| News, nowcast (quarter *t*) | 2022–2023 | −0.28 | −0.83 to +0.22 |
| Everything | 2022–2023 | +0.06 | −0.44 to +0.55 |

62% of test positions in 2022–2023 had at least one headline that quarter.

## What RQ2 tells us

**Neither the foundation model nor the news adds any predictive power.** Every interval includes zero. The nowcast is especially telling: even news from the *same* quarter the fund traded doesn't help.

Two readings, and the paper should present both honestly:

1. **The skill story.** Hedge funds' trading isn't a reaction to public news, so what drives it isn't in public information. That fits RQ1 (hedge funds are hard to predict) and RQ3 (their unpredictable trades are the better-timed ones).
2. **The measurement story.** Our news features are coarse: a quarter's headlines averaged into one sentiment number. Richer signals could matter, such as an LLM reading the news, specific events like earnings surprises, or timing within the quarter. That's what Phase 2/3 tests next, so we must not over-claim "news never matters".

## Check yourself (Part 3)

5. Why is a confidence interval that includes zero important here?
6. Give the two readings of the "news adds nothing" result. Which extra experiment would help tell them apart?

## Check yourself

1. What does "zero-shot" mean, and why is it both an advantage and a weakness here?
2. Why might share-count histories be hard for a model pre-trained on general time series?
3. What's the difference between the "forecast" and "nowcast" news settings? Which is more useful to an outsider, and which is a fairer test of the model?
4. Give an example of a headline the leakage filter must remove, and explain why.
