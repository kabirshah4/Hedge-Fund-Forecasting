# Step 5: Features and baselines, the first RQ1 result

**Date:** 2026-11-13 · **Scripts:** `src/build_features.py`, `src/baselines.py`
**Outputs:** `data/processed/features.parquet`, `experiments/rq1_baselines.csv`, `rq1_per_fund.csv`, `rq1_feature_importance.csv`

## The prediction task

For every stock a fund **held at the end of quarter *t−1***, predict what it does by the end of quarter *t*: **BUY more, HOLD or SELL** (±1% rule). This is the same task as Mimicking Finance ("positions with continuous presence"), so the numbers compare.

**249,276 rows**, split by time and never shuffled:

| Part | Quarters | Rows | Used for |
|---|---|---|---|
| Train | 2013 Q3 – 2020 Q4 | 130,352 | fitting the models |
| Validation | 2021 | 21,398 | deciding when LightGBM should stop training |
| Test | 2022 Q1 – 2026 Q2 | 97,526 | the reported results; never touched while building |

## Features (all known at the end of quarter *t−1*)

| Group | Features |
|---|---|
| Fund's history with this stock | its last 3 actions; how many quarters it has held the stock |
| Position | weight in the portfolio; rank (1 = biggest position) |
| Fund overall | number of stocks held; share of positions changed and bought last quarter; long-run average action |
| Stock | return last quarter; 4-quarter momentum; size rank |
| Peers | how many of our other funds held it; what share of them bought or sold it last quarter |
| Macro | average policy uncertainty (EPU) over the quarter |
| Fund identity | which fund it is |

## Results (test period)

| Predictor | Accuracy | Macro-F1 | Mean accuracy per fund |
|---|---|---|---|
| Majority class | 56.0% | 0.34 | 56.7% |
| Repeat last action | 44.4% | 0.41 | 46.7% |
| **LightGBM** | **62.3%** | **0.52** | **61.0%** |

The test period has 56% SELL, 41% BUY and 3% HOLD.

## What it means

1. **Hedge funds look much less predictable than mutual funds.** Mimicking Finance gains **19 points** over its naive baseline (52% → 71%). We gain about **4 points** (57% → 61%). This supports the idea that hedge funds act on information the public doesn't have.
   - *Caveat:* different data (13F vs Morningstar) and models (LightGBM vs a per-fund LSTM), so this isn't a final comparison yet.
2. **Big differences between funds** (38% to 82%).
   - **Hardest to predict:** concentrated, long-term or activist funds (Greenlight, Baupost, Dragoneer, Altimeter, TCI).
   - **Easiest:** Scion, Candlestick and **Berkshire Hathaway** (77%). Berkshire is our non-hedge-fund comparison, so it being among the easiest is a good sanity check.
3. **"Repeat last action" is worse than guessing.** Hedge funds don't keep doing the same thing to a stock quarter after quarter.
4. **What the model relies on:** which fund it is (33% of its decisions), position rank and size, and recent fund activity. **Stock returns and peers matter little,** which sets up RQ2: does *news* add anything?
5. **LightGBM stopped after only 78 rounds.** It found limited extra signal before it started fitting noise, another sign of low predictability.

## Is the gain real? (bootstrap, done 2026-11-13)

`src/rq1_bootstrap.py` redraws the test data 2,000 times and recomputes the gain each time. It redraws **whole funds**, or **whole quarters**, because rows from the same fund or quarter aren't independent.

| Redrawn by | Gain (per-fund accuracy) | 95% confidence interval |
|---|---|---|
| fund | +4.3 points | **+2.0 to +7.0** |
| quarter | +4.3 points | +3.4 to +5.1 |

- **The interval never includes zero:** the gain isn't luck.
- **The upper end (7) is far below Mimicking Finance's 19.** Hedge funds really are less predictable than mutual funds.

## To do in Phase 2
- **A fairer comparison with Mimicking Finance:** per-fund models, and their exact metric.
- **The HOLD class is only 3%** and almost never predicted; report a BUY/SELL-only version too.
- **Check whether results hold across years** (2022 rate hikes vs 2024–26).

## Check yourself

1. Why do we split data by *time* instead of randomly?
2. What does the majority-class baseline do, and why is it the right thing to beat?
3. Why do we report macro-F1 as well as accuracy?
4. Our model beat its baseline by 4 points; Mimicking Finance's beat theirs by 19. What could explain the difference? Name two possibilities.
5. Why is it reassuring that Berkshire Hathaway is among the easiest to predict?
