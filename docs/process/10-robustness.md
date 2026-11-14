# Step 10: RQ1 robustness checks

**Phase 2. Script:** [src/robustness.py](../../src/robustness.py) **Outputs:** `experiments/rq1_robustness.csv`, `experiments/rq1_by_style.csv`

## Why

A reviewer's first question will be: "Is 4 points vs 19 points a real difference, or did you just set the test up differently from Mimicking Finance?" So we re-ran RQ1 several ways.

## 1. Mimicking Finance's exact protocol (the most important check)

They train **one model per fund** on 20 quarters and test on the next 8, comparing against that fund's most common action. We did the same over four 8-quarter test blocks (2018Q3–2026Q2):

| Model | Mean per-fund accuracy |
|---|---|
| Majority (per fund) | 58.2% |
| **LightGBM, one per fund** | **57.1% (−1.1 points)** |
| LightGBM, pooled across funds | 60.7% (+2.5) |

**Under their rules, hedge funds aren't predictable at all.** A single hedge fund's 20 quarters of history are too short and irregular to learn from. Only pooling funds together, so the model learns patterns shared across funds, gets any gain. For mutual funds the same setup gives +19 points.

## 2. Other checks

| Check | Baseline | LightGBM | Gain |
|---|---|---|---|
| BUY vs SELL only (HOLD rows dropped) | 64.8% | 67.6% | +2.8 (AUC 0.66) |
| HOLD band ±5% instead of ±1% | 56.6% | 58.7% | +2.1 |
| Hedge funds only (no Berkshire) | 56.4% | 60.6% | +4.2 |

**AUC** is the chance the model ranks a random BUY above a random SELL. 0.5 is a coin flip and 1.0 is perfect, so 0.66 is weak.

## 3. By fund style (main test predictions)

| Style | Funds | Gain (points) |
|---|---|---|
| Multi-strategy | 6 | +7.8 |
| Activist / event-driven | 13 | +7.0 |
| Quantitative | 5 | +5.0 |
| Long/short equity and growth | 21 | +2.1 |
| Macro / value | 3 | −1.9 |
| Berkshire (control) | 1 | +8.5 |

Long/short stock pickers, the classic "skilled" hedge funds, are the least predictable.

## Takeaway

Every version gives the same answer: **2–4 points at most, and zero under the original paper's own protocol.** The result doesn't depend on our setup choices.

## Check yourself

1. Why might a per-fund model do *worse* than just guessing the fund's usual action?
2. What does an AUC of 0.66 mean in plain words?
3. Why is the "their protocol" check more convincing to a reviewer than our main result?
