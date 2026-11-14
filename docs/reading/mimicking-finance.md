# "Mimicking Finance": summary

Cohen, Lu & Nguyen, NBER Working Paper 34849 (Feb 2026). 79 pages: the core is pages 1–35 and the rest is tables, figures and appendices. Page numbers below are the PDF's.

## In one sentence

An AI model trained on mutual fund managers' past trades predicts **71%** of their next-quarter trade directions (buy / hold / sell), and **the managers and trades the model can't predict are the ones that make money.**

## The 5 lines you'd write

1. **Question:** how much of a fund manager's "active" decision-making can be predicted from their past behaviour, and is the unpredictable part where skill lives?
2. **Data:** 1,706 actively managed US equity **mutual funds**, holdings from **Morningstar** (from SEC N-30D/N-CSR/N-Q/N-PORT filings), **1990–2023**, plus stock characteristics (JKP factor database) and macro data (FRED). (p. 12–13)
3. **Method:** an **LSTM** (a type of recurrent neural network), trained separately on rolling 28-quarter windows per fund. The first 20 quarters are for training and the last 8 for testing, never shuffled. (p. 15–17)
4. **Findings:**
   - **71% accuracy**, versus **52%** for always guessing the manager's most common action.
   - The managers whose trades are hardest to predict beat their peers. The easiest to predict underperform. Over four quarters the gap is **0.79%** (about 0.8% after risk adjustment).
   - Within a portfolio, the trades the model got wrong beat the ones it got right by **0.91% a year**.
   - At the stock level, stocks whose holders' trades are hardest to predict beat the most predictable ones by **4.24% a year** (3.86% alpha after the Fama-French five-factor adjustment). (p. 18, 27–30)
5. **What they don't do:**
   - **no hedge funds** (mutual funds only);
   - **no 13F data**;
   - **no news**;
   - **no LLMs or foundation models** (a custom LSTM instead);
   - no short positions.

   They suggest sell-side analysts and CEOs as future work. (p. 33)

## Details that matter for your paper

**Their trade definition is the same as ours** (p. 14): change in shares from quarter t to t+1:
- **BUY** if up by 1% or more;
- **SELL** if down by 1% or more;
- **HOLD** otherwise.

That's the same ±1% band we chose independently, so our numbers are directly comparable to theirs. Your sentence for the paper: *"Following Cohen, Lu and Nguyen (2026), we classify..."*

**Their baseline** (p. 16–17) is the "majority class": predict the manager's most common action from the training period every time. That gives **52%**. We'll use the same baseline so the comparison is fair.

**Their features** (p. 14–15), all lagged so they're known at prediction time:
- **Fund:** past returns, money flowing in or out, size.
- **Stock:** company size, value, momentum, factor exposures.
- **Peers:** how often similar funds bought, sold or held.
- **Macro:** interest rates, yield curve, credit spreads.
- **Position:** the position's size, its share counts over the last 6 quarters, and its rank within the portfolio.

**What predicts predictability** (p. 20–28):
- **More predictable:** longer-tenured managers, managers running several funds, larger funds, large- and mid-cap styles, and stable, mature companies with steady dividends.
- **Less predictable:** managers who own more of their fund, and growth, R&D-heavy or earnings-surprise companies.

**Their return tests** (p. 26–32), which we can copy for RQ3:
1. **Fund level:** rank managers into fifths by predictability and compare future returns against category peers.
2. **Within a fund:** compare stocks whose trades were predicted correctly against those predicted wrongly.
3. **Stock level:** rank stocks by how predictable their holders were and compare next-quarter returns, adjusted with Fama-French factors.

## How your paper extends it

| They | You |
|---|---|
| Mutual funds | **Hedge funds** (13F filings) |
| Morningstar (paid data) | **SEC 13F** (free, public, reproducible) |
| Past behaviour, company traits, macro | **plus news** (FinBERT sentiment) and policy uncertainty |
| Custom LSTM trained per fund | **Foundation models** (Chronos-2, zero-shot) and LLMs, against strong baselines |
| — | **Leakage and lookahead-bias checks** (the 45-day filing lag, LLM memorization) |

The finding to test: **is hedge fund skill also concentrated in the trades nobody can predict?** If yes, it confirms their result for a different, more secretive kind of investor. If no (hedge funds' unpredictable trades don't earn more), that's interesting too.

## Check yourself

1. What does "71% vs 52%" mean, and what is the 52% baseline?
2. Write their BUY / HOLD / SELL rule from memory.
3. Why do they split training and testing by time instead of randomly?
4. Which managers are hardest to predict, and how do those managers perform?
5. Name three things your paper does that theirs doesn't.
