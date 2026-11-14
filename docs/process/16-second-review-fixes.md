# Step 16: fixing the second blind review

The second blind reviewer gave **2/5** again, but for deeper reasons than the first. The first review found mistakes; this one questioned whether the study can tell hedge funds apart from its own pipeline. What we did:

## 1. A fund-level anchor: SEC Form N-PORT (the reviewer's main request)

**Problem:** the long-only sample shares every quirk of our pipeline, so "hedge funds look like long-only managers" could just mean "our pipeline can't predict anyone".

**Fix:** N-PORT gives holdings for **individual mutual funds** (each "series" is one fund) since late 2019 ([src/download_nport.py](../../src/download_nport.py), [src/build_nport.py](../../src/build_nport.py)).
- **Choosing the funds** (as of December 2019): 300 active U.S. large-cap equity funds:
  - at least 20 universe stocks, making up 60%+ of net assets;
  - active share ≥ 0.6 (not index-like);
  - not ETFs.
- **Parent level:** the same funds added up to their registrant (fund family trust), which copies what a firm-wide 13F looks like.

If funds are predictable individually but not once added up, **aggregation** is the reason. If they're unpredictable either way, **our pipeline or the period** is the limit.

**Limitation:** N-PORT starts in late 2019, so training covers only 2020Q1–2021Q2 (the split is set per sample in `src/sample.py`).

## 2. RQ3 redesigned ([src/rq3_v2.py](../../src/rq3_v2.py))

- **The right window.** I had called the next quarter "before the filing is public", but filings appear 45 days in. The primary test now uses the exact window from quarter-end to **each fund's own filing date**. Hedge funds file on day 45 (median); only 0.2% file within a month.
- **Unconditional benchmark:** do the funds' buys beat their sells at all? No (+0.09%, n.s.).
- **Holm correction** across the three primary tests in each sample.
- **Result:** in the hand-picked sample, surprise sells fall 0.42% more before disclosure (t = 3.1, Holm p = 0.005). But:
  - after matching each stock on **size and momentum** it's 0.29% (t = 1.8);
  - it doesn't replicate in the Form ADV or long-only samples.

  So it's reported as a sample-specific, momentum-sensitive lead.
- **Surprise intensity:** sorting by how unlikely the model found each trade. Nothing.
- **Delisting bound:** treat every trade without a price as a −30% delisting. The result holds.
- **Newey-West lags** now equal the holding period (3, 6 or 12 months).

## 3. RQ1 checks ([src/rq1_extra.py](../../src/rq1_extra.py))

| Check | What it answers | Result (hand-picked) |
|---|---|---|
| Fund-weighted training | Is "median fund gains 0" caused by big funds dominating training? | No: median still 0.0 |
| Recent-majority rule | Is there a stronger simple baseline? | Worse (52.9% vs 56.7%): hedge funds' direction flips |
| Without peer features | Do different peer sets matter? | No (+4.6) |
| **Oracle** (told the fund's real direction) | Is it fund-wide scaling or stock picking? | Even knowing the direction, only +7.0: **stock picks stay unpredictable** |
| Foundation-model AUC | Is the threshold collapse hiding signal? | Some signal (0.52–0.58), less than LightGBM (0.66) |

## 4. Smaller fixes

- **Table 3:** bolding is computed automatically (the LSTM is best on fund accuracy), every model has macro-F1, and there's an AUC column.
- **Wording:** "at most half a point" vs "about one point" made consistent; "not significantly different" no longer implies "equal"; the 99-of-100 funds and the 51 months are explained.
- **Data quality:** at most 0.7% possible missed splits (an upper bound); 6 CUSIP-change artefacts (0.01%); universe filter counts.
- **Method:** LSTM and foundation-model settings stated, hyperparameters fixed in advance, exits = SELL, the news-filter phrases listed.
- **Title:** "News" dropped until the LLM experiment is done.

## Check yourself

1. Why does comparing funds individually vs added up to their family test the aggregation explanation?
2. Why is "quarter-end to the fund's own filing date" a better window than "the next quarter"?
3. The surprise-sells result has t = 3.1. Why don't we call it a finding?
