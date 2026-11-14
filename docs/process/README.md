# Process log

A plain-language record of every step: what we did, why, how, what it made,
and what could go wrong. Read these in order. Each step ends with **Check
yourself** questions; you should be able to answer them in your own words.

Before moving to the next phase we do a **learning check**: a short recap and
questions you answer without looking. That's how you'll be able to explain and
defend every part of the paper.

| Step | File | Status |
|---|---|---|
| 1 | [01-setup-and-sec-data.md](01-setup-and-sec-data.md): project setup, downloading SEC 13F data | done |
| 2 | [02-fund-list.md](02-fund-list.md): choosing which hedge funds to study | done |
| 3 | [03-holdings-and-labels.md](03-holdings-and-labels.md): turning filings into BUY / HOLD / SELL labels | done (labels finalized in step 4) |
| 4 | [04-universe-and-market-data.md](04-universe-and-market-data.md): stock universe, tickers, prices, real splits, final labels | done |
| 5 | [05-baselines-rq1.md](05-baselines-rq1.md): features and baselines, the first RQ1 result | done |
| 6 | [06-chronos-and-news.md](06-chronos-and-news.md): Chronos-2 (zero-shot) and news sentiment, RQ2 | done |
| 7 | [07-rq3-returns.md](07-rq3-returns.md): do unpredicted trades earn more? (rough; superseded by 8) | done |
| 8 | [08-rq3-factors.md](08-rq3-factors.md): RQ3 with Fama-French alphas, and the lookahead bug it exposed | done |
| 9 | [09-leakage-audit.md](09-leakage-audit.md): future-scramble test, filing timing | done |
| 10 | [10-robustness.md](10-robustness.md): Mimicking Finance protocol, BUY/SELL, HOLD band, by style | done |
| 11 | [11-adv-fund-list.md](11-adv-fund-list.md): rule-based fund list from Form ADV, rerun on 100 funds | done |
| 12 | [12-llm-news.md](12-llm-news.md): LLM reads the news + memorization check | built; runs overnight on your Mac |
| 13 | [13-rewrite.md](13-rewrite.md): rewrite toward the final 8-page paper | in progress |
| 14 | [14-reviewer-critiques.md](14-reviewer-critiques.md): answering five mock-review critiques | done |
| 15 | [15-review-fixes.md](15-review-fixes.md): fixing every blind-review point (long-only sample, mechanism, data fixes) | done |
| 16 | [16-second-review-fixes.md](16-second-review-fixes.md): second blind review: N-PORT anchor, RQ3 redesign, RQ1 checks | done |

**Phase 1 learning check:** [phase1-learning-check.md](phase1-learning-check.md), 12 questions to answer before Phase 2.

Reading notes:
- [../reading/mimicking-finance.md](../reading/mimicking-finance.md): summary of the paper we extend

Related files:
- [../decisions.md](../decisions.md): every decision in one table (becomes the Methods section)
- [../plan.md](../plan.md): the research plan
- [../phases.txt](../phases.txt): the schedule
