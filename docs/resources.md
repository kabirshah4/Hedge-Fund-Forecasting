# Resources: Foundation Models for Anticipating Hedge Fund Trades

Collected 2026-11-12. Verify every paper's claims against its full text before citing.

## Closest prior work (read first)

| Paper | Why it matters |
|---|---|
| Cohen, Lu, Nguyen, **"Mimicking Finance"**, NBER WP 34849 (Feb 2026). [NBER](https://www.nber.org/papers/w34849) · [PDF](https://www.nber.org/system/files/working_papers/w34849/w34849.pdf) | Reports that ML predicts ~71% of *mutual fund* manager trade directions (vs ~52% for a simple baseline). Less predictable managers and positions outperform. This is the closest existing version of our question, so the paper must position against it. |
| Izadifar & Rahmati, **"Institutional Equity Holdings Prediction Using Node Affinities of Dynamic Graphs"**, [arXiv 2607.12067](https://arxiv.org/abs/2607.12067) (Jul 2026) | Nearly the same data as ours: 99 managers × S&P 500, 48 quarters, 2013–2025 13F. Uses temporal graph ML to predict portfolio weights. **Persistence and EMA baselines are almost as good as the best model (NDCG 0.889 vs 0.913), and extra features add <1.2%.** Must cite. Include persistence/EMA baselines. Our differences: hedge funds only, buy/sell direction, news, foundation models, skill test. |
| "Predicting Institutional Investment Trends from SEC Form 13F Filings Using Machine Learning", [IEEE Xplore 11086238](https://ieeexplore.ieee.org/document/11086238/) | Tree models on 2013–2024 13F data, predicting aggregate holdings value. Related-work citation. |
| Kaniel, Lin, Pelger, Van Nieuwerburgh, "Machine-Learning the Skill of Mutual Fund Managers", JFE 150(1), 2023. [NBER](https://www.nber.org/papers/w29723) | ML predicts fund *performance* (not trades). Related-work citation. |

## TSFMs in finance (expect weak zero-shot results)

| Paper | Finding (from abstracts; verify) |
|---|---|
| Rahimikia & Ni, "Re(Visiting) Time Series Foundation Models in Finance", [arXiv 2511.18578](https://arxiv.org/abs/2511.18578) | Zero-shot Chronos/TimesFM trail CatBoost/LightGBM on daily returns (94 countries, 34 years). Pretraining on financial data helps. |
| Noguer I Alonso & Franklin, "Pretrained TSFMs for Financial Return Forecasting", [arXiv 2606.27100](https://arxiv.org/abs/2606.27100) | TimesFM-2.5 and Moirai-2.0 rank best, but gains over random walk are small. |
| FinCast, [arXiv 2508.19609](https://arxiv.org/abs/2508.19609) | Finance-specific foundation model. |
| EXAONE Finance 1.0, [arXiv 2609.04239](https://arxiv.org/abs/2609.04239) | Attention-free finance TSFM that handles missing data in panels. |
| Marconi, [arXiv 2507.07296](https://arxiv.org/abs/2507.07296) | Tiny Time Mixers beat naive baselines on equity spreads; specialised models often match or beat it. |

## LLM lookahead bias and memorization (needed for the LLM arm)

- Gao, Jiang, Yan, "A Test of Lookahead Bias in LLM Forecasts", [arXiv 2512.23847](https://arxiv.org/abs/2512.23847). Proposes the Lookahead Propensity (LAP) test; reports memorization inflates the apparent predictive power of LLM forecasts.
- Lopez-Lira, Tang, Zhu, "The Memorization Problem: Can We Trust LLMs' Economic Forecasts?", [arXiv 2504.14765](https://arxiv.org/abs/2504.14765)
- FinCAD (inference-time suppression of memorized outcomes), [arXiv 2605.24564](https://arxiv.org/abs/2605.24564)
- ESSEC, ["AI's Predictable Memory in Financial Analysis"](https://faculty.essec.edu/research/16148-ais-predictable-memory-in-financial-analysis)

## Reference paper (style model)

- Zhang & Zhang, "A Review of LLMs for Stock Price Forecasting from a Hedge-Fund Perspective", [arXiv 2605.05211](https://arxiv.org/abs/2605.05211). A survey. Its Section III pitfalls (leakage, short datasets, weak baselines, MSE vs P&L, illiquidity) become our evaluation checklist.

## Data (all free)

| Need | Source | Notes |
|---|---|---|
| Hedge fund holdings | [SEC Form 13F Data Sets](https://www.sec.gov/dera/data/form-13f) ([readme](https://sec.gov/files/form_13f_readme.pdf)) | Official flattened 13F data, Jul 2013 onward, updated quarterly. Use INFOTABLE + SUBMISSION + COVERPAGE. |
| Pre-2013 13F | [EDGAR-Parsing project](https://elsaifym.github.io/EDGAR-Parsing) | Parsed 13F filings for 1999–2020, code on GitHub. |
| Which 13F filers are hedge funds | SEC Form ADV (private fund section, Schedule D 7.B) via [adviser data](https://www.sec.gov/foia-services/frequently-requested-documents/form-adv-data) | No ready-made list exists; build one by matching ADV hedge fund advisers to 13F filers. [UNC note](https://uncipc.org/wp-content/uploads/2024/05/Institutional_Hedge_Funds_note.pdf) gives a template. |
| News + prices | [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) ([paper](https://arxiv.org/abs/2402.06698), CC BY 4.0) | 15.7M news records + prices for 4,775 S&P 500 companies, 1999–2023. Has an updater for later years. |
| News, global | [GDELT](https://www.gdeltproject.org/) | Free, has tone scores, 2015 onward. |
| Company events | SEC 8-K filings via `edgartools` | Free. |
| Prices | yfinance / Stooq | Free. Watch for survivorship (delisted names). |

## Models

- TSFMs: Chronos / Chronos-2 (amazon-science/chronos-forecasting), TimesFM 2.5 (google-research/timesfm), Moirai 2.0 (SalesforceAIResearch/uni2ts), Tiny Time Mixers (ibm-granite/granite-tsfm). All run zero-shot on a laptop at small or base size.
- LLMs: a local 7–8B model via Ollama for the bulk run, plus an API model on a sample. Record each model's training cutoff for the lookahead test.
- Baselines: no-change, last-quarter trend, LightGBM/CatBoost, logistic regression.

## Venues

- **IEEE CIFEr 2027** (part of SSCI, Gold Coast, Feb 14–17 2027): submission deadline was Aug 10 2026, now passed ([CFP post](https://groups.google.com/g/aifin-worldwide/c/VW8BMlX5BhI)). The next cycle is likely around Aug 2027.
- **ACM ICAIF 2027**: deadline not yet found. Check icaif.acm.org.

## Installed agent skills

`ml-paper-writing`, `literature-review`, `literature-search-arxiv`, `latex-paper-en`, `edgartools-sec-data`, `backtesting-frameworks` (in `~/.claude/skills`). Note: `edgartools-sec-data` was written for a workshop. It covers 10-K/10-Q, not 13F, so use the `edgartools` docs for 13F.
