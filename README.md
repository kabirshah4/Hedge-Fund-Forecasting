# Hedge fund trade predictability with foundation models

Code for the paper *Herding or Skill? Foundation Models and the Predictability of Hedge Fund Trades*.

It asks three questions:
- **RQ1:** How predictable are hedge funds' quarterly buy/hold/sell decisions from public information?
- **RQ2:** Do time-series foundation models (Chronos-2, TimesFM, a finance-pretrained Chronos) or news (FinBERT, an LLM) help?
- **RQ3:** Do the trades no model predicts earn more?

All inputs are free and public: SEC Form 13F and Form ADV data, OpenFIGI, Yahoo Finance, FNSPID news, the EPU index and Fama–French factors.

## Setup

Requires Python 3.12. Install with `uv`:

```bash
uv venv && uv pip install -e .
export EDGAR_IDENTITY="Your Name your.email@example.com"   # SEC requires a contact in the User-Agent
```

If the project path contains a `:`, `uv run` fails. Call `.venv/bin/python` directly, as in the commands below. Run every script from `src/`.

## Data you download by hand

Put these files in `data/raw/`:

| Folder | Source |
|---|---|
| `13f/` | Downloaded by `download_13f.py` (SEC Form 13F data sets, 2013Q3 onward) |
| `adv/` | SEC Form ADV filing data 2011–2024, parts 1 and 2. Unzip into `adv/extracted/`. |
| `news/` | FNSPID: `All_external.csv` and `nasdaq_exteral_data.csv` (github.com/Zdong104/FNSPID_Financial_News_Dataset) |
| `epu/` | `US_Policy_Uncertainty_Data.xlsx` (policyuncertainty.com) |
| `ff/` | Fama–French 5 factors (2x3) and momentum, monthly CSVs (Ken French data library) |

## Pipeline (main, hand-picked sample)

| Step | Script | Output |
|---|---|---|
| 1 | `download_13f.py --go` | 13F data sets |
| 2 | `find_fund_ciks.py` → check by hand → `data/fund_list.csv` | fund list |
| 3 | `build_holdings.py` | holdings, 13F-implied prices |
| 4 | `build_universe.py` | top-500 stock universe per quarter |
| 5 | `map_tickers.py`, `map_tickers_fallback.py`, `get_market_data.py` | CUSIP→ticker, prices, splits |
| 6 | `build_labels.py`, `build_features.py` | BUY/HOLD/SELL labels, features |
| 7 | `baselines.py`, `rq1_bootstrap.py`, `rq1_metrics.py`, `robustness.py`, `fig_rq1.py` | RQ1 |
| 8 | `prepare_news.py`, `score_news.py` (FinBERT) | news sentiment |
| 9 | `run_chronos.py`, `chronos_scores_all.py`, `run_tsfm_extra.py` | foundation-model forecasts |
| 10 | `llm_news.py --export`, `tools/llm_runner/run_jobs.py`, `llm_news.py --import …` | LLM reading of news |
| 11 | `rq2_models.py`, `llm_memorization.py` | RQ2 |
| 12 | `rq3_returns.py`, `rq3_factors.py`, `rq3_by_style.py` | RQ3 |
| 13 | `leakage_audit.py` | future-scramble test |

## Rule-based sample (SEC Form ADV)

```bash
.venv/bin/python build_fund_list_adv.py      # data/fund_list_adv.csv
export SAMPLE=adv                             # scripts now use data/processed_adv, experiments_adv
for s in build_holdings build_labels build_features baselines rq1_bootstrap rq1_metrics robustness rq3_returns rq3_factors; do
  ../.venv/bin/python $s.py
done
```

Files that don't depend on the funds (universe, prices, ticker map) are shared between the two samples.

## Layout

- `src/`: pipeline scripts. Each starts with a docstring explaining what it does and why.
- `experiments/` and `experiments_adv/`: result tables (CSV) and figures.
- `paper/`: LaTeX source (ACM `acmart`). Builds with `tectonic main.tex` or on Overleaf.
- `docs/`: research plan, decision log (`decisions.md`) and step-by-step notes (`process/`).
- `tools/llm_runner/`: a standalone LLM job runner. It needs only Ollama and the system `python3`.

## AI use

The analysis code and much of the text were written with Claude (Anthropic) via Claude Code, under the author's direction. See the AI disclosure in the paper.
