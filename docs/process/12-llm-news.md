# Step 12: the LLM reads the news (built; runs overnight on your Mac)

**Scripts:** [src/llm_news.py](../../src/llm_news.py) (runs the LLM), [src/llm_memorization.py](../../src/llm_memorization.py) (memorization check). The results feed into [src/rq2_models.py](../../src/rq2_models.py) automatically.

## Why

RQ2 found that FinBERT sentiment, **averaged over a quarter**, doesn't help predict trades. One explanation: hedge funds don't trade on public news. The other: averaging is too blunt, and a model that *reads* the headlines together might pick up what matters. An LLM tests the second explanation.

## What it does

For each stock and quarter with news (24,598 stock-quarters from 2013 to 2023), Llama 3.1 8B reads up to 15 of the latest headlines and answers:
- **outlook** (−2 to +2): how good this news is for the stock next quarter.
- **surprise** (0 to 2): how much genuinely new information there is.

These become LightGBM features, like FinBERT's: last quarter's news is a **forecast**, this quarter's is a **nowcast**. Headlines about funds' stakes or 13F filings are removed first.

**Speed:** about 2.9 seconds per stock-quarter, so about 25 hours on one Mac for all 30,560 prompts.

## The memorization check (lookahead bias)

Llama's training data runs to **December 2023**, which covers all our news. It may have *read what happened next*, e.g. "Tesla stock soared in Q3". Then its "outlook" would really be a memory, not a reading of the news.

**Test (from Gao, Jiang & Yan):** score 2021–2023 a second time with the **company name and ticker hidden** ("the company"). If the named version predicts next-quarter returns clearly better than the anonymized one, the model is remembering.

**Limitation:** hiding names isn't perfect. Product names can give a company away ("Veklury" means Gilead), and FNSPID sometimes tags a headline to the wrong stock.

## Run plan: overnight on your Mac (about 3 nights)

You chose to run it on your own Mac, at night only. There are 30,560 prompts, about 25 hours of work in total, so about **3 nights of 8 hours**. The memorization-check prompts (2021–2023, named and anonymized, about 10 hours) run first, so they're done after night 1 or 2.

**Each night:**
1. Plug the Mac in. Keep the lid **open**; the screen can turn off, but closing the lid makes it sleep.
2. Double-click **`tools/llm_runner/overnight.command`** in Finder.
   - The first time, macOS may block it. Right-click → Open → Open.
3. Go to sleep. It **stops by itself at 7:00 AM** and tells you how much is left.
   - To stop it early, press Ctrl-C in its window, or just close the window.
   - Nothing is lost. The next run continues where this one stopped.

**When a run prints "ALL DONE"**, tell Claude, who then runs:
```
.venv/bin/python src/llm_news.py --import tools/llm_runner/results.jsonl
cd src && ../.venv/bin/python rq2_models.py && ../.venv/bin/python llm_memorization.py
```
After the first two nights, the memorization check can already be run on the partial results.

**If the prompts ever need rebuilding** (only if the news data changes): `.venv/bin/python src/llm_news.py --export`. This takes seconds and runs no AI.

## Check yourself

1. Why might an LLM help where averaged FinBERT sentiment didn't?
2. What would it mean if the named runs predicted returns much better than the anonymized runs?
3. Why does our news ending in 2023 make memorization a bigger worry for Llama 3.1 than it would be for news from 2025?
