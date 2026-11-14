# Step 15: fixing every point of the blind review

A blind reviewer (a subagent that saw only the paper) gave the draft **2/5, weak reject**. This step fixes every point. Several fixes changed the paper's story, so read this one carefully.

## The biggest change: a long-only comparison sample (reviewer's W1)

**Problem:** our headline compared *our* hedge-fund numbers with *another paper's* mutual-fund numbers: a different model, different data, and a different unit. A 13F covers a whole firm; Mimicking Finance studied single funds.

**Fix:** run the identical pipeline on **long-only, mutual-fund-style managers**, chosen by rule from Form ADV ([src/build_fund_list_longonly.py](../../src/build_fund_list_longonly.py)):
- more than 75% of assets from mutual funds (field 5D2d of the 2012 form);
- essentially no hedge funds;
- at least $1B in assets;
- **active share ≥ 0.6** (not index-like).

The result is 57 managers, 39 of them active in the test period.

**Result:**
- Long-only gain: +6.2 points [3.0, 9.5].
- Hedge funds: +4.0 [1.6, 6.7].
- The **difference isn't significant** (`src/compare_samples.py`).
- Both are far below the 40% of errors fixed for single mutual funds.

**So the story changed:** hedge funds aren't uniquely unpredictable. At the firm level that 13F reports, *everyone* is hard to predict. The title is now "How Predictable Are Hedge Fund Trades?", and "Herding or Skill?" is gone; the reviewer also noted we never measured herding.

## A mechanism test (your "PhD-level" point)

`src/mechanism.py` tests two explanations:
1. **Private information:** unpredictable trades should concentrate in smaller, more opaque stocks, more for hedge funds than for long-only managers. They don't: the hedge fund minus long-only difference is about zero.
2. **Aggregation:** firms with more positions should be less predictable. If anything they're *more* predictable.

Neither singles out hedge funds. Honest, if unexciting.

## Data errors found and fixed

- **Berkshire in the headline.** The "49 hedge funds" numbers included Berkshire. It's now removed from the sample (`INCLUDE_CONTROL=1` brings it back for a sanity run).
- **Winton missing after 2014.** It changed SEC ID (Winton Capital Management → Winton Group). Our list had only the old ID, so a big quant fund silently vanished. Fixed by joining both IDs, and I checked every other fund for the same problem (none clear).
- **One wrong claim removed.** We'd said hand-picking (hindsight) explained the faint RQ3 edge. Splitting the hand-picked sample showed it didn't, so the claim is gone.

## Other fixes

- **Stronger and more comparable models:** logistic regression, a simple rule, and an **LSTM** (Mimicking Finance's model family), both pooled and per fund (`src/rq1_sequence_models.py`).
- **More foundation models:** **TimesFM 2.5** and a **finance-trained Chronos** (FinText, trained on returns up to 2020 only, so it can't have seen our test years) (`src/run_tsfm_extra.py`).
- **News from the last 15 days only:** tests whether averaging a whole quarter washes out the signal. Still nothing.
- **RQ3 power** (`src/rq3_power.py`):
  - confidence intervals and the smallest effect we could detect;
  - we **can't** rule out a mutual-fund-sized effect, and the paper now says so;
  - "surprise" also measured against the majority guess;
  - activist portfolios too small (2 stocks) to estimate.
- **Reviewer's Q3 (HOLD):** HOLD is concentrated in Elliott, Icahn and Pershing, which explains why dropping HOLD moves the baseline.
- **Timing wording:** features use t−1 filings, which are public 45 days into quarter t.
- **Literature:** herding (Lakonishok–Shleifer–Vishny, Wermers, Sias), trade-level skill, characteristic-based benchmarks, confidential filings, active share. Every citation was checked against Crossref or arXiv.
- **Figures:**
  - Fig. 1: the overlapping label is fixed, sample medians are marked, and the long-only sample is added.
  - Fig. 2: which side is subtracted is now in each row label.
- **AI disclosure:** moved to the end, worded neutrally ("the authors").

## New system: numbers come from files

`src/make_numbers.py` writes `paper/numbers.tex` (one LaTeX command per number) and `paper/tables/*.tex`. The paper never contains a hand-typed result, so the abstract can't disagree with a table again. After any rerun:
```
.venv/bin/python src/make_numbers.py && (cd paper && tectonic main.tex)
```

## Check yourself

1. Why does comparing with *long-only managers in our own pipeline* answer the reviewer better than comparing with Mimicking Finance's published number?
2. What does "active share ≥ 0.6" filter out, and why does that matter here?
3. Why can't we say "unpredicted trades don't earn more"? What *can* we say?
4. How did the Winton bug happen, and how would you catch the next one?
