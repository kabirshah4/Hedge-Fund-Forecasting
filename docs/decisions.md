# Decisions log

Every choice made while building the data and models, with the reason.
This becomes the Methods section of the paper almost word for word.

| Date | Decision | Why |
|---|---|---|
| 2026-11-12 | Holdings data: SEC Form 13F Data Sets (official, Jul 2013 onward) | Free, official, reproducible |
| 2026-11-12 | Fund CIKs taken from filer names in the 13F data, then checked by hand on EDGAR | IDs typed from memory are unreliable |
| 2026-11-12 | Candidate list mixes multi-strategy, quant, long/short, activist, macro and "tiger cub" funds | Avoids a list that is all one style |
| 2026-11-12 | Berkshire Hathaway kept as a control, not counted as a hedge fund | Well-known non-hedge-fund comparison |
| 2026-11-12 | Python 3.12 environment via uv; run scripts with `.venv/bin/python` | The colons in the folder name break `uv run` |
| 2026-11-12 | 13F data: all 54 SEC data sets downloaded (2013 Q2 to 2026 Q2 filings), all zips verified | Full history; the study window is 2013-2023 |
| 2026-11-12 | One SEC ID per fund: the main investment-manager filer, never fund vehicles, GP entities, insurance subsidiaries or foundations | Avoids counting one fund's holdings several times |
| 2026-11-12 | Funds that changed SEC ID (Appaloosa, Elliott, Schonfeld, JANA, BlueCrest) are joined across both IDs | Same manager, new legal entity |
| 2026-11-12 | Excluded: Susquehanna (market maker), Touradji (stopped filing 2014) | Not a hedge fund / no usable history |
| 2026-11-12 | Open questions: family offices (Soros, Duquesne, BlueCrest), mostly long-only managers (Sands, AQR) | To decide when checking the list |
| 2026-11-12 | Use the filer that actually reports holdings: D. E. Shaw & Co., Inc. (0001009207), Carl Icahn (0000921669), ValueAct Holdings (0001418814) | Sister entities file 13F notices only, with no holdings |
| 2026-11-12 | Dropped family offices (Soros, Duquesne, BlueCrest), Sands Capital (long-only), Man Group (no single filer before 2018) | Keep only managers of outside capital with one continuous filing history |
| 2026-11-12 | Kept AQR | Major quant hedge fund manager; long-only share noted as a limitation |
| 2026-11-12 | Holdings: plain shares only (no options/bonds), one report per fund-quarter (latest restatement + "new holdings" amendments), VALUE converted to dollars (thousands before 2023-01-03) | Clean share counts for every fund, stock and quarter |
| 2026-11-12 | Labels: BUY if shares up >1%, SELL if down >1%, else HOLD; no labels across a missed filing | Simple, transparent rule; result 49% BUY / 47% SELL / 4% HOLD |
| 2026-11-12 | Split adjustment from holdings data is provisional; replace with real split history once stocks are mapped to tickers | Heuristics can't separate splits from same-quarter trading or rebalancing (see process/03) |
| 2026-11-12 | Stock universe = each quarter's 500 largest stocks by total value reported across ALL 13F filers (common stock and REITs; no ETFs, funds or ADRs). Chosen by the user (option A) over the S&P 500. | Built entirely from SEC data, so no renamed, acquired or delisted company drops out (no survivorship bias). Free data can't link old S&P tickers to 13F stock IDs: 10-28% of early S&P members would be lost |
| 2026-11-12 | Universe membership for quarter t uses the ranking at the end of quarter t-1 | No look-ahead: a stock can't join the study because of something that happened later |
| 2026-11-13 | Universe ranking = sum over filers of shares x typical (median) price, excluding filers whose own price is >5x off; equity CUSIPs only (chars 7-8 digits); 100+ holders; known non-company types excluded; unknown types kept | Robust to filers' unit mistakes; drops bonds and mutual-fund shares; keeps acquired/delisted companies |
| 2026-11-13 | Splits: product of Yahoo split ratios between quarter-ends for tickered stocks (85% of universe); holdings heuristic for the rest | Exact for known stocks; passes 9/9 test cases |
| 2026-11-13 | Final labels: 291,053 (50 funds, top-500 universe, Q3 2013 to Q2 2026); 48.8% SELL / 48.5% BUY / 2.7% HOLD | Same +/-1% three-class rule as Cohen, Lu & Nguyen (2026), so results are comparable |
| 2026-11-13 | RQ1 task: positions held at t-1, predict BUY/HOLD/SELL at t; split train 2013-2020, validation 2021, test 2022-2026Q2 | Same task as Mimicking Finance; time split prevents leakage |
| 2026-11-13 | First RQ1 result: LightGBM 61.0% mean per-fund accuracy vs 56.7% majority baseline (test) | Hedge funds gain ~4 points over naive vs ~19 for mutual funds in Mimicking Finance |
| 2026-11-13 | News: FNSPID company-tagged headlines cover 2013 to mid-2020 (Benzinga file) and mid-2020 to 2023 (Nasdaq file); none after 2023. RQ2 (with vs without news) is tested on 2022-2023; RQ1 keeps the full 2022-2026Q2 test. Chosen by the user. | Simple and honest; no unreliable name-matching of a second news source. State as a limitation; GDELT for 2024-26 is a possible extension |
| 2026-11-13 | RQ3 rough: LightGBM test predictions; "surprise" = model wrong; market-adjusted returns next quarter and post-filing (from quarter-end + 45 days); per-quarter means, t-test across 17 quarters | Mirrors Mimicking Finance section 7; Fama-French alphas in Phase 2 |
| 2026-11-13 | Downloaded SEC Form ADV filing data 2011-2024 (data/raw/adv/, 1.1 GB); Schedule D 7.B.(1) lists each adviser's private funds and their type | Basis for a rule-based hedge fund list in the Thanksgiving sprint (replaces the hand-picked list) |
| 2026-11-13 | Local LLM for the Phase 2/3 LLM experiment: Llama 3.1 8B via Ollama (training cutoff Dec 2023) | Runs on the Mac; known cutoff needed for the lookahead-bias test |
| 2026-11-13 | LLM test (20 headlines): Llama 3.1 8B labels sentiment sensibly; its "reports a stake" flag catches fund-stake headlines but also corporate M&A stakes (over-flags, acceptable for a leakage filter); ~8 s per headline while FinBERT runs | LLM can't process all 1.4M headlines: Thanksgiving design must sample or pre-filter by keyword |
| 2026-11-13 | Chronos-2 zero-shot on position share histories (up to 32 quarters, split-adjusted, normalized to 1 at t-1); label from P(up>1%)-P(down>1%) with cut-offs tuned on 2021 only | Result: 55.6% mean per-fund accuracy, no better than the majority baseline; covariates add nothing |
| 2026-11-13 | News leakage filter: drop headlines matching stake/13F/13D/13G/position-change phrases (0.7% of headlines) | Removes news that reports our funds' own trades (e.g. "Point72 Lowers Tyson Food Stake") |
| 2026-11-13 | News features in two settings: forecast (quarter t-1 news) and nowcast (quarter t news, before the 13F filing) | Forecast is the strict test; nowcast matches "predict before disclosure" |
| 2026-11-13 | RQ2 result: adding the Chronos-2 forecast (+0.14 pts, CI -0.44/+0.90), news forecast (+0.04, CI -0.50/+0.56), news nowcast (-0.28, CI -0.83/+0.22) or all (+0.06) to LightGBM: no gain, all intervals include zero | Hedge fund trades don't respond to public news as measured; coarse quarterly sentiment is the alternative explanation (test with LLM in Phase 2/3) |
| 2026-11-13 | BUG FIX (lookahead): RQ3 v1 kept a stock's t+1 return only if the stock was in the universe at t+1. Now every stock in the universe at t keeps its t+1 return; returns use Yahoo prices only | Universe membership at t+1 is future information; the bug inflated stock-level Q1-Q5 from about 0 to +0.79%/quarter |
| 2026-11-13 | RQ3 main test: calendar-time monthly portfolios (51 months), CAPM / FF3 / FF5+momentum alphas, Newey-West (3 lags) | Standard in fund-skill papers; more observations than 17 quarters |
| 2026-11-13 | RQ3 result: only the sell spread has a raw edge (+0.17%/month, t=2.33), explained by market and momentum (FF5+Mom alpha +0.08%, t=1.15); buys and Q1-Q5 are null | Paper's RQ3 claim changed to "unlike mutual funds, unpredicted hedge fund trades do not reliably earn more" |
| 2026-11-13 | Leakage audit: future-scramble test (randomize all inputs dated >= T, rebuild features, compare quarter T) at 2022Q1, 2024Q2, 2026Q1 -> PASS; filing timing: 0.1% of test-input fund-quarters filed after the next 13F is due | Direct evidence for the no-lookahead claim; remaining open risks: hindsight fund selection (Form ADV fix), pretrained-model memorization (LLM check), missing delisting returns |
| 2026-11-13 | Fair comparison: copy Mimicking Finance's protocol (per-fund model, 20-quarter train / 8-quarter test, 4 rolling blocks 2018Q3-2026Q2, per-fund majority baseline) | Result: per-fund LightGBM 57.1% vs majority 58.2% (-1.1); pooled 60.7% (+2.5). Hedge funds aren't predictable under the original protocol |
| 2026-11-13 | Robustness: BUY vs SELL only (+2.8, AUC 0.66), HOLD band +-5% (+2.1), drop Berkshire (+4.2); by style multi-strategy +7.8, activist +7.0, quant +5.0, L/S equity +2.1, macro/value -1.9 | All variants agree: small gains at most |
| 2026-11-13 | RQ3 robustness: value-weighted and 6/12-month horizons. All 24 FF5+Mom alphas positive, 4 with t>=1.96; tests overlap heavily | Paper wording: "small edge in the same direction as mutual funds that 51 months cannot establish" |
| 2026-11-13 | Rule-based replication sample from Form ADV (as of 2012-07..2013-06): hedge GAV >= $1B, >= 50% of private-fund GAV, >= 25% of RAUM, feeder/FoF excluded; linked to 13F by ADV CIK or exact legal name; >= 10 holdings in 2013Q2; top 100 by hedge GAV | Removes hindsight in fund selection; includes funds that later closed. 25 overlap with the hand-picked list |
| 2026-11-13 | Pipeline sample switch (src/sample.py, SAMPLE=adv); fund-independent files shared via symlinks | Same stocks and prices in both samples, so only the fund list differs |
| 2026-11-13 | ADV results: RQ1 gain +3.5 [1.6, 5.6] (vs +4.3); per-fund protocol -1.1 (same); RQ3 alphas 13/24 positive, none significant | RQ1 robust; the faint RQ3 edge was a hand-picked-sample artifact. Paper now says RQ3 conclusions rest on the ADV sample |
| 2026-11-13 | Added citations: SEC Private Funds Statistics 2025Q1 (hedge fund NAV $5.4T, GAV $12.6T), Brunnermeier & Nagel 2004, Griffin & Xu 2009, Agarwal et al. 2013, Noguer i Alonso & Franklin 2026 (arXiv 2606.27100, verified from abstract) | Fills intro and related-work TODOs |
| 2026-11-13 | LLM arm design: Llama 3.1 8B reads up to 15 latest headlines per stock-quarter (stake headlines removed) -> outlook (-2..2) and surprise (0..2); used as LightGBM features (forecast t-1, nowcast t). ~2.9 s per stock-quarter; 24,598 stock-quarters | Tests whether reading news beats averaged FinBERT sentiment; affordable on two Macs (~10 h) |
| 2026-11-13 | Memorization check: rerun 2021-2023 with company names/tickers replaced by "the company"; compare rank correlation of outlook with next-quarter returns, named vs anonymized | Model cutoff Dec 2023 covers all our news; a named > anonymized gap would indicate lookahead bias (Gao, Jiang & Yan) |
| 2026-11-13 | LLM runs only on Mom's Mac (user's request): prompts exported to tools/llm_runner/jobs.jsonl (30,560; memorization-check prompts first), stdlib-only runner run_jobs.py, results AirDropped back and imported | User's Mac stays free; other Mac needs only Ollama + built-in python3 |
| 2026-11-13 | Changed: LLM runs overnight on the user's own Mac (user's choice), ~3 nights of 8 h via tools/llm_runner/overnight.command (auto-stops at 07:00, resumable) | User wants the Mac free during the day |
| 2026-11-13 | AI disclosure in Method (visible in review) + Acknowledgments (IEEE style: tool, sections, level of use) | ACM and IEEE both require disclosing AI-generated code/text/figures; user must confirm every claim before submitting |
| 2026-11-13 | Page budget: 8 pages INCLUDING references (ICAIF 2026 call) | Final length target |
| 2026-11-13 | New RQ1 per-fund figure; finding: median per-fund gain is 0.0 in both samples; average gain driven by ~12 funds with baseline < 50% (+13-14 pts) | Strengthens RQ1: for a typical hedge fund the model adds nothing |
| 2026-11-13 | Mock-review response: class-robust RQ1 metrics (error reduction 10%/9% vs 40% for mutual funds; log-loss skill 4.6%/4.4%; BUY-vs-SELL AUC 0.66/0.67) | Answers "apples to oranges"; the critique's claim of a dominant HOLD class among mutual funds contradicts their 52% baseline |
| 2026-11-13 | Exclude new positions (as Mimicking Finance), stated explicitly with rationale; added to limitations | Initiation prediction is a different, highly imbalanced task |
| 2026-11-13 | By-style table (RQ1 gain, RQ3 FF5+Mom alphas): 2 of 10 RQ3 spreads significant (activist surprise buys t=2.3; multi-strategy surprise sells t=2.8), treated as leads | Answers the pooling critique; multi-strategy + quant = 79% of test rows |
| 2026-11-13 | RQ3 wording: "statistically indistinguishable from noise over this window", not "no skill" | 51 months = limited power |
| 2026-11-13 | Added late-quarter (last 15 days) news features to RQ2 | Tests whether quarterly averaging washes out the news signal |
| 2026-11-13 | Added TimesFM 2.5 (zero-shot + hybrid) and FinText Chronos-Small 2020 US (finance-pretrained on returns to 2020 only, fed share log-changes) | RQ2 conclusion should not rest on one TSFM; FinText vintage avoids lookahead |
| 2026-11-13 | Berkshire removed from the main sample (INCLUDE_CONTROL=1 restores it) | Headline "49 hedge funds" must be hedge funds only (reviewer W4e) |
| 2026-11-13 | Winton: joined successor CIK 0001612063 (data error: Winton vanished after 2014Q1); all funds re-checked for missed entity changes | Paper states we join histories across entity changes |
| 2026-11-13 | Long-only comparison sample from Form ADV (5D2d ">75%" mutual-fund AUM, hedge GAV < 5% RAUM, RAUM >= $1B, active share >= 0.6, top 100 -> 57 managers) run through the identical pipeline | Reviewer W1: within-study comparison. Result: long-only +6.2 vs hedge +4.0, difference n.s. -> gap with Cohen et al. not hedge-fund-specific |
| 2026-11-13 | Mechanism tests (opaque stocks, number of holdings): neither singles out hedge funds | Answers "why"; reported honestly |
| 2026-11-13 | Removed claim that hindsight selection explains the hand-picked RQ3 edge | Overlap split: funds in/not in the ADV sample have similar spreads |
| 2026-11-13 | All paper numbers generated by src/make_numbers.py (numbers.tex + tables/) | Prevents hand-typed mismatches (reviewer found several) |
| 2026-11-13 | Title changed to "How Predictable Are Hedge Fund Trades? Evidence from 13F Filings, Foundation Models and News" | Herding was never measured; framing now matches evidence |
| 2026-11-13 | N-PORT fund-level anchor: 300 active U.S. equity mutual funds (selected Dec 2019: >=20 universe stocks, >=60% of NAV, active share >=0.6, not ETF), plus families (registrant sums); train 2020Q1-2021Q2 | Review 2 W1. Result: individual mutual funds +2.9 (6% errors fixed), families +2.2 (5%), n.s. different -> hedge funds not unusual; aggregation not the explanation |
| 2026-11-13 | Same short training window for all samples (short_training.py) | Like-for-like vs N-PORT; hand-picked HF gain becomes n.s. (+1.8 [-0.7, 4.4]) and the paper says so |
| 2026-11-13 | Positive controls: fund-level synthetic injection (pipeline recovers 100% of attainable gain), earlier period 2017-2019 (no more predictable), 20-config tuning (no change) | Review 3 W1/W2: a null is informative only if the method can find a positive |
| 2026-11-13 | First injection design (random positions) discarded: model cannot identify injected positions, so its "ceiling" was unattainable; redesigned by fund | Design error caught before reporting |
| 2026-11-13 | RQ3 primary window = quarter-end to each fund's first filing date (capped at 45 days); Holm within and across samples; trade-quarter window and manager-level test added; prices spliced to 2026-10-30 (18 formation quarters) | Reviews 2-3. Hand-picked surprise sells: +0.43% (Holm p 0.009 across samples) but weaker with size-momentum matching and not replicated |
| 2026-11-13 | Removed "pre-specified" wording; text now says "primary" | Tests were revised after review, so not pre-registered |
| 2026-11-13 | AI disclosure reworded at the user's request: AI wrote code/figures/tables and aided drafting; the author revises the text | Must stay true at submission |
| 2026-11-13 | Author: Kabir Shah (hidden by acmart anonymous option for review); affiliation TODO | User request |
| 2026-11-13 | Page budget: removed fund-list table (to repo), RQ3 calendar figure, style table and robustness table (merged into five-sample table); DOIs dropped from printed references | 8 pages including references |
| 2026-11-13 | Integrity audit (user request re: ACM authorship policy): all 30 references re-verified against Crossref/arXiv; every typed number traced to a source; 7-word overlap check vs Mimicking Finance full text and all cited arXiv abstracts = 0 shared sequences | Caught and fixed two misdescriptions of cited work (Zhang & Zhang pitfalls; Izadifar & Rahmati persistence result) |
| 2026-11-13 | RQ3 p-values now from the t distribution (17 df) with NW(1); hand-picked sells Holm p 0.013 within / 0.038 across samples | Fourth review W5: normal p-values were too optimistic |
| 2026-11-13 | Added within-fund-quarter spread (+0.50, t=1.0, n.s.) and copier window after filing (n.s.); cite Alexander-Cici-Gibson (2007) and Koijen-Yogo (2019) | Fourth review W4, W6, W11 |
| 2026-11-13 | Learning curves (6/12/20/all training quarters): gains move by <=1.6 points; graded injection (rule sets 55/60/70% of labels): ~98% of attainable gain recovered | Fourth review W1, W2: history length does not explain the gap; pipeline detects moderate signals |
| 2026-11-13 | Stopped the blind-review loop after four rounds (all 2/5, reasons deepening); remaining asks (fine-tuning TSFMs, rebuilding Cohen et al.'s feature set) deferred | User agreed; diminishing returns vs LLM experiment and author's rewrite |

## 2026-11-13: AI-use section removed until the paper is final
Kabir asked to remove the "Use of Generative AI" section for now and add it back
when the paper is completely done. ACM requires full disclosure of generative-AI
use (code, figures, tables and drafted text), so it MUST be restored before arXiv
or any submission. Checklist item `ai1`; reminder comment in paper/sections/conclusion.tex.
Draft wording proposed: "The author used Claude (Anthropic) to write the
data-processing and analysis code, generate the figures and tables, and draft the
text of this paper. The author designed the study, directed every analysis,
checked all results and references against their sources, and takes full
responsibility for the content."

## 2026-11-13: Fifth blind review (score 3/5) and fixes
- Diebold-Mariano tests (src/dm_test.py -> experiments/dm_tests.csv): LightGBM beats the
  per-fund baseline in 18/18 test quarters (hand-picked, DM t = 7.4) and in every sample
  (t >= 5.3); log-loss likewise. Chronos-2 (t = -2.3) and TimesFM (t = -5.6) are significantly
  WORSE than the baseline; finance-trained Chronos not different (p = .11). Citations
  Diebold & Mariano 1995 and Harvey, Leybourne & Newbold 1997 verified on Crossref.
- "98% at every strength" is real (means 97.8/97.8/97.6; same random funds across strengths),
  but those rules were worth ~+22 points. New weak-signal test (src/weak_injection.py):
  rules worth +1.2 to +8.5 points raise the model's gain on injected funds above other funds
  in 12/12 draws. Abstract now says "planted signals as small as the real effect".
- Writing: attributed the 71% to the study; "says nothing" -> "gives no evidence"; abstract's
  last sentence conditioned on public data and standard features; withdrew "not specific to
  firm-level filings or recent years"; "surprise" = against usual direction stated in abstract;
  error-reduction no longer called class-mix independent; intro TSFM citation fixed (Noguer
  instead of the LLM review); duplicate "first" claim removed from Related Work.
- Cut for space (reviewer: too many secondary numbers): per-fund HOLD shares, by-style RQ1
  ranking, by-style RQ3 alpha, peer-feature ablation (also removed from Method).
- Open, needs Kabir's decision: firm-characteristic features (W1), large-trade labels (W4),
  yearly retraining (W6), foundation-model framing in title (W3).

## 2026-11-13 (night): reviewer W1, W4, W6 experiments (Kabir asleep, approved "a" then "continue")
- W1 firm characteristics (src/firm_chars.py, src/rq1_firmchars.py): 15 features (9 from SEC XBRL
  frames, used only if the period ended >= 120 days before the cutoff; 6 from prices). CIKs matched
  for 91% of universe stock-quarters (ticker, then exact name so delisted firms are covered).
  Extra gain at most +0.47 points (hand-picked, CI [-0.01, 1.00], DM p = .03, Holm across 5
  samples .16); others +0.02 to +0.32 or negative; max errors fixed 12.6%. Same on positions with
  fundamentals only. Abstract/discussion now say adding these barely helps.
- W4 large trades (src/big_trades.py): +-25% band gain +2.4 [0.3, 4.4] hand-picked (5% of errors),
  -1.7 to +4.3 elsewhere; full exits AUC 0.76-0.82 vs 0.65-0.75 from fund exit rate alone.
- W6 yearly expanding-window retraining (src/yearly_retrain.py): accuracy +0.7 points vs static
  (DM p = .007) but baseline also rises; gain +3.5 [1.6, 5.7], 8% of errors (4-13% across samples).
- Space: cut trade-quarter RQ3 window (text + method), news-expectation sentence, tuning sentence
  (still in Table 2), Zhang & Zhang review sentence, LSTM optimizer details (in code), per-fund
  capacity details, misplaced N-PORT clause in the Figure 1 paragraph. Both PDFs 8 pages.
- LLM runner: watcher (PID-based) restarts run_jobs.py without a stop time after the 7 AM stop,
  at Kabir's request ("keep it running after 7am").

## 2026-11-14: LLM arm results filled in
- Llama 3.1 8B scored 30,604 stock-quarters (24,596 named, 6,008 anonymized incl. a 44-row test run from Nov 13; same model and prompt). 1.1% of answers unreadable, treated as missing.
- RQ2: LLM features add -0.05 [-0.56, 0.44] (t-1) and +0.11 [-0.61, 0.93] (t) points; no change to any other RQ2 number after the rerun.
- Memorization (2021-2023, 5,812 stock-quarters, 12 quarters): rank corr with next-quarter return 0.00 named vs -0.02 anonymized; gap t = 2.5, p = .031, positive in 9/12 quarters; 71% identical outlooks. Name changes some answers but named outlooks don't predict returns.
- Fixed llm_memorization.py: a 1-stock 2024Q1 quarter was counted in the t-test's n; now quarters need >= 30 stocks, and a p-value is reported.
- To stay at 8 pages: shortened the method/results LLM text, the news sentence, the future-work line (LLM test now done), and turned Threats to validity from a list into a paragraph.
