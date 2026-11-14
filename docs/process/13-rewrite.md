# Step 13: rewrite toward the final paper (in progress)

**Target:** ICAIF allows at most **8 pages *including* references** (2026 call; recheck the 2027 call). We're at 6.

## Done so far (Nov 13)

- **AI disclosure,** in two places:
  - a paragraph at the end of Method, which reviewers see;
  - an Acknowledgments block, which IEEE asks for. It's hidden in anonymous review mode and appears in the arXiv version.
  - **You must check every sentence is true before submitting**, especially "reviewed the code's outputs and the final text". That becomes true after the teaching session and your full read.
- **Preprint rules:**
  - ACM gives authors the right to post preprints on arXiv. The ICAIF 2026 call said nothing either way.
  - I couldn't find CIFEr's rules.
  - Recheck both 2027 calls before submitting (checklist pb2 stays open).
- **New RQ1 figure** (`src/fig_rq1.py`): each fund's accuracy against its own baseline, in both samples. **New finding:** the *median* fund gains 0.0 points. The average gain comes from about 12 funds with no dominant habit (baseline below 50%), which gain 13–14 points; the rest gain about 1.
- **Sample summary table** in Data, covering both samples.
- **Introduction:** contributions and preview rewritten for the final findings (two samples, the protocol replication, the leakage test, the lookahead lesson).
- **Discussion:** new "Implications" paragraph for investors, foundation-model research, and two pitfalls to avoid (hand-picked survivors, future universe membership).

## Still to do

- The LLM results (after the overnight runs) go in RQ2, plus the memorization check.
- A final pass for flow and length; tighten to 8 pages including references.
- The repository link, after review.

## Before making the code public (after review)

- [x] README with setup, data sources and pipeline order (Nov 13)
- [x] pyproject lists every dependency (statsmodels, timesfm, pypdf added)
- [ ] Make a **separate public repo** without the personal planning files: `docs/phases.txt`, `docs/checklist.json`, `docs/mentors.md`, `docs/process/phase1-learning-check.md`, and any `*.log` (logs contain your Mac's home folder path)
- [ ] Put the repo link in the paper's Reproducibility paragraph (anonymized during review, e.g. via anonymous.4open.science)
