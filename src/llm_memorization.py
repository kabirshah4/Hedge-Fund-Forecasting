"""Step 12b (Thanksgiving): does the LLM remember what happened next?

Llama 3.1 8B was trained on text up to December 2023, so it may have read
about our stocks' later price moves. If it "knows" the future, its outlook
would predict next-quarter returns better when it is told which company the
news is about (named) than when the company is hidden (anonymized).

For stock-quarters scored both ways (src/llm_news.py, 2021-2023):
  1. Each quarter, Spearman correlation between the LLM's outlook and the
     stock's NEXT-quarter return (market-adjusted within the universe).
  2. Average across quarters for named and anonymized runs, and the named
     minus anonymized difference with a t-test across quarters.
  3. How often the two runs give the same outlook.
A clearly positive named-minus-anonymized gap = memorization (lookahead bias).

Output: experiments/llm_memorization.csv

Usage: .venv/bin/python src/llm_memorization.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rq2_models import llm_features
from rq3_returns import quarterly_yahoo_returns

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"


def main() -> None:
    named, anon = llm_features("named"), llm_features("anon")
    if named is None or anon is None:
        print("need both named_*.jsonl and anon_*.jsonl in data/processed/llm/")
        return
    B = named.merge(anon, on=["cusip", "q"], suffixes=("_named", "_anon")).dropna()
    R = quarterly_yahoo_returns()
    R["ret_next"] = R.groupby("q")["ret"].transform(lambda s: s - s.mean())
    B = B.merge(R.assign(q=R["q"] - 1)[["cusip", "q", "ret_next"]], on=["cusip", "q"]).dropna()
    print(f"stock-quarters scored both ways with a next-quarter return: {len(B):,}")

    per_q = B.groupby("q").apply(lambda g: pd.Series({
        "named": g["llm_outlook_named"].corr(g["ret_next"], method="spearman"),
        "anon": g["llm_outlook_anon"].corr(g["ret_next"], method="spearman"),
        "n": len(g)}), include_groups=False)
    per_q = per_q[per_q["n"] >= 30]  # drop near-empty quarters (news ends in 2023)
    per_q["gap"] = per_q["named"] - per_q["anon"]
    t = per_q["gap"].mean() / (per_q["gap"].std(ddof=1) / np.sqrt(len(per_q)))
    pval = 2 * stats.t.sf(abs(t), len(per_q) - 1)
    agree = (B["llm_outlook_named"] == B["llm_outlook_anon"]).mean()
    out = pd.DataFrame([{
        "quarters": len(per_q), "stock_quarters": len(B),
        "rank_corr_named": per_q["named"].mean(), "rank_corr_anon": per_q["anon"].mean(),
        "gap_named_minus_anon": per_q["gap"].mean(), "t_gap": t, "p_gap": pval, "quarters_gap_positive": int((per_q["gap"] > 0).sum()), "same_outlook_share": agree,
    }])
    EXP.mkdir(exist_ok=True)
    out.to_csv(EXP / "llm_memorization.csv", index=False)
    print(out.round(3).T.to_string())
    print("\nper quarter:"); print(per_q.round(3).to_string())


if __name__ == "__main__":
    main()
