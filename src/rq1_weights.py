"""Step 25: RQ1 without multi-strategy and quant firms, and dollar-weighted (fourth review, W10).

  excluding multi-strategy and quantitative firms (80% of test positions)
  dollar-weighted accuracy: each position weighted by its value at t-1,
  per fund, then averaged across funds

Output: experiments/rq1_weights.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from robustness import style_groups

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(51)


def boot(g):
    d = rng.choice(g, (2000, len(g))).mean(1)
    return g.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    P = pd.read_parquet(ROOT / "data/processed/rq1_test_preds.parquet")
    T = pd.read_parquet(ROOT / "data/processed/features.parquet", columns=["fund", "cusip", "q", "log_weight"])
    P = P.merge(T, on=["fund", "cusip", "q"], how="left")
    P["m"] = (P["pred_lightgbm"] == P["y"]).astype(float)
    P["b"] = (P["pred_majority"] == P["y"]).astype(float)
    P["w"] = np.exp(P["log_weight"].fillna(P["log_weight"].min()))
    P["style"] = P["fund"].map(style_groups())
    rows = []
    sub = P[~P["style"].isin(["Multi-strategy", "Quantitative"])]
    g = (sub.groupby("fund")["m"].mean() - sub.groupby("fund")["b"].mean()) * 100
    mu, lo, hi = boot(g.to_numpy())
    rows.append({"check": "excluding multi-strategy and quant", "funds": len(g), "gain": mu, "lo": lo, "hi": hi})
    wm = P.groupby("fund").apply(lambda d: np.average(d["m"], weights=d["w"]) - np.average(d["b"], weights=d["w"]),
                                 include_groups=False) * 100
    mu, lo, hi = boot(wm.to_numpy())
    rows.append({"check": "dollar-weighted within fund", "funds": len(wm), "gain": mu, "lo": lo, "hi": hi})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "experiments" / "rq1_weights.csv", index=False)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
