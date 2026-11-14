"""Phase 2 (early): is LightGBM's RQ1 gain over the majority baseline real?

Bootstrap = re-draw the test data many times (with replacement) and recompute
the gain each time; the middle 95% of those gains is the confidence interval.
If the interval stays above zero, the gain is unlikely to be luck.

We re-draw whole clusters, not single rows, because rows from the same fund or
the same quarter are not independent:
  by fund     re-draw the 50 funds        (does the gain depend on a few funds?)
  by quarter  re-draw the test quarters   (does it depend on a few quarters?)

Metric: mean per-fund accuracy (as in Mimicking Finance) and pooled accuracy.

Output: experiments/rq1_bootstrap.csv
Usage:  .venv/bin/python src/rq1_bootstrap.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
PROC = SMP.PROC
B = 2000
rng = np.random.default_rng(7)


def main() -> None:
    P = pd.read_parquet(PROC / "rq1_test_preds.parquet")
    P["c_lgb"] = (P["pred_lightgbm"] == P["y"]).astype(float)
    P["c_maj"] = (P["pred_majority"] == P["y"]).astype(float)

    # per (fund, quarter) sums, so resampling is fast
    g = P.groupby(["fund", "q"]).agg(n=("y", "size"), lgb=("c_lgb", "sum"), maj=("c_maj", "sum")).reset_index()
    funds = g["fund"].unique()
    quarters = g["q"].unique()

    def stats(sub: pd.DataFrame) -> tuple[float, float]:
        f = sub.groupby("fund")[["n", "lgb", "maj"]].sum()
        per_fund = (f["lgb"] - f["maj"]) / f["n"]
        pooled = (f["lgb"].sum() - f["maj"].sum()) / f["n"].sum()
        return per_fund.mean(), pooled

    point = stats(g)
    rows = []
    for name, units, col in [("by fund", funds, "fund"), ("by quarter", quarters, "q")]:
        by_unit = {u: d for u, d in g.groupby(col)}
        draws = []
        for _ in range(B):
            pick = rng.choice(units, size=len(units), replace=True)
            sub = pd.concat([by_unit[u].assign(_k=i) for i, u in enumerate(pick)])
            if col == "fund":  # a fund drawn twice counts as two separate funds
                sub["fund"] = sub["fund"] + "#" + sub["_k"].astype(str)
            draws.append(stats(sub))
        d = np.array(draws)
        for j, metric in enumerate(["mean per-fund accuracy", "pooled accuracy"]):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            rows.append({"resampling": name, "metric": metric, "gain_points": point[j] * 100,
                         "ci95_low": lo * 100, "ci95_high": hi * 100,
                         "share_of_draws_above_0": (d[:, j] > 0).mean()})
    R = pd.DataFrame(rows)
    SMP.EXP.mkdir(exist_ok=True)
    R.to_csv(SMP.EXP / "rq1_bootstrap.csv", index=False)
    print("LightGBM minus majority baseline, in accuracy points (test 2022-2026Q2):")
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
