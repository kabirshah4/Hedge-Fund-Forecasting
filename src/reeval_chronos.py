"""Re-score the saved Chronos-2 zero-shot forecasts on the current sample.

run_chronos.py saved one score per position (chronos_preds.parquet). When the
sample changes (Berkshire removed, Nov 13) the forecasts themselves don't
change, since each depends only on that position's own history, but the
cut-offs (tuned on 2021) and the test metrics must be recomputed on the
rows that remain. No model is rerun.

Output: experiments/rq2_chronos.csv (overwrites; old file kept as rq2_chronos_with_berkshire.csv)
"""

import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import run_chronos as RC


def main() -> None:
    T = pd.read_parquet(RC.PROC / "features.parquet", columns=["fund", "cusip", "q", "period", "y"])
    C = pd.read_parquet(RC.PROC / "chronos_preds.parquet")
    D = T.merge(C.drop(columns=["period", "y"], errors="ignore"), on=["fund", "cusip", "q"], how="left")
    old = RC.EXP / "rq2_chronos.csv"
    keep = RC.EXP / "rq2_chronos_with_berkshire.csv"
    if old.exists() and not keep.exists():
        shutil.copy(old, keep)
    rows = []
    for s, cov in [("A", "own history"), ("B", "own history+ret"), ("D", "own history+ret+epu")]:
        col = f"chronos_{s}_score"
        if col not in D:
            continue
        sub = D[D["period"] > RC.TRAIN_END].dropna(subset=[col])
        val, test = sub[sub["period"] <= RC.VAL_END], sub[sub["period"] > RC.VAL_END]
        lo, hi = RC.tune(val[col].to_numpy(), val["y"].to_numpy())
        pred = RC.score_to_label(test[col].to_numpy(), lo, hi)
        y = test["y"].to_numpy()
        pf = pd.Series(pred == y).groupby(test["fund"].to_numpy()).mean()
        rows.append({"input_set": s, "covariates": cov, "test_rows": len(test),
                     "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
                     "mean_fund_accuracy": pf.mean(), "cutoffs": f"{lo:.2f}/{hi:.2f}",
                     "share_predicted_hold": float((pred == 1).mean())})
    R = pd.DataFrame(rows)
    R.to_csv(old, index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
