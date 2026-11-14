"""Step 22b: synthetic injection by FUND (corrected design).

A random share of funds (10% or 25%) is made to follow a rule on a feature
the model can see: BUY if the stock's four-quarter momentum at t-1 is
positive, else SELL. In the "noisy" version each of their positions follows
the rule with probability 0.8 and keeps its real label otherwise. All of
these funds' labels change (training, validation and test), so a model that
uses fund identity together with momentum can learn the rule. The question:
if some funds really traded by a simple visible rule, would the pipeline see
it?

Reported, for the injected funds: mean test accuracy of LightGBM, of their
majority baseline, and of the rule itself (the attainable ceiling); and the
share of the attainable gain the model recovers.

Output: experiments/injection_test.csv
Usage:  .venv/bin/python src/injection_test.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import baselines as BS
import robustness as RB

ROOT = Path(__file__).resolve().parent.parent


def main():
    T0 = pd.read_parquet(ROOT / "data/processed/features.parquet").reset_index(drop=True)
    funds = T0["fund"].unique()
    rows = []
    for share in [0.10, 0.25]:
        for noise in [0.0, 0.2]:
            for rep in range(3):
                rng = np.random.default_rng(100 + rep)
                inj = set(rng.choice(funds, max(1, int(round(share * len(funds)))), replace=False))
                T = T0.copy()
                rule = np.where(T["mom4_lag1"].fillna(0) > 0, 2, 0)
                follow = T["fund"].isin(inj).to_numpy() & (rng.random(len(T)) >= noise)
                T.loc[follow, "y"] = rule[follow]
                tr, va, te = RB.train_split(T)
                dtr = lgb.Dataset(tr[BS.FEATURES], tr["y"], categorical_feature=["fund_id"])
                dva = lgb.Dataset(va[BS.FEATURES], va["y"], reference=dtr)
                m = lgb.train(RB.PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
                p = m.predict(te[BS.FEATURES], num_iteration=m.best_iteration).argmax(1)
                maj = RB.majority_pred(tr, te)
                y = te["y"].to_numpy()
                d = pd.DataFrame({"fund": te["fund"].to_numpy(), "m": p == y, "b": maj == y,
                                  "r": rule[te.index] == y}).groupby("fund").mean()
                di = d[d.index.isin(inj)]
                attain = (di["r"] - di["b"]).mean()
                rows.append({"share_of_funds": share, "noise": noise, "rep": rep, "n_injected_funds": len(di),
                             "acc_model": di["m"].mean() * 100, "acc_base": di["b"].mean() * 100,
                             "acc_rule": di["r"].mean() * 100,
                             "recovered_share_pct": 100 * (di["m"] - di["b"]).mean() / attain,
                             "gain_injected": (di["m"] - di["b"]).mean() * 100,
                             "gain_other_funds": (d[~d.index.isin(inj)]["m"] - d[~d.index.isin(inj)]["b"]).mean() * 100})
                print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "experiments" / "injection_test.csv", index=False)
    print(R.groupby(["share_of_funds", "noise"])[["acc_base", "acc_model", "acc_rule", "recovered_share_pct",
                                                  "gain_other_funds"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
