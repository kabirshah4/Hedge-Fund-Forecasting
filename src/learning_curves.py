"""Step 24: learning curves and graded signal strength (fourth review, W1, W2).

1. Learning curves. For each 13F sample, train LightGBM on only the last K
   quarters before 2021 (K = 6, 12, 20, all ~30), validate on 2021, test on
   2022Q1-2026Q2. If predictability rose steeply with history, the six
   quarters available for N-PORT could explain why mutual funds look
   unpredictable in our pipeline.
2. Graded injection. A random 25% of hand-picked funds follow the visible
   rule (BUY if four-quarter momentum is positive) for only 55%, 60% or 70%
   of their positions (real labels otherwise). We report how much of the
   attainable gain on those funds the pipeline recovers (3 draws each).

Output: experiments/learning_curves.csv, experiments/graded_injection.csv
Usage:  .venv/bin/python src/learning_curves.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import baselines as BS
import robustness as RB

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {"hedge funds, hand-picked": "processed", "hedge funds, Form ADV": "processed_adv",
           "long-only managers": "processed_longonly"}


def fit_predict(tr, va, te):
    dtr = lgb.Dataset(tr[BS.FEATURES], tr["y"], categorical_feature=["fund_id"])
    dva = lgb.Dataset(va[BS.FEATURES], va["y"], reference=dtr)
    m = lgb.train(RB.PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(te[BS.FEATURES], num_iteration=m.best_iteration).argmax(1)


def main():
    rng = np.random.default_rng(41)
    rows = []
    for name, folder in SAMPLES.items():
        T = pd.read_parquet(ROOT / "data" / folder / "features.parquet")
        tr_all, va, te = RB.train_split(T)
        qs = sorted(tr_all["q"].unique())
        for K in [6, 12, 20, len(qs)]:
            tr = tr_all[tr_all["q"].isin(qs[-K:])]
            p = fit_predict(tr, va, te)
            maj = RB.majority_pred(tr, te)
            y = te["y"].to_numpy()
            d = pd.DataFrame({"f": te["fund"].to_numpy(), "m": p == y, "b": maj == y}).groupby("f").mean()
            g = (d["m"] - d["b"]).to_numpy() * 100
            dr = rng.choice(g, (2000, len(g))).mean(1)
            rows.append({"sample": name, "train_quarters": K, "gain": g.mean(), "gain_lo": np.percentile(dr, 2.5),
                         "gain_hi": np.percentile(dr, 97.5),
                         "error_reduction_pct": (d["m"].mean() - d["b"].mean()) / (1 - d["b"].mean()) * 100})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "experiments" / "learning_curves.csv", index=False)

    T0 = pd.read_parquet(ROOT / "data/processed/features.parquet").reset_index(drop=True)
    funds = T0["fund"].unique()
    rule = np.where(T0["mom4_lag1"].fillna(0) > 0, 2, 0)
    inj_rows = []
    for strength in [0.55, 0.60, 0.70]:
        for rep in range(3):
            r = np.random.default_rng(200 + rep)
            inj = set(r.choice(funds, int(round(0.25 * len(funds))), replace=False))
            T = T0.copy()
            follow = T["fund"].isin(inj).to_numpy() & (r.random(len(T)) < strength)
            T.loc[follow, "y"] = rule[follow]
            tr, va, te = RB.train_split(T)
            p = fit_predict(tr, va, te)
            maj = RB.majority_pred(tr, te)
            y = te["y"].to_numpy()
            d = pd.DataFrame({"f": te["fund"].to_numpy(), "m": p == y, "b": maj == y,
                              "r": rule[te.index] == y}).groupby("f").mean()
            di = d[d.index.isin(inj)]
            best = np.maximum(di["r"], di["b"])  # attainable: the better of the rule and the baseline
            inj_rows.append({"rule_strength": strength, "rep": rep, "acc_base": di["b"].mean() * 100,
                             "acc_model": di["m"].mean() * 100, "acc_rule": di["r"].mean() * 100,
                             "attainable_gain": (best - di["b"]).mean() * 100,
                             "model_gain": (di["m"] - di["b"]).mean() * 100})
            print(inj_rows[-1], flush=True)
    G = pd.DataFrame(inj_rows)
    G["recovered_pct"] = 100 * G["model_gain"] / G["attainable_gain"]
    G.to_csv(ROOT / "experiments" / "graded_injection.csv", index=False)
    print(G.groupby("rule_strength")[["acc_base", "acc_model", "acc_rule", "attainable_gain", "model_gain",
                                      "recovered_pct"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
