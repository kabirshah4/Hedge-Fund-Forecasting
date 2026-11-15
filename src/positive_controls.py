"""Step 22: can the pipeline find predictability where it exists? (third review, W1, W2)

1. Synthetic injection (hand-picked sample). For a random share p of ALL
   positions (training, validation and test alike), replace the real label
   with one set by a rule the model can see: BUY if the stock's four-quarter
   momentum at t-1 is positive, else SELL. A perfect learner would then gain
   about p x (100% - baseline accuracy on those rows) points; we report how
   much of that the pipeline recovers.
2. Earlier period. Train 2013Q3-2015Q4, validate 2016, test 2017Q1-2019Q4,
   which overlaps Cohen, Lu & Nguyen's sample years, for the 13F samples.
3. Tuning budget. 20 random LightGBM configurations, chosen on 2021
   validation log-loss, then evaluated once on the test period.

Output: experiments/positive_controls.csv
Usage:  .venv/bin/python src/positive_controls.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import baselines as BS
import robustness as RB

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(21)
SAMPLES = {"hedge funds, hand-picked": "processed", "hedge funds, Form ADV": "processed_adv",
           "long-only managers": "processed_longonly"}


def fit_eval(tr, va, te, params=RB.PARAMS):
    dtr = lgb.Dataset(tr[BS.FEATURES], tr["y"], categorical_feature=["fund_id"])
    dva = lgb.Dataset(va[BS.FEATURES], va["y"], reference=dtr)
    m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict(te[BS.FEATURES], num_iteration=m.best_iteration).argmax(1)
    maj = RB.majority_pred(tr, te)
    y = te["y"].to_numpy()
    pf = pd.DataFrame({"m": p == y, "b": maj == y, "fund": te["fund"].to_numpy()}).groupby("fund").mean()
    g = (pf["m"] - pf["b"]).to_numpy() * 100
    d = rng.choice(g, (2000, len(g))).mean(1)
    base = pf["b"].mean()
    return {"base": base * 100, "model": pf["m"].mean() * 100, "gain": g.mean(),
            "gain_lo": np.percentile(d, 2.5), "gain_hi": np.percentile(d, 97.5),
            "error_reduction_pct": (pf["m"].mean() - base) / (1 - base) * 100,
            "val_logloss": m.best_score["valid_0"]["multi_logloss"]}


def split(T, tr_end, va_end, te_end=None, tr_start=None):
    tr = T[(T["period"] <= tr_end) & ((T["period"] >= tr_start) if tr_start is not None else True)]
    va = T[(T["period"] > tr_end) & (T["period"] <= va_end)]
    te = T[T["period"] > va_end]
    if te_end is not None:
        te = te[te["period"] <= te_end]
    return tr, va, te


def main():
    rows = []
    T = pd.read_parquet(ROOT / "data/processed/features.parquet").reset_index(drop=True)
    tr, va, te = RB.train_split(T)
    # 1. synthetic injection
    for p in [0.1, 0.25, 0.5]:
        U = T.copy()
        mask = rng.random(len(U)) < p
        rule = np.where(U["mom4_lag1"].fillna(0) > 0, 2, 0)
        U.loc[mask, "y"] = rule[mask]
        a, b, c = RB.train_split(U)
        r = fit_eval(a, b, c)
        inj = mask[c.index]
        maj = RB.majority_pred(a, c)
        ceiling = (1 - (maj[inj] == c["y"].to_numpy()[inj]).mean()) * inj.mean() * 100
        rows.append({"check": "synthetic injection", "sample": "hedge funds, hand-picked",
                     "setting": f"{int(p * 100)}% of labels set by a visible rule", "ceiling_gain": ceiling, **r})
        print(rows[-1], flush=True)
    for name, folder in SAMPLES.items():
        T = pd.read_parquet(ROOT / "data" / folder / "features.parquet")
        # 2. earlier period
        a, b, c = split(T, pd.Timestamp("2015-12-31"), pd.Timestamp("2016-12-31"), pd.Timestamp("2019-12-31"))
        rows.append({"check": "earlier period", "sample": name, "setting": "train 2013Q3-2015, test 2017-2019",
                     **fit_eval(a, b, c)})
        print(rows[-1], flush=True)
        # 3. tuning budget on the main split
        a, b, c = RB.train_split(T)
        best = None
        for _ in range(20):
            prm = {**RB.PARAMS, "num_leaves": int(rng.choice([7, 15, 31, 63, 127, 255])),
                   "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.1])),
                   "min_data_in_leaf": int(rng.choice([20, 50, 100, 200, 500, 1000])),
                   "feature_fraction": float(rng.choice([0.5, 0.7, 0.8, 1.0])),
                   "lambda_l2": float(rng.choice([0.0, 1.0, 5.0, 20.0]))}
            dtr = lgb.Dataset(a[BS.FEATURES], a["y"], categorical_feature=["fund_id"])
            dva = lgb.Dataset(b[BS.FEATURES], b["y"], reference=dtr)
            mdl = lgb.train(prm, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            vl = mdl.best_score["valid_0"]["multi_logloss"]
            if best is None or vl < best[0]:
                best = (vl, prm)
        r = fit_eval(a, b, c, best[1])
        rows.append({"check": "tuned (20 configurations on 2021)", "sample": name,
                     "setting": f"leaves {best[1]['num_leaves']}, lr {best[1]['learning_rate']}, "
                                f"min leaf {best[1]['min_data_in_leaf']}", **r})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "experiments" / "positive_controls.csv", index=False)


if __name__ == "__main__":
    main()
