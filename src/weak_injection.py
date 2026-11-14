"""Step 26: weak planted signals (review: is the positive control too easy?).

Same design as the graded injection in learning_curves.py, but with weak rules:
a random 25% of hand-picked funds follow the visible rule (BUY if four-quarter
momentum is positive) for only 5%, 10%, 20% or 30% of their positions (real
labels otherwise), so the attainable gain on those funds is a few points, the
size of the real effect. 3 draws each. Reports the attainable gain (better of
rule and baseline, per fund), the model's gain, and the share recovered.

Output: experiments/weak_injection.csv
Usage:  .venv/bin/python src/weak_injection.py
"""

import numpy as np
import pandas as pd

import learning_curves as LC
import robustness as RB

ROOT = LC.ROOT


def main():
    T0 = pd.read_parquet(ROOT / "data/processed/features.parquet").reset_index(drop=True)
    funds = T0["fund"].unique()
    rule = np.where(T0["mom4_lag1"].fillna(0) > 0, 2, 0)
    rows = []
    for strength in [0.05, 0.10, 0.20, 0.30]:
        for rep in range(3):
            r = np.random.default_rng(300 + rep)
            inj = set(r.choice(funds, int(round(0.25 * len(funds))), replace=False))
            T = T0.copy()
            follow = T["fund"].isin(inj).to_numpy() & (r.random(len(T)) < strength)
            T.loc[follow, "y"] = rule[follow]
            tr, va, te = RB.train_split(T)
            p = LC.fit_predict(tr, va, te)
            maj = RB.majority_pred(tr, te)
            y = te["y"].to_numpy()
            d = pd.DataFrame({"f": te["fund"].to_numpy(), "m": p == y, "b": maj == y,
                              "r": rule[te.index] == y}).groupby("f").mean()
            di, do = d[d.index.isin(inj)], d[~d.index.isin(inj)]
            best = np.maximum(di["r"], di["b"])
            rows.append({"rule_strength": strength, "rep": rep, "attainable_gain": (best - di["b"]).mean() * 100,
                         "model_gain": (di["m"] - di["b"]).mean() * 100,
                         "model_gain_other_funds": (do["m"] - do["b"]).mean() * 100})
            print(rows[-1], flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(ROOT / "experiments" / "weak_injection.csv", index=False)
    print(G.groupby("rule_strength").mean().round(2).to_string())


if __name__ == "__main__":
    main()
