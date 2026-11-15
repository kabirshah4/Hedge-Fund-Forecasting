"""Step 20b: the same short training window for every sample.

N-PORT starts in late 2019, so its models train on 2020Q1-2021Q2 only (six
quarters) and validate on 2021H2. To compare like with like, this reruns the
13F samples with exactly that window (same test period 2022Q1-2026Q2), and
reports the gain of LightGBM over each fund's majority action (majority
computed on the same six training quarters), with 95% intervals from
resampling funds.

Output: experiments/short_training.csv
Usage:  .venv/bin/python src/short_training.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import baselines as BS
import robustness as RB

ROOT = Path(__file__).resolve().parent.parent
TR0, TR1, VA1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30"), pd.Timestamp("2021-12-31")
rng = np.random.default_rng(9)
SAMPLES = {"hedge funds, hand-picked": "processed", "hedge funds, Form ADV": "processed_adv",
           "long-only managers": "processed_longonly", "mutual funds, N-PORT": "processed_nport",
           "fund families, N-PORT": "processed_nportparent"}


def main():
    rows = []
    for name, folder in SAMPLES.items():
        f = ROOT / "data" / folder / "features.parquet"
        if not f.exists():
            continue
        T = pd.read_parquet(f)
        tr = T[(T["period"] >= TR0) & (T["period"] <= TR1)]
        va = T[(T["period"] > TR1) & (T["period"] <= VA1)]
        te = T[T["period"] > VA1]
        dtr = lgb.Dataset(tr[BS.FEATURES], tr["y"], categorical_feature=["fund_id"])
        dva = lgb.Dataset(va[BS.FEATURES], va["y"], reference=dtr)
        m = lgb.train(RB.PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(te[BS.FEATURES], num_iteration=m.best_iteration).argmax(1)
        maj = RB.majority_pred(tr, te)
        y = te["y"].to_numpy()
        pf = pd.DataFrame({"m": p == y, "b": maj == y, "fund": te["fund"].to_numpy()}).groupby("fund").mean()
        g = (pf["m"] - pf["b"]).to_numpy() * 100
        draws = rng.choice(g, (2000, len(g))).mean(1)
        base = pf["b"].mean()
        rows.append({"sample": name, "funds": len(pf), "base": base * 100, "model": pf["m"].mean() * 100,
                     "gain": g.mean(), "gain_lo": np.percentile(draws, 2.5), "gain_hi": np.percentile(draws, 97.5),
                     "error_reduction_pct": (pf["m"].mean() - base) / (1 - base) * 100})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "experiments" / "short_training.csv", index=False)


if __name__ == "__main__":
    main()
