"""Step 30: yearly expanding-window retraining (fifth review, W6).

The main model is trained once (through 2020, early stopping on 2021) and
tested on 2022Q1-2026Q2. Here, for each test year Y, LightGBM is retrained on
all labels up to the end of Y-2, early-stopped on Y-1, and tested on Y; for
Y = 2022 this is the main split. Labels for Y-1 are public by the time
predictions for Y are made (each quarter's 13F is filed within 45 days), so
no future information is used. The per-fund majority baseline is re-estimated
on the same expanding training window.

Reports, over the whole test period: mean per-fund accuracy of the yearly
model and of the yearly baseline, the gain with a fund bootstrap CI, errors
fixed, and the yearly model's change versus the main (static) model with a
Diebold-Mariano test.

Output: <experiments>/yearly_retrain.csv
Usage:  SAMPLE=... .venv/bin/python src/yearly_retrain.py
"""

import numpy as np
import pandas as pd

import baselines as BS
import dm_test as DM
import robustness as RB
import sample as SMP


def main() -> None:
    rng = np.random.default_rng(13)
    T = pd.read_parquet(SMP.PROC / "features.parquet")
    static = pd.read_parquet(SMP.PROC / "rq1_test_preds.parquet")[["fund", "cusip", "q", "pred_lightgbm"]]
    parts = []
    for Y in range(2022, 2027):
        if Y == 2022:
            tr, va, _ = RB.train_split(T)
        else:
            tr = T[T["period"] <= pd.Timestamp(f"{Y - 2}-12-31")]
            va = T[(T["period"] > pd.Timestamp(f"{Y - 2}-12-31")) & (T["period"] <= pd.Timestamp(f"{Y - 1}-12-31"))]
        te = T[(T["period"] > pd.Timestamp(f"{Y - 1}-12-31")) & (T["period"] <= pd.Timestamp(f"{Y}-12-31"))].copy()
        if te.empty:
            continue
        m, it = RB.fit(tr, va, BS.FEATURES, RB.PARAMS)
        te["pred_yearly"] = m.predict(te[BS.FEATURES], num_iteration=it).argmax(1)
        te["base_yearly"] = RB.majority_pred(pd.concat([tr, va]), te)
        parts.append(te[["fund", "cusip", "q", "y", "pred_yearly", "base_yearly"]])
        print(Y, len(tr), len(va), len(te), flush=True)
    D = pd.concat(parts).merge(static, on=["fund", "cusip", "q"], how="inner")
    y = D["y"].to_numpy()
    D["b"], D["m"], D["s"] = D["base_yearly"] == y, D["pred_yearly"] == y, D["pred_lightgbm"] == y
    f = D.groupby("fund")[["b", "m", "s"]].mean()
    g = (f["m"] - f["b"]).to_numpy() * 100
    bs = rng.choice(g, (2000, len(g))).mean(1)
    D["diff"] = D["m"].astype(float) - D["s"]
    _, t, p = DM.dm(DM.quarterly(D, "diff"))
    R = pd.DataFrame([{"sample": SMP.SAMPLE, "test_rows": len(D), "acc_base_yearly": f["b"].mean() * 100,
                       "acc_yearly": f["m"].mean() * 100, "acc_static": f["s"].mean() * 100, "gain": g.mean(),
                       "gain_lo": np.percentile(bs, 2.5), "gain_hi": np.percentile(bs, 97.5),
                       "err_fixed": (f["m"].mean() - f["b"].mean()) / (1 - f["b"].mean()) * 100,
                       "yearly_minus_static": (f["m"] - f["s"]).mean() * 100, "dm_t_vs_static": t, "dm_p_vs_static": p}])
    R.to_csv(SMP.EXP / "yearly_retrain.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
