"""Step 29: predictability of large trades and exits (fifth review, W4).

With a +-1% band, much of a firm's quarterly "trading" may be rebalancing.
For the current SAMPLE (features unchanged, target relabeled):
  1. +-25% band: BUY if shares rise more than 25%, SELL if they fall more than
     25% (including full exits), HOLD otherwise. LightGBM vs the per-fund
     majority baseline: mean per-fund accuracy, gain with fund bootstrap CI,
     share of errors fixed, Diebold-Mariano test.
  2. Full exits only: binary LightGBM (exit vs. not), AUC, and log-loss skill
     relative to each fund's training exit rate.

Output: <experiments>/big_trades.csv
Usage:  SAMPLE=... .venv/bin/python src/big_trades.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import baselines as BS
import dm_test as DM
import robustness as RB
import sample as SMP

BAND = 0.25


def main() -> None:
    rng = np.random.default_rng(11)
    T = pd.read_parquet(SMP.PROC / "features.parquet")
    L = pd.read_parquet(SMP.PROC / "labels.parquet", columns=["fund", "cusip", "period", "change_pct"])
    W = T.merge(L, on=["fund", "cusip", "period"], how="left")
    c = W["change_pct"]
    rows = []

    # 1. +-25% band
    W["y"] = np.where(c > BAND, 2, np.where(c < -BAND, 0, 1))
    tr, va, te = RB.train_split(W)
    m, it = RB.fit(tr, va, BS.FEATURES, RB.PARAMS)
    p = m.predict(te[BS.FEATURES], num_iteration=it).argmax(1)
    base = RB.majority_pred(tr, te)
    y = te["y"].to_numpy()
    d = pd.DataFrame({"fund": te["fund"].to_numpy(), "q": te["q"].to_numpy(), "b": base == y, "m": p == y})
    f = d.groupby("fund")[["b", "m"]].mean()
    g = (f["m"] - f["b"]).to_numpy() * 100
    bs = rng.choice(g, (2000, len(g))).mean(1)
    d["diff"] = d["m"].astype(float) - d["b"]
    _, t, pv = DM.dm(DM.quarterly(d, "diff"))
    share = pd.Series(y).value_counts(normalize=True)
    rows.append({"sample": SMP.SAMPLE, "task": "+-25% band", "test_rows": len(te),
                 "share_sell": share.get(0, 0), "share_hold": share.get(1, 0), "share_buy": share.get(2, 0),
                 "acc_base": f["b"].mean() * 100, "acc_model": f["m"].mean() * 100, "gain": g.mean(),
                 "gain_lo": np.percentile(bs, 2.5), "gain_hi": np.percentile(bs, 97.5),
                 "err_fixed": (f["m"].mean() - f["b"].mean()) / (1 - f["b"].mean()) * 100,
                 "dm_t": t, "dm_p": pv})

    # 2. full exits
    W["y"] = (c <= -0.999).astype(int)
    tr, va, te = RB.train_split(W)
    params = {**RB.PARAMS, "objective": "binary"}
    params.pop("num_class")
    dtr = lgb.Dataset(tr[BS.FEATURES], tr["y"], categorical_feature=["fund_id"])
    dva = lgb.Dataset(va[BS.FEATURES], va["y"], reference=dtr)
    mb = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    pe = mb.predict(te[BS.FEATURES], num_iteration=mb.best_iteration)
    rate = tr.groupby("fund")["y"].mean()
    pb = te["fund"].map(rate).fillna(tr["y"].mean()).clip(1e-3, 1 - 1e-3).to_numpy()
    ye = te["y"].to_numpy()
    rows.append({"sample": SMP.SAMPLE, "task": "full exits", "test_rows": len(te), "share_exit": ye.mean(),
                 "auc": roc_auc_score(ye, pe), "auc_fund_rate": roc_auc_score(ye, pb),
                 "logloss_skill": 1 - log_loss(ye, np.clip(pe, 1e-6, 1 - 1e-6)) / log_loss(ye, pb)})
    R = pd.DataFrame(rows)
    R.to_csv(SMP.EXP / "big_trades.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
