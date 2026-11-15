"""Step 17: RQ1 checks requested by the second blind review.

For each sample (SAMPLE=main/adv/longonly), on the main split:
  1. fund-weighted LightGBM: every fund carries equal total weight in
     training (weight 1/n_fund), matching the per-fund evaluation metric (W5)
  2. recent-majority baseline: the fund's most common action over its last
     8 quarters of labels (all its positions), a stronger simple rule (W6)
  3. oracle: predict, for every position of a fund-quarter, the fund's
     ACTUAL net direction that quarter (BUY if it bought more positions than
     it sold); an upper bound on what knowing the fund-wide component gives (W6)
  4. LightGBM + oracle feature (the fund's actual quarter-t buy share), to
     split errors into fund-wide scaling vs. stock selection (W6)
  5. LightGBM without peer features: peer sets differ across samples (W6)
For the main sample also: foundation models' BUY-vs-SELL AUC from their raw
scores, which does not depend on the collapsed cut-offs (W3/Q13).

Output: <experiments>/rq1_extra.csv
Usage:  SAMPLE=... .venv/bin/python src/rq1_extra.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import baselines as BS
import robustness as RB
import sample as SMP

PEERS = ["peer_n", "peer_buy_share", "peer_sell_share"]


def fit(train, val, feats, weight=None):
    dtr = lgb.Dataset(train[feats], train["y"], weight=weight, categorical_feature=["fund_id"])
    dva = lgb.Dataset(val[feats], val["y"], reference=dtr)
    m = lgb.train(RB.PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.best_iteration


def main() -> None:
    T = pd.read_parquet(SMP.PROC / "features.parquet")
    L = pd.read_parquet(SMP.PROC / "labels.parquet", columns=["fund", "period", "label"])
    L["q"] = L["period"].dt.year * 4 + (L["period"].dt.month - 1) // 3
    L["y"] = L["label"].map({"SELL": 0, "HOLD": 1, "BUY": 2})
    tr, va, te = RB.train_split(T)
    te = te.copy()
    rows = []

    def add(name, pred, sub=te):
        y = sub["y"].to_numpy()
        pf = pd.Series(pred == y).groupby(sub["fund"].to_numpy()).mean()
        base = pd.Series(RB.majority_pred(tr, sub) == y).groupby(sub["fund"].to_numpy()).mean()
        g = (pf - base) * 100
        rows.append({"model": name, "mean_fund_accuracy": pf.mean(), "median_fund_gain": g.median(),
                     "mean_fund_gain": g.mean(), "accuracy": float((pred == y).mean()),
                     "macro_f1": f1_score(y, pred, average="macro")})

    add("majority (per fund, training years)", RB.majority_pred(tr, te))

    # 2. recent majority: fund's most common label over quarters t-8..t-1
    counts = L.groupby(["fund", "q", "y"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    counts = counts.sort_index()
    roll = counts.groupby(level=0).apply(lambda d: d.droplevel(0).rolling(8, min_periods=1).sum().shift(1))
    roll = roll[roll.sum(axis=1) > 0]  # a fund's first quarter has no history: falls back below
    rec = roll.idxmax(axis=1).rename("recent").reset_index()
    rec.columns = ["fund", "q", "recent"]
    te = te.merge(rec, on=["fund", "q"], how="left")
    fb = pd.Series(RB.majority_pred(tr, te), index=te.index)
    add("recent majority (last 8 quarters)", te["recent"].fillna(fb).astype(int).to_numpy())

    # 1. LightGBM, standard and fund-weighted
    m, it = fit(tr, va, BS.FEATURES)
    add("LightGBM", m.predict(te[BS.FEATURES], num_iteration=it).argmax(1))
    w = 1.0 / tr.groupby("fund")["fund"].transform("size").to_numpy()
    m, it = fit(tr, va, BS.FEATURES, weight=w * len(w) / w.sum())
    add("LightGBM, fund-weighted training", m.predict(te[BS.FEATURES], num_iteration=it).argmax(1))

    # 5. no peer features
    nf = [f for f in BS.FEATURES if f not in PEERS]
    m, it = fit(tr, va, nf)
    add("LightGBM without peer features", m.predict(te[nf], num_iteration=it).argmax(1))

    # 3./4. oracle: the fund's actual quarter-t direction across its predicted positions
    def oracle_share(d):
        g = d.groupby(["fund", "q"])["y"]
        return g.transform(lambda s: (s == 2).mean()), g.transform(lambda s: (s == 0).mean())
    bs, ss = oracle_share(te)
    add("oracle: fund's actual net direction at t", np.where(bs >= ss, 2, 0))
    for d in (tr, va, te):
        b, s = oracle_share(d)
        d["oracle_buy_share"] = b.to_numpy()
    of = BS.FEATURES + ["oracle_buy_share"]
    m, it = fit(tr, va, of)
    add("LightGBM + oracle fund buy share at t", m.predict(te[of], num_iteration=it).argmax(1))

    # foundation-model AUCs (main sample only)
    if SMP.IS_MAIN:
        key = ["fund", "cusip", "q"]
        bsub = te[te["label"] != "HOLD"][key + ["label"]]
        isbuy = lambda d: (d["label"] == "BUY").astype(int)
        for name, path, col in [("Chronos-2 (raw score)", "chronos_preds.parquet", "chronos_A_score"),
                                ("TimesFM 2.5 (raw score)", "timesfm_scores_all.parquet", "timesfm_score"),
                                ("Chronos finance-trained (raw score)", "fintext_scores.parquet", "fintext_score")]:
            S = pd.read_parquet(SMP.PROC / path)[key + [col]]
            d = bsub.merge(S, on=key).dropna()
            rows.append({"model": name, "auc_buy_sell": roc_auc_score(isbuy(d), d[col]), "auc_rows": len(d)})
        P = pd.read_parquet(SMP.PROC / "rq1_test_preds.parquet")
        pb = P[P["label"] != "HOLD"]
        rows.append({"model": "LightGBM (p_buy - p_sell)", "auc_buy_sell": roc_auc_score(isbuy(pb), pb["p_buy"] - pb["p_sell"]),
                     "auc_rows": len(pb)})
    R = pd.DataFrame(rows)
    R.to_csv(SMP.EXP / "rq1_extra.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
