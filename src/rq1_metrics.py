"""Step 10b: class-balance-robust RQ1 metrics (answers "apples to oranges").

Accuracy gains are hard to compare across data sets with different class
mixes. So, per fund, then averaged across funds:
  error_reduction   (model acc - baseline acc) / (1 - baseline acc): the share
                    of the baseline's mistakes the model fixes. Mimicking
                    Finance: (71 - 52) / (100 - 52) = 40%.
  logloss_skill     1 - logloss(model) / logloss(baseline probabilities), where
                    the baseline predicts each fund's TRAINING class frequencies
                    (a probabilistic majority guess). 0 = no better.
  auc_ovr           one-vs-rest AUC (BUY, SELL, HOLD), macro average, pooled.
  auc_buy_sell      AUC for BUY vs SELL using p_buy - p_sell, HOLD rows dropped.

Also: does a fund's predictability depend on how many positions it trades?
(Spearman correlation of per-fund gain with the fund's number of test rows.)

Output: <experiments>/rq1_metrics.csv     (SAMPLE=adv for the Form ADV sample)
Usage:  .venv/bin/python src/rq1_metrics.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import baselines as BS
import sample as SMP


def main() -> None:
    P = pd.read_parquet(SMP.PROC / "rq1_test_preds.parquet")
    T = pd.read_parquet(SMP.PROC / "features.parquet", columns=["fund", "period", "y"])
    train = T[T["period"] <= BS.TRAIN_END]
    freq = train.groupby("fund")["y"].value_counts(normalize=True).unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    overall = train["y"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    base = freq.reindex(P["fund"]).to_numpy()
    base = np.where(np.isnan(base), overall.to_numpy(), base)
    base = np.clip(base, 1e-3, 1)
    base = base / base.sum(axis=1, keepdims=True)
    model = np.clip(P[["p_sell", "p_hold", "p_buy"]].to_numpy(), 1e-6, 1)
    y = P["y"].to_numpy()

    rows = []
    for f, idx in P.groupby("fund").indices.items():
        yy = y[idx]
        acc_m = (P["pred_lightgbm"].to_numpy()[idx] == yy).mean()
        acc_b = (P["pred_majority"].to_numpy()[idx] == yy).mean()
        ll_m = log_loss(yy, model[idx], labels=[0, 1, 2])
        ll_b = log_loss(yy, base[idx], labels=[0, 1, 2])
        rows.append({"fund": f, "rows": len(idx), "acc_model": acc_m, "acc_base": acc_b,
                     "error_reduction": (acc_m - acc_b) / (1 - acc_b) if acc_b < 1 else np.nan,
                     "logloss_skill": 1 - ll_m / ll_b})
    F = pd.DataFrame(rows)
    bs = P["label"] != "HOLD"
    out = {
        "funds": len(F),
        "mean_fund_acc_model": F["acc_model"].mean(), "mean_fund_acc_base": F["acc_base"].mean(),
        "error_reduction_of_means": (F["acc_model"].mean() - F["acc_base"].mean()) / (1 - F["acc_base"].mean()),
        "error_reduction_median_fund": F["error_reduction"].median(),
        "logloss_skill_mean_fund": F["logloss_skill"].mean(),
        "logloss_skill_median_fund": F["logloss_skill"].median(),
        "logloss_skill_pooled": 1 - log_loss(y, model, labels=[0, 1, 2]) / log_loss(y, base, labels=[0, 1, 2]),
        "auc_ovr_macro": roc_auc_score(y, model / model.sum(axis=1, keepdims=True), multi_class="ovr", average="macro"),
        "auc_buy_sell": roc_auc_score((P.loc[bs, "label"] == "BUY").astype(int), (P.loc[bs, "p_buy"] - P.loc[bs, "p_sell"])),
        "spearman_gain_vs_rows": F["acc_model"].sub(F["acc_base"]).corr(F["rows"], method="spearman"),
    }
    SMP.EXP.mkdir(exist_ok=True)
    pd.DataFrame([out]).to_csv(SMP.EXP / "rq1_metrics.csv", index=False)
    F.to_csv(SMP.EXP / "rq1_metrics_per_fund.csv", index=False)
    for k, v in out.items():
        print(f"{k:32s} {v:.3f}" if isinstance(v, float) else f"{k:32s} {v}")


if __name__ == "__main__":
    main()
