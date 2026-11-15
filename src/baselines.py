"""Step 5b: baselines for RQ1 (how predictable are hedge fund trades?).

Split by time, never shuffled:
  train       2013 Q3 - 2020 Q4
  validation  2021            (used only to stop LightGBM training early)
  test        2022 Q1 - 2026 Q2

Predictors
  1. majority   each fund's most common action in training (Mimicking
                Finance's "naive" baseline, 52% for mutual funds)
  2. repeat     whatever the fund did to this stock last quarter (falls back
                to the fund's majority action when there is no last action)
  3. lightgbm   gradient-boosted trees on the features from build_features.py

Metrics (test period)
  accuracy           share of correct predictions over all rows
  macro-F1           average F1 over BUY / HOLD / SELL (so ignoring the rare
                     HOLD class can't look good)
  mean fund accuracy accuracy computed per fund, then averaged across funds:
                     the number Mimicking Finance reports (71% vs 52%)

Outputs:
  experiments/rq1_baselines.csv, experiments/rq1_per_fund.csv,
  experiments/rq1_feature_importance.csv

Usage:
  .venv/bin/python src/baselines.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
PROC = SMP.PROC
EXP = SMP.EXP

TRAIN_END = pd.Timestamp(SMP.TRAIN_END)  # per sample, see sample.py
VAL_END = pd.Timestamp(SMP.VAL_END)
FEATURES = [
    "act_lag1", "act_lag2", "act_lag3", "holding_age", "log_weight", "rank_in_fund",
    "n_holdings", "fund_changed", "fund_buy_share", "fund_hist_mean_action",
    "ret_lag1", "mom4_lag1", "size_rank", "peer_n", "peer_buy_share", "peer_sell_share",
    "epu_lag1", "fund_id",
]
NAMES = np.array(["SELL", "HOLD", "BUY"])


def scores(y: np.ndarray, p: np.ndarray, fund: pd.Series) -> dict:
    per_fund = pd.Series(y == p).groupby(fund.to_numpy()).mean()
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "mean_fund_accuracy": per_fund.mean(),
    }


def main() -> None:
    T = pd.read_parquet(PROC / "features.parquet")
    train = T[T["period"] <= TRAIN_END]
    val = T[(T["period"] > TRAIN_END) & (T["period"] <= VAL_END)]
    test = T[T["period"] > VAL_END]
    print(f"rows  train {len(train):,}  val {len(val):,}  test {len(test):,}")
    print("test class share:", test["label"].value_counts(normalize=True).round(3).to_dict())

    results, preds = [], {}

    # 1. majority class per fund (from training data only)
    majority = train.groupby("fund")["y"].agg(lambda s: s.value_counts().idxmax())
    overall = int(train["y"].value_counts().idxmax())
    p_major = test["fund"].map(majority).fillna(overall).astype(int).to_numpy()
    preds["majority"] = p_major

    # 2. repeat the fund's last action on this stock
    p_repeat = test["act_lag1"].fillna(pd.Series(p_major, index=test.index)).astype(int).to_numpy()
    preds["repeat"] = p_repeat

    # 3. LightGBM
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
    dtrain = lgb.Dataset(train[FEATURES], train["y"], categorical_feature=["fund_id"])
    dval = lgb.Dataset(val[FEATURES], val["y"], categorical_feature=["fund_id"], reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    proba = model.predict(test[FEATURES], num_iteration=model.best_iteration)
    preds["lightgbm"] = proba.argmax(axis=1)

    for name, p in preds.items():
        results.append({"model": name, **scores(test["y"].to_numpy(), p, test["fund"])})
    R = pd.DataFrame(results)
    EXP.mkdir(exist_ok=True)
    R.to_csv(EXP / "rq1_baselines.csv", index=False)

    # Per-fund accuracy, majority vs LightGBM
    pf = pd.DataFrame({
        "fund": test["fund"].to_numpy(),
        "correct_majority": preds["majority"] == test["y"].to_numpy(),
        "correct_lightgbm": preds["lightgbm"] == test["y"].to_numpy(),
    }).groupby("fund").agg(rows=("correct_lightgbm", "size"),
                           majority=("correct_majority", "mean"),
                           lightgbm=("correct_lightgbm", "mean")).sort_values("lightgbm")
    pf.to_csv(EXP / "rq1_per_fund.csv")

    # Per-row test predictions, for RQ3 (do unpredicted trades earn more?)
    out = test[["fund", "cusip", "q", "period", "y", "label"]].copy()
    out["pred_lightgbm"] = preds["lightgbm"]
    out["pred_majority"] = preds["majority"]
    out[["p_sell", "p_hold", "p_buy"]] = proba
    out.to_parquet(PROC / "rq1_test_preds.parquet", index=False)

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    (imp / imp.sum()).round(4).to_csv(EXP / "rq1_feature_importance.csv", header=["gain_share"])

    print(f"\nLightGBM stopped at {model.best_iteration} rounds")
    print("\nTEST RESULTS (2022 Q1 - 2026 Q2)")
    print(R.round(3).to_string(index=False))
    print("\nper-fund LightGBM accuracy: "
          f"min {pf['lightgbm'].min():.2f}, median {pf['lightgbm'].median():.2f}, max {pf['lightgbm'].max():.2f}")
    print("hardest 5 funds to predict:"); print(pf.head(5).round(3).to_string())
    print("easiest 5:"); print(pf.tail(5).round(3).to_string())
    print("\ntop features (share of total gain):"); print((imp / imp.sum()).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
