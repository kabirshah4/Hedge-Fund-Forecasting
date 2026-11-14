"""Step 10 (Phase 2): RQ1 robustness checks.

1. Mimicking Finance protocol (the fair comparison).
   Cohen, Lu & Nguyen train a separate model PER FUND on rolling 28-quarter
   windows: 20 quarters to train, the next 8 to test. Their baseline is the
   fund's most common action in its training window. We copy that: four
   non-overlapping 8-quarter test blocks (2018Q3-2026Q2), each trained on the
   20 quarters before it, with
     a. one LightGBM per fund (fixed small settings, no tuning), and
     b. one pooled LightGBM across funds on the same window,
   against the per-fund majority. Score: mean per-fund accuracy per block.

2. BUY vs SELL only: drop HOLD (2.7% of rows) and predict the direction.

3. Wider HOLD band: relabel the TARGET with +-5% instead of +-1%
   (features unchanged), so small rebalancing counts as HOLD.

4. Breakdowns on the main test predictions: by fund style, and without
   Berkshire (our non-hedge-fund comparison).

Outputs: experiments/rq1_robustness.csv, experiments/rq1_by_style.csv

Usage: .venv/bin/python src/robustness.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import baselines as BS
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
PROC = SMP.PROC
EXP = SMP.EXP

PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
SMALL = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15,
             min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
NO_ID = [f for f in BS.FEATURES if f != "fund_id"]


def majority_pred(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    maj = train.groupby("fund")["y"].agg(lambda s: s.value_counts().idxmax())
    overall = int(train["y"].value_counts().idxmax())
    return test["fund"].map(maj).fillna(overall).astype(int).to_numpy()


def fund_mean_acc(test: pd.DataFrame, pred: np.ndarray) -> float:
    return pd.Series(pred == test["y"].to_numpy()).groupby(test["fund"].to_numpy()).mean().mean()


def train_split(T):
    return (T[T["period"] <= BS.TRAIN_END],
            T[(T["period"] > BS.TRAIN_END) & (T["period"] <= BS.VAL_END)],
            T[T["period"] > BS.VAL_END])


def fit(train, val, feats, params, rounds=2000):
    dtr = lgb.Dataset(train[feats], train["y"], categorical_feature=[f for f in ["fund_id"] if f in feats])
    if val is None:
        return lgb.train(params, dtr, rounds), None
    dva = lgb.Dataset(val[feats], val["y"], reference=dtr)
    m = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.best_iteration


def mimicking_protocol(T: pd.DataFrame) -> list[dict]:
    qs = sorted(T["q"].unique())
    rows = []
    for b in range(20, len(qs) - 7, 8):
        tr_q, te_q = qs[b - 20:b], qs[b:b + 8]
        tr, te = T[T["q"].isin(tr_q)], T[T["q"].isin(te_q)]
        maj = majority_pred(tr, te)
        pooled, _ = fit(tr, None, BS.FEATURES, PARAMS, rounds=300)
        p_pool = pooled.predict(te[BS.FEATURES]).argmax(axis=1)
        p_fund = maj.copy()
        for f, g in te.groupby("fund"):
            gtr = tr[tr["fund"] == f]
            if len(gtr) < 200 or gtr["y"].nunique() < 2:
                continue  # too little history: keep the majority guess
            m, _ = fit(gtr, None, NO_ID, SMALL, rounds=200)
            idx = te.index.get_indexer(g.index)
            p_fund[idx] = m.predict(g[NO_ID]).argmax(axis=1)
        y0, q0 = divmod(int(te_q[0]), 4)
        y1, q1 = divmod(int(te_q[-1]), 4)
        block = f"{y0}Q{q0 + 1}-{y1}Q{q1 + 1}"
        for name, p in [("majority (per fund)", maj), ("LightGBM per fund", p_fund), ("LightGBM pooled", p_pool)]:
            rows.append({"check": "Mimicking Finance protocol", "variant": block, "model": name,
                         "mean_fund_accuracy": fund_mean_acc(te, p), "accuracy": float((p == te["y"]).mean()),
                         "rows": len(te), "funds": te["fund"].nunique()})
    return rows


def buy_vs_sell(T: pd.DataFrame) -> list[dict]:
    B = T[T["label"] != "HOLD"].copy()
    B["y"] = (B["label"] == "BUY").astype(int)
    tr, va, te = train_split(B)
    params = {**PARAMS, "objective": "binary"}
    params.pop("num_class")
    m, it = fit(tr, va, BS.FEATURES, params)
    pr = m.predict(te[BS.FEATURES], num_iteration=it)
    p = (pr > 0.5).astype(int)
    maj = majority_pred(tr, te)
    return [
        {"check": "BUY vs SELL only", "variant": "test 2022-2026", "model": "majority (per fund)",
         "mean_fund_accuracy": fund_mean_acc(te, maj), "accuracy": float((maj == te["y"]).mean()), "rows": len(te)},
        {"check": "BUY vs SELL only", "variant": "test 2022-2026", "model": "LightGBM",
         "mean_fund_accuracy": fund_mean_acc(te, p), "accuracy": float((p == te["y"]).mean()),
         "auc": roc_auc_score(te["y"], pr), "rows": len(te)},
    ]


def wide_hold(T: pd.DataFrame, band: float = 0.05) -> list[dict]:
    L = pd.read_parquet(PROC / "labels.parquet", columns=["fund", "cusip", "period", "change_pct"])
    W = T.merge(L, on=["fund", "cusip", "period"], how="left")
    c = W["change_pct"]
    W["y"] = np.where(c > band, 2, np.where(c < -band, 0, 1))
    tr, va, te = train_split(W)
    m, it = fit(tr, va, BS.FEATURES, PARAMS)
    p = m.predict(te[BS.FEATURES], num_iteration=it).argmax(axis=1)
    maj = majority_pred(tr, te)
    share = pd.Series(te["y"]).map({0: "SELL", 1: "HOLD", 2: "BUY"}).value_counts(normalize=True).round(3).to_dict()
    out = []
    for name, pr in [("majority (per fund)", maj), ("LightGBM", p)]:
        out.append({"check": f"HOLD band +-{band:.0%}", "variant": f"test classes {share}", "model": name,
                    "mean_fund_accuracy": fund_mean_acc(te, pr), "accuracy": float((pr == te["y"]).mean()),
                    "macro_f1": f1_score(te["y"], pr, average="macro"), "rows": len(te)})
    return out


def style_groups() -> dict[str, str]:
    """Fund -> the 5 broad groups used in the paper's fund table."""
    import re
    tex = (ROOT / "paper" / "sections" / "table_funds.tex").read_text()
    out = {}
    for g, _, funds in re.findall(r"\n\s*([A-Z][^&\n]+?)\s*&\s*(\d+)\s*&\s*(.+?)\\\\", tex, re.S):
        for f in re.sub(r"\s+", " ", funds).split(","):
            out[f.strip().replace("\\&", "&")] = g.strip()
    out["Berkshire Hathaway"] = "Berkshire (not a hedge fund)"
    return out


def breakdowns() -> tuple[list[dict], pd.DataFrame]:
    P = pd.read_parquet(PROC / "rq1_test_preds.parquet")
    F = pd.read_csv(SMP.FUND_LIST)[["fund", "style", "include"]]
    P = P.merge(F, on="fund", how="left")
    P["style"] = P["fund"].map(style_groups()).fillna(P["style"])
    P["c_lgb"] = P["pred_lightgbm"] == P["y"]
    P["c_maj"] = P["pred_majority"] == P["y"]
    pf = P.groupby(["fund", "style", "include"])[["c_lgb", "c_maj"]].mean().reset_index()
    style = pf.groupby("style").agg(funds=("fund", "size"), majority=("c_maj", "mean"),
                                   lightgbm=("c_lgb", "mean")).reset_index()
    style["gain_points"] = (style["lightgbm"] - style["majority"]) * 100
    hf = pf[pf["include"] == "yes"]
    rows = [{"check": "Hedge funds only (drop Berkshire)", "variant": "main test predictions", "model": m,
             "mean_fund_accuracy": hf[c].mean(), "funds": len(hf)}
            for m, c in [("majority (per fund)", "c_maj"), ("LightGBM", "c_lgb")]]
    return rows, style


def main() -> None:
    T = pd.read_parquet(PROC / "features.parquet")
    rows = []
    print("1. Mimicking Finance protocol (per-fund rolling windows)...")
    rows += mimicking_protocol(T)
    print("2. BUY vs SELL only...")
    rows += buy_vs_sell(T)
    print("3. wider HOLD band...")
    rows += wide_hold(T)
    print("4. breakdowns...")
    r4, style = breakdowns()
    rows += r4
    R = pd.DataFrame(rows)
    EXP.mkdir(exist_ok=True)
    R.to_csv(EXP / "rq1_robustness.csv", index=False)
    style.to_csv(EXP / "rq1_by_style.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    show = R.copy()
    for c in ["mean_fund_accuracy", "accuracy"]:
        show[c] = (show[c] * 100).round(1)
    print(show.drop(columns=[c for c in ["rows"] if c in show]).to_string(index=False))
    mf = R[R["check"] == "Mimicking Finance protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100
    print("\nMimicking protocol, averaged over the 4 blocks:", mf.round(1).to_dict())
    print("\nby style:"); print(style.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
