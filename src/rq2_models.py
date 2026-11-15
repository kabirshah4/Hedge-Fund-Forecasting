"""Step 7c: RQ2, do foundation models and news add predictive power?

LightGBM trained five ways, same split as baselines.py, same base features:
  base          features from build_features.py
  +chronos      + the Chronos-2 (zero-shot) forecast for the position (the "hybrid")
  +news_prev    + news about the stock during quarter t-1 (a true FORECAST,
                made at the start of quarter t)
  +news_now     + news during quarter t itself (a NOWCAST made at quarter-end,
                before the 13F filing reveals the trades 45 days later)
  +all          everything

News features per (stock, quarter): number of headlines, average FinBERT
sentiment, share of clearly negative / positive headlines, and the change in
average sentiment from the quarter before.

LEAKAGE FILTER: headlines that report an investor's own stake or filing
("Pershing Square Reports Stake In...", 13F/13D/13G, "boosts position") are
removed before computing news features: they would reveal the very trades we
are predicting.

The news comparisons are scored on 2022-2023 only (news data ends in 2023);
the Chronos comparison on the full test period too.
Significance: bootstrap over funds of (model minus base) mean per-fund accuracy.

Output: experiments/rq2_lightgbm.csv
Usage:  .venv/bin/python src/rq2_models.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import baselines as BS

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EXP = ROOT / "experiments"
NEWS_END = pd.Timestamp("2023-12-31")
STAKE = (r"\bstake\b|\bstakes\b|13F|13D|13G|takes position|boosts position|cuts position|"
         r"raises position|lowers position|new position|holdings in|buys shares of|sells shares of|"
         r"increases stake|decreases stake|exits position")
rng = np.random.default_rng(11)


def news_features(last_days: int | None = None) -> pd.DataFrame:
    """Per (cusip, quarter) news features. last_days: only headlines from the
    last N days of the quarter (fast-moving news, not a 90-day average)."""
    N = pd.read_parquet(PROC / "news_headlines.parquet", columns=["cusip", "date", "quarter", "headline"])
    S = pd.concat([pd.read_parquet(p) for p in sorted((PROC / "news_scores").glob("*.parquet"))])
    # scores were saved in the same order as headlines within each year: join on keys
    N = N.reset_index(drop=True)
    S = S.reset_index(drop=True)
    key = ["cusip", "date", "quarter"]
    N["_k"] = N.groupby(key).cumcount()
    S["_k"] = S.groupby(key).cumcount()
    M = N.merge(S, on=key + ["_k"], how="inner")
    leak = M["headline"].str.contains(STAKE, case=False, regex=True)
    print(f"news: {len(M):,} scored headlines; removed {leak.sum():,} ({leak.mean():.1%}) that report a stake or filing")
    M = M[~leak]
    if last_days:
        q = M["quarter"].astype(int)
        qend = pd.to_datetime(dict(year=q // 4, month=3 * (q % 4) + 3, day=1)) + pd.offsets.MonthEnd(0)
        d = pd.to_datetime(M["date"], utc=True).dt.tz_localize(None)  # news times are UTC
        M = M[(d > qend - pd.Timedelta(days=last_days)).to_numpy()]
        print(f"  last {last_days} days of each quarter: {len(M):,} headlines")
    F = M.groupby(["cusip", "quarter"]).agg(
        news_n=("sentiment", "size"),
        news_sent=("sentiment", "mean"),
        news_neg=("p_neg", lambda s: (s > 0.5).mean()),
        news_pos=("p_pos", lambda s: (s > 0.5).mean()),
    ).reset_index().rename(columns={"quarter": "q"})
    F["news_n"] = np.log1p(F["news_n"])
    F = F.sort_values(["cusip", "q"])
    F["news_sent_chg"] = F.groupby("cusip")["news_sent"].diff()
    return F


def llm_features(kind: str = "named") -> pd.DataFrame | None:
    """LLM judgements per (cusip, quarter) from src/llm_news.py, if they exist."""
    files = sorted((PROC / "llm").glob(f"{kind}_*.jsonl"))
    if not files:
        return None
    L = pd.concat([pd.read_json(f, lines=True) for f in files])
    L = L[L["outlook"].notna()].drop_duplicates(["cusip", "quarter"])
    L = L.rename(columns={"quarter": "q", "outlook": "llm_outlook", "surprise": "llm_surprise"})
    return L[["cusip", "q", "llm_outlook", "llm_surprise"]]


def fit_predict(train, val, test, feats):
    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
    dtr = lgb.Dataset(train[feats], train["y"], categorical_feature=["fund_id"])
    dva = lgb.Dataset(val[feats], val["y"], categorical_feature=["fund_id"], reference=dtr)
    m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(test[feats], num_iteration=m.best_iteration).argmax(axis=1)


def per_fund_acc(df, pred):
    return pd.Series(pred == df["y"].to_numpy()).groupby(df["fund"].to_numpy()).mean()


def boot_ci(df, pred, base_pred, B=2000):
    c = pd.DataFrame({"fund": df["fund"].to_numpy(), "d": (pred == df["y"].to_numpy()).astype(float)
                      - (base_pred == df["y"].to_numpy()).astype(float)})
    by = c.groupby("fund")["d"].mean().to_numpy()
    draws = [rng.choice(by, len(by), replace=True).mean() for _ in range(B)]
    return np.percentile(draws, [2.5, 97.5]) * 100


def main() -> None:
    T = pd.read_parquet(PROC / "features.parquet")
    C = pd.read_parquet(PROC / "chronos_scores_all.parquet")
    T = T.merge(C, on=["fund", "cusip", "q"], how="left")
    tf_path = PROC / "timesfm_scores_all.parquet"
    has_tf = tf_path.exists()
    if has_tf:
        T = T.merge(pd.read_parquet(tf_path), on=["fund", "cusip", "q"], how="left")
    F = news_features()
    prev = F.add_suffix("_prev").rename(columns={"cusip_prev": "cusip", "q_prev": "q"})
    prev["q"] = prev["q"] + 1  # quarter t-1 news, used to predict quarter t
    now = F.add_suffix("_now").rename(columns={"cusip_now": "cusip", "q_now": "q"})
    T = T.merge(prev, on=["cusip", "q"], how="left").merge(now, on=["cusip", "q"], how="left")

    Lf = llm_features()
    llm_prev, llm_now = [], []
    if Lf is not None and len(Lf) >= 10000:  # skip partial / test runs
        # only stock-quarters the LLM has read count; full coverage needs the Thanksgiving run
        print(f"LLM features: {len(Lf):,} stock-quarters")
        lp = Lf.add_suffix("_prev").rename(columns={"cusip_prev": "cusip", "q_prev": "q"})
        lp["q"] = lp["q"] + 1
        ln = Lf.add_suffix("_now").rename(columns={"cusip_now": "cusip", "q_now": "q"})
        T = T.merge(lp, on=["cusip", "q"], how="left").merge(ln, on=["cusip", "q"], how="left")
        llm_prev = ["llm_outlook_prev", "llm_surprise_prev"]
        llm_now = ["llm_outlook_now", "llm_surprise_now"]

    FL = news_features(last_days=15)
    lprev = FL.add_suffix("_lprev").rename(columns={"cusip_lprev": "cusip", "q_lprev": "q"})
    lprev["q"] = lprev["q"] + 1
    lnow = FL.add_suffix("_lnow").rename(columns={"cusip_lnow": "cusip", "q_lnow": "q"})
    T = T.merge(lprev, on=["cusip", "q"], how="left").merge(lnow, on=["cusip", "q"], how="left")
    late_prev = [c for c in T.columns if c.endswith("_lprev")]
    late_now = [c for c in T.columns if c.endswith("_lnow")]

    news_prev = [c for c in T.columns if c.endswith("_prev") and c.startswith("news")]
    news_now = [c for c in T.columns if c.endswith("_now") and c.startswith("news")]
    sets = {
        "base": BS.FEATURES,
        "+chronos": BS.FEATURES + ["chronos_score", "chronos_median"],
        "+news_prev": BS.FEATURES + news_prev,
        "+news_now": BS.FEATURES + news_now,
        "+all": BS.FEATURES + ["chronos_score", "chronos_median"] + news_prev + news_now,
    }
    if has_tf:
        sets["+timesfm"] = BS.FEATURES + ["timesfm_score", "timesfm_median"]
    sets["+news_late15_prev"] = BS.FEATURES + late_prev
    sets["+news_late15_now"] = BS.FEATURES + late_now
    if llm_prev:
        sets["+llm_prev"] = BS.FEATURES + llm_prev
        sets["+llm_now"] = BS.FEATURES + llm_now
    train = T[T["period"] <= BS.TRAIN_END]
    val = T[(T["period"] > BS.TRAIN_END) & (T["period"] <= BS.VAL_END)]
    test = T[T["period"] > BS.VAL_END]
    news_test = test["period"] <= NEWS_END
    print(f"test rows {len(test):,}; in the news window (2022-2023) {news_test.sum():,}; "
          f"with any news that quarter: {test.loc[news_test, 'news_n_now'].notna().mean():.0%}")

    preds = {name: fit_predict(train, val, test, feats) for name, feats in sets.items()}
    rows = []
    for window, mask in [("2022-2026Q2 (full test)", np.ones(len(test), bool)), ("2022-2023 (news window)", news_test.to_numpy())]:
        sub = test[mask]
        for name, p in preds.items():
            if window.startswith("2022-2026") and ("news" in name or "llm" in name or name == "+all"):
                continue
            pm, pb = p[mask], preds["base"][mask]
            lo, hi = boot_ci(sub, pm, pb) if name != "base" else (np.nan, np.nan)
            rows.append({
                "window": window, "model": name,
                "accuracy": accuracy_score(sub["y"], pm), "macro_f1": f1_score(sub["y"], pm, average="macro"),
                "mean_fund_accuracy": per_fund_acc(sub, pm).mean(),
                "gain_vs_base_points": (per_fund_acc(sub, pm).mean() - per_fund_acc(sub, pb).mean()) * 100,
                "ci95_low": lo, "ci95_high": hi,
            })
    R = pd.DataFrame(rows)
    EXP.mkdir(exist_ok=True)
    R.to_csv(EXP / "rq2_lightgbm.csv", index=False)
    pd.set_option("display.width", 160)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
