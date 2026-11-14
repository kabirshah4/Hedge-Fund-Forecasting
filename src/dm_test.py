"""Step 25: Diebold-Mariano tests of forecast accuracy (is the gain luck?).

For each sample, compare each model's test-period forecasts with the per-fund
majority baseline, quarter by quarter:

  d_t = mean over funds active in quarter t of
        [fund's model accuracy in t - fund's baseline accuracy in t]

(funds weighted equally, matching the paper's mean per-fund accuracy), and the
same for log-loss (baseline = each fund's training class frequencies, as in
rq1_metrics.py). The DM statistic is mean(d) / sqrt(LRV / T) with a Newey-West
long-run variance (1 lag), multiplied by the Harvey-Leybourne-Newbold
small-sample correction for one-step forecasts, sqrt((T - 1) / T), and compared
with a t-distribution with T - 1 degrees of freedom (T = 18 test quarters).

Models: LightGBM (all five samples); zero-shot Chronos-2, TimesFM 2.5 and the
finance-trained Chronos (hand-picked sample; labels from their saved scores and
the 2021 cut-offs, as in reeval_chronos.py / run_tsfm_extra.py).

Output: experiments/dm_tests.csv
Usage:  .venv/bin/python src/dm_test.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import run_chronos as RC

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {"M": ("processed", "2020-12-31"), "A": ("processed_adv", "2020-12-31"),
           "L": ("processed_longonly", "2020-12-31"), "N": ("processed_nport", "2021-06-30"),
           "P": ("processed_nportparent", "2021-06-30")}
LAGS = 1


def dm(d: np.ndarray, lags: int = LAGS) -> tuple[float, float, float]:
    """Return (mean, HLN-corrected DM t, two-sided p) for a loss-differential series."""
    T = len(d)
    u = d - d.mean()
    lrv = (u @ u) / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T
    t = d.mean() / np.sqrt(lrv / T) * np.sqrt((T - 1) / T)
    return d.mean(), t, 2 * stats.t.sf(abs(t), T - 1)


def quarterly(df: pd.DataFrame, col: str) -> np.ndarray:
    """Fund-equal-weighted quarterly mean of a per-position differential."""
    return df.groupby(["q", "fund"])[col].mean().groupby(level=0).mean().sort_index().to_numpy()


def base_probs(folder: str, train_end: str, P: pd.DataFrame) -> np.ndarray:
    T = pd.read_parquet(ROOT / "data" / folder / "features.parquet", columns=["fund", "period", "y"])
    tr = T[T["period"] <= pd.Timestamp(train_end)]
    freq = tr.groupby("fund")["y"].value_counts(normalize=True).unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    overall = tr["y"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    b = freq.reindex(P["fund"]).to_numpy()
    b = np.where(np.isnan(b), overall.to_numpy(), b)
    b = np.clip(b, 1e-3, 1)
    return b / b.sum(axis=1, keepdims=True)


def main() -> None:
    rows = []
    for s, (folder, train_end) in SAMPLES.items():
        P = pd.read_parquet(ROOT / "data" / folder / "rq1_test_preds.parquet")
        y = P["y"].to_numpy()
        P["d_acc"] = (P["pred_lightgbm"].to_numpy() == y).astype(float) - (P["pred_majority"].to_numpy() == y)
        model = np.clip(P[["p_sell", "p_hold", "p_buy"]].to_numpy(), 1e-6, 1)
        base = base_probs(folder, train_end, P)
        idx = np.arange(len(y))
        P["d_ll"] = -np.log(base[idx, y]) + np.log(model[idx, y])  # positive = model has lower log-loss
        for metric, col in [("accuracy", "d_acc"), ("log-loss", "d_ll")]:
            d = quarterly(P, col)
            mean, t, p = dm(d)
            rows.append({"sample": s, "model": "LightGBM", "metric": metric, "quarters": len(d),
                         "mean_diff": mean * (100 if metric == "accuracy" else 1), "dm_t": t, "p_two_sided": p,
                         "quarters_positive": int((d > 0).sum())})

        if s == "M":
            key = ["fund", "cusip", "q"]
            F = pd.read_parquet(ROOT / "data" / folder / "features.parquet", columns=key + ["period", "y"])
            for name, path, col in [("Chronos-2", "chronos_preds.parquet", "chronos_A_score"),
                                    ("TimesFM 2.5", "timesfm_scores_all.parquet", "timesfm_score"),
                                    ("Chronos, finance-trained", "fintext_scores.parquet", "fintext_score")]:
                S = pd.read_parquet(ROOT / "data" / folder / path)[key + [col]]
                D = F.merge(S, on=key).dropna(subset=[col])
                D = D[D["period"] > RC.TRAIN_END]
                val, test = D[D["period"] <= RC.VAL_END], D[D["period"] > RC.VAL_END].copy()
                lo, hi = RC.tune(val[col].to_numpy(), val["y"].to_numpy())
                test["pred"] = RC.score_to_label(test[col].to_numpy(), lo, hi)
                test = test.merge(P[key + ["pred_majority"]], on=key)
                test["d_acc"] = (test["pred"] == test["y"]).astype(float) - (test["pred_majority"] == test["y"])
                d = quarterly(test, "d_acc")
                mean, t, p = dm(d)
                rows.append({"sample": s, "model": name, "metric": "accuracy", "quarters": len(d),
                             "mean_diff": mean * 100, "dm_t": t, "p_two_sided": p,
                             "quarters_positive": int((d > 0).sum())})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "experiments" / "dm_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
