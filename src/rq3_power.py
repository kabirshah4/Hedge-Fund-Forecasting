"""Step 12c: RQ3 power, alternative surprise definition, and the cross-sample test.

Answers reviewer W2/Q6, W5b, W5c, W5e:
  1. For each sample's next-quarter spreads (FF5 + momentum alphas): 95% CI,
     the minimum detectable effect at 80% power (2.8 x standard error), and
     whether we can reject an effect the size of Cohen, Lu & Nguyen's
     stock-level FF5 alpha (3.86% a year = 0.32% a month).
  2. "Surprise" defined against the fund's majority action instead of the
     LightGBM prediction (main sample).
  3. Cross-sample: is the hand-picked minus Form ADV difference significant?
     Monthly difference of the two samples' spreads, regressed on the factors
     (Newey-West). Plus the hand-picked sample split into funds that are and
     are not also in the Form ADV sample.
  4. Portfolio sizes behind the by-style estimates (stocks per month).

Output: experiments/rq3_power.csv (and printed tables)
Usage:  .venv/bin/python src/rq3_power.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import rq3_factors as F3
from robustness import style_groups

ROOT = Path(__file__).resolve().parent.parent
MF_ALPHA_MONTHLY = 3.86 / 12  # % per month, Cohen, Lu & Nguyen stock-level FF5 alpha
FF = F3.MODELS["FF5 + momentum"]


def preds(sample: str) -> pd.DataFrame:
    proc = ROOT / "data" / ("processed" if sample == "main" else f"processed_{sample}")
    P = pd.read_parquet(proc / "rq1_test_preds.parquet")
    P["correct"] = P["pred_lightgbm"] == P["y"]
    return P


def spreads(P: pd.DataFrame, R: pd.DataFrame, quintiles=True) -> dict[str, pd.Series]:
    W = F3.portfolios(P, quintiles=quintiles)
    ret = {k: F3.portfolio_returns(w, R, [1, 2, 3])["ret"] for k, w in W.items()}
    out = {"sells": ret["predicted_sell"] - ret["surprise_sell"], "buys": ret["surprise_buy"] - ret["predicted_buy"]}
    if quintiles:
        out["Q1 - Q5"] = ret["Q1"] - ret["Q5"]
    return out


def row(label, spread, y, F):
    a, t, n = F3.alpha(y, F, FF)
    a, se = a * 100, abs(a * 100 / t) if t else np.nan
    return {"test": label, "spread": spread, "alpha_pct_month": a, "t": t, "months": n,
            "ci_low": a - 1.96 * se, "ci_high": a + 1.96 * se, "mde80": 2.8 * se,
            "rejects_mutual_fund_size": (a + 1.96 * se) < MF_ALPHA_MONTHLY}


def main() -> None:
    R = F3.monthly_returns()
    F = F3.read_ff("F-F_Research_Data_5_Factors_2x3.csv").join(F3.read_ff("F-F_Momentum_Factor.csv"), how="inner")
    rows, sp = [], {}
    samples = [s for s in ["main", "adv", "longonly"]
               if (ROOT / "data" / ("processed" if s == "main" else f"processed_{s}") / "rq1_test_preds.parquet").exists()]
    for s in samples:
        sp[s] = spreads(preds(s), R)
        for k, y in sp[s].items():
            rows.append(row(f"{s} sample", k, y, F))

    P = preds("main")
    Pm = P.assign(correct=P["pred_majority"] == P["y"])
    for k, y in spreads(Pm, R).items():
        rows.append(row("main, surprise = against the fund's majority action", k, y, F))

    if "adv" in sp:
        for k in ["sells", "buys", "Q1 - Q5"]:
            rows.append(row("difference: hand-picked minus Form ADV", k, sp["main"][k] - sp["adv"][k], F))
        fl = pd.read_csv(ROOT / "data" / "fund_list.csv", dtype=str)
        fa = pd.read_csv(ROOT / "data" / "fund_list_adv.csv", dtype=str)
        adv_ciks = {c.lstrip("0") for c in fa["ciks"]}
        overlap = {r.fund for r in fl.itertuples() if any(c.lstrip("0") in adv_ciks for c in str(r.ciks).split(";"))}
        for name, part in [("hand-picked, funds also in Form ADV", P[P["fund"].isin(overlap)]),
                           ("hand-picked, funds not in Form ADV", P[~P["fund"].isin(overlap)])]:
            for k, y in spreads(part, R, quintiles=False).items():
                rows.append(row(f"{name} ({part['fund'].nunique()} funds)", k, y, F))

    # portfolio sizes behind the by-style estimates
    P["style"] = P["fund"].map(style_groups())
    sizes = []
    for st, g in P.groupby("style"):
        W = F3.portfolios(g, quintiles=False)
        for k in ["surprise_buy", "predicted_buy", "surprise_sell", "predicted_sell"]:
            sizes.append({"style": st, "portfolio": k, "median_stocks_per_quarter": W[k].groupby("q").size().median()})
    S = pd.DataFrame(sizes).pivot(index="style", columns="portfolio", values="median_stocks_per_quarter")

    out = pd.DataFrame(rows)
    out.to_csv(F3.EXP / "rq3_power.csv", index=False)
    S.to_csv(F3.EXP / "rq3_style_portfolio_sizes.csv")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print("\nmedian stocks per quarter in each style portfolio:")
    print(S.to_string())


if __name__ == "__main__":
    main()
