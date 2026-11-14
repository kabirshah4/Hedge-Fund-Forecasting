"""Step 10c: RQ1 and RQ3 by fund style (hand-picked sample; styles from Table 1).

Answers "you pool radically different funds": per style, the RQ1 gain over
the majority baseline and the RQ3 next-quarter spreads (surprise vs predicted
buys and sells), with FF5 + momentum alphas. Styles have few funds, so RQ3
here is descriptive: low power by construction.

Output: experiments/rq3_by_style.csv
Usage:  .venv/bin/python src/rq3_by_style.py
"""

import pandas as pd

import rq3_factors as F3
from robustness import style_groups


def main() -> None:
    P = pd.read_parquet(F3.PROC / "rq1_test_preds.parquet")
    P["correct"] = P["pred_lightgbm"] == P["y"]
    P["style"] = P["fund"].map(style_groups())
    R = F3.monthly_returns()
    F = F3.read_ff("F-F_Research_Data_5_Factors_2x3.csv").join(F3.read_ff("F-F_Momentum_Factor.csv"), how="inner")
    rows = []
    for style, g in P.groupby("style"):
        pf = g.groupby("fund").agg(m=("pred_majority", lambda s: (s == g.loc[s.index, "y"]).mean()),
                                   l=("correct", "mean"))
        W = F3.portfolios(g, quintiles=False)
        ret = {k: F3.portfolio_returns(w, R, [1, 2, 3])["ret"] for k, w in W.items() if k.endswith(("buy", "sell"))}
        out = {"style": style, "funds": g["fund"].nunique(), "test_rows": len(g),
               "rq1_gain_points": (pf["l"].mean() - pf["m"].mean()) * 100}
        for name, y in [("sells", ret["predicted_sell"] - ret["surprise_sell"]),
                        ("buys", ret["surprise_buy"] - ret["predicted_buy"])]:
            a, t, n = F3.alpha(y, F, F3.MODELS["FF5 + momentum"])
            out[f"{name}_alpha_pct"], out[f"{name}_t"], out["months"] = a * 100, t, n
        rows.append(out)
    D = pd.DataFrame(rows).sort_values("funds", ascending=False)
    D.to_csv(F3.EXP / "rq3_by_style.csv", index=False)
    print(D.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
