"""Step 16: why are 13F trades hard to predict? Two candidate mechanisms.

1. Private information. If hedge funds trade on information the public lacks,
   their trades should be hardest to predict in opaque stocks (here: the
   smaller stocks of the 500-stock universe, which have less coverage), and
   more so than for long-only managers. Test: the model's gain over the
   majority baseline in large / middle / small thirds of the universe
   (size_rank at t-1), per sample, and the small-minus-large difference.
2. Aggregation. A 13F nets the trades of every portfolio in a firm. If
   netting blurs predictability, firms with more positions (more
   sub-portfolios) should be less predictable. Test: Spearman correlation
   across funds between the gain and the fund's median number of holdings.

Uncertainty: resample whole funds (2,000 draws).

Output: experiments/mechanism.csv
Usage:  .venv/bin/python src/mechanism.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {"hedge funds, hand-picked": "processed", "hedge funds, Form ADV": "processed_adv",
           "long-only": "processed_longonly"}
rng = np.random.default_rng(3)


def per_fund(P):
    g = P.groupby("fund").apply(lambda d: pd.Series({
        "gain": ((d["pred_lightgbm"] == d["y"]).mean() - (d["pred_majority"] == d["y"]).mean()) * 100,
        "n": len(d)}), include_groups=False)
    return g


def boot_mean(x, B=2000):
    x = np.asarray(x)
    return np.percentile(rng.choice(x, size=(B, len(x)), replace=True).mean(1), [2.5, 97.5])


def main():
    rows, diffs = [], {}
    for name, folder in SAMPLES.items():
        proc = ROOT / "data" / folder
        P = pd.read_parquet(proc / "rq1_test_preds.parquet")
        T = pd.read_parquet(proc / "features.parquet", columns=["fund", "cusip", "q", "size_rank", "n_holdings"])
        P = P.merge(T, on=["fund", "cusip", "q"], how="left")
        P["size_third"] = pd.cut(P["size_rank"], [0, 167, 334, 10_000], labels=["large", "middle", "small"])
        by = {}
        for third in ["large", "middle", "small"]:
            g = per_fund(P[P["size_third"] == third])
            g = g[g["n"] >= 20]
            by[third] = g["gain"]
            lo, hi = boot_mean(g["gain"])
            rows.append({"sample": name, "test": f"gain in {third} stocks", "value": g["gain"].mean(),
                         "ci_low": lo, "ci_high": hi, "funds": len(g)})
        d = (by["small"] - by["large"]).dropna()
        diffs[name] = d
        lo, hi = boot_mean(d)
        rows.append({"sample": name, "test": "small minus large (within fund)", "value": d.mean(),
                     "ci_low": lo, "ci_high": hi, "funds": len(d)})
        g = per_fund(P)
        nh = P.groupby("fund")["n_holdings"].median()
        rho = g["gain"].corr(nh.reindex(g.index), method="spearman")
        draws = []
        idx = np.arange(len(g))
        for _ in range(2000):
            s = rng.choice(idx, len(idx), replace=True)
            draws.append(pd.Series(g["gain"].to_numpy()[s]).corr(pd.Series(nh.reindex(g.index).to_numpy()[s]), method="spearman"))
        rows.append({"sample": name, "test": "Spearman: gain vs. fund's median holdings", "value": rho,
                     "ci_low": np.nanpercentile(draws, 2.5), "ci_high": np.nanpercentile(draws, 97.5), "funds": len(g)})
    for hf in ["hedge funds, hand-picked", "hedge funds, Form ADV"]:
        a, b = diffs[hf].to_numpy(), diffs["long-only"].to_numpy()
        dd = rng.choice(a, (2000, len(a))).mean(1) - rng.choice(b, (2000, len(b))).mean(1)
        rows.append({"sample": f"{hf} minus long-only", "test": "small-minus-large difference",
                     "value": a.mean() - b.mean(), "ci_low": np.percentile(dd, 2.5), "ci_high": np.percentile(dd, 97.5)})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "experiments" / "mechanism.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
