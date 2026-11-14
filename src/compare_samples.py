"""Step 15b: is the hedge-fund vs long-only difference in predictability real?

Per-fund gains (LightGBM minus majority baseline, test period) from each
sample's rq1_per_fund.csv. Bootstrap: resample funds within each sample
independently (5,000 draws) and take the difference in mean gain and in
error reduction ((mean model - mean base) / (1 - mean base)). Also reports the
long-only sample without the three ETF sponsors that pass the active-share
rule only because sector ETFs differ from the market (Van Eck, Rafferty,
ALPS).

Output: experiments/compare_samples.csv
Usage:  .venv/bin/python src/compare_samples.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(5)
ETF_SPONSORS = ("Van Eck", "Rafferty", "Alps Advisors")


def load(folder):
    return pd.read_csv(ROOT / folder / "rq1_per_fund.csv")


def stats(d):
    m, b = d["lightgbm"].mean(), d["majority"].mean()
    return (m - b) * 100, (m - b) / (1 - b) * 100


def boot(d, B=5000):
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m, b = d["lightgbm"].to_numpy()[idx].mean(1), d["majority"].to_numpy()[idx].mean(1)
    return (m - b) * 100, (m - b) / (1 - b) * 100


def main():
    S = {"hedge funds, hand-picked": load("experiments"), "hedge funds, Form ADV": load("experiments_adv"),
         "long-only": load("experiments_longonly")}
    if (ROOT / "experiments_nport" / "rq1_per_fund.csv").exists():
        S["mutual funds, N-PORT"] = load("experiments_nport")
        fp = pd.read_csv(ROOT / "data" / "nport_fund_parent.csv")
        S["mutual funds in multi-fund families"] = S["mutual funds, N-PORT"][S["mutual funds, N-PORT"]["fund"].isin(fp["fund"])]
    if (ROOT / "experiments_nportparent" / "rq1_per_fund.csv").exists():
        S["fund families (funds added up)"] = load("experiments_nportparent")
    lo = S["long-only"]
    S["long-only without ETF sponsors"] = lo[~lo["fund"].str.contains("|".join(ETF_SPONSORS), case=False)]
    rows = []
    for k, d in S.items():
        g, e = stats(d)
        bg, be = boot(d)
        rows.append({"sample": k, "funds": len(d), "gain_points": g, "gain_lo": np.percentile(bg, 2.5),
                     "gain_hi": np.percentile(bg, 97.5), "error_reduction_pct": e,
                     "err_lo": np.percentile(be, 2.5), "err_hi": np.percentile(be, 97.5)})
    for hf in ["hedge funds, hand-picked", "hedge funds, Form ADV"]:
        for lk in ["long-only", "long-only without ETF sponsors"]:
            bg1, be1 = boot(S[lk])
            bg2, be2 = boot(S[hf])
            dg, de = bg1 - bg2, be1 - be2
            rows.append({"sample": f"DIFF {lk} minus {hf}",
                         "gain_points": stats(S[lk])[0] - stats(S[hf])[0], "gain_lo": np.percentile(dg, 2.5),
                         "gain_hi": np.percentile(dg, 97.5),
                         "error_reduction_pct": stats(S[lk])[1] - stats(S[hf])[1],
                         "err_lo": np.percentile(de, 2.5), "err_hi": np.percentile(de, 97.5),
                         "share_draws_long_only_higher": float((de > 0).mean())})
    pairs = [("mutual funds, N-PORT", "hedge funds, hand-picked"),
             ("mutual funds in multi-fund families", "fund families (funds added up)")]
    for a_, b_ in pairs:
        if a_ in S and b_ in S:
            bg1, be1 = boot(S[a_]); bg2, be2 = boot(S[b_])
            dg, de = bg1 - bg2, be1 - be2
            rows.append({"sample": f"DIFF {a_} minus {b_}", "gain_points": stats(S[a_])[0] - stats(S[b_])[0],
                         "gain_lo": np.percentile(dg, 2.5), "gain_hi": np.percentile(dg, 97.5),
                         "error_reduction_pct": stats(S[a_])[1] - stats(S[b_])[1],
                         "err_lo": np.percentile(de, 2.5), "err_hi": np.percentile(de, 97.5)})
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "experiments" / "compare_samples.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
