"""Step 23: two-way bootstrap and reference gains (third review, W3, W6).

  two-way bootstrap  resample funds AND quarters independently (pigeonhole
                     bootstrap), so the interval reflects correlation across
                     funds within a quarter and across quarters within a fund
  news-window gain   LightGBM's gain over the majority baseline within the
                     2022-2023 news years only (yardstick for Table 5)

Output: experiments/inference_extra.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(31)
SAMPLES = {"M": "processed", "A": "processed_adv", "L": "processed_longonly", "N": "processed_nport",
           "P": "processed_nportparent"}


def main():
    rows = []
    for k, folder in SAMPLES.items():
        P = pd.read_parquet(ROOT / "data" / folder / "rq1_test_preds.parquet")
        P["d"] = (P["pred_lightgbm"] == P["y"]).astype(float) - (P["pred_majority"] == P["y"]).astype(float)
        cell = P.groupby(["fund", "q"])["d"].agg(["sum", "size"]).reset_index()
        funds, qs = cell["fund"].unique(), cell["q"].unique()
        fi = {f: i for i, f in enumerate(funds)}
        qi = {q: i for i, q in enumerate(qs)}
        S = np.zeros((len(funds), len(qs)))
        N = np.zeros((len(funds), len(qs)))
        S[cell["fund"].map(fi), cell["q"].map(qi)] = cell["sum"]
        N[cell["fund"].map(fi), cell["q"].map(qi)] = cell["size"]
        def stat(fw, qw):
            s = (S * qw[None, :]).sum(1)
            n = (N * qw[None, :]).sum(1)
            ok = n > 0
            return np.average(s[ok] / n[ok], weights=fw[ok]) * 100
        point = stat(np.ones(len(funds)), np.ones(len(qs)))
        draws = []
        for _ in range(2000):
            fw = np.bincount(rng.integers(0, len(funds), len(funds)), minlength=len(funds)).astype(float)
            qw = np.bincount(rng.integers(0, len(qs), len(qs)), minlength=len(qs)).astype(float)
            draws.append(stat(fw, qw))
        rows.append({"sample": k, "test": "two-way bootstrap (funds x quarters)", "gain": point,
                     "lo": np.percentile(draws, 2.5), "hi": np.percentile(draws, 97.5)})
        if k == "M":
            W = P[P["period"] <= "2023-12-31"]
            g = W.groupby("fund")["d"].mean() * 100
            d = rng.choice(g.to_numpy(), (2000, len(g))).mean(1)
            rows.append({"sample": k, "test": "gain within 2022-2023", "gain": g.mean(),
                         "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "experiments" / "inference_extra.csv", index=False)


if __name__ == "__main__":
    main()
