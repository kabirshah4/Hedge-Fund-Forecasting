"""Step 28: does adding firm characteristics make trades more predictable? (review W1)

For the current SAMPLE, train the same LightGBM (fixed settings, early stopping
on validation) on BS.FEATURES and on BS.FEATURES + firm characteristics
(firm_chars.py: fundamentals from SEC XBRL, price-based characteristics), and
compare on the test period:
  - mean per-fund accuracy, gain over the per-fund majority baseline, its fund
    bootstrap CI (2,000 draws), share of baseline errors fixed;
  - the extra gain from firm characteristics, with a fund bootstrap CI and a
    Diebold-Mariano test on quarterly fund-averaged accuracy differences;
  - the same on the subset of test positions that have fundamentals, so that
    missing data (more common for delisted firms) cannot drive the result.

Output: <experiments>/rq1_firmchars.csv
Usage:  SAMPLE=... .venv/bin/python src/rq1_firmchars.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

import baselines as BS
import dm_test as DM
import firm_chars as FC
import robustness as RB
import sample as SMP


def fit_predict(tr, va, te, feats):
    dtr = lgb.Dataset(tr[feats], tr["y"], categorical_feature=["fund_id"])
    dva = lgb.Dataset(va[feats], va["y"], reference=dtr)
    m = lgb.train(RB.PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(te[feats], num_iteration=m.best_iteration).argmax(1)


def boot(x: np.ndarray, rng) -> tuple[float, float]:
    d = rng.choice(x, (2000, len(x))).mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def evaluate(te, base, p0, p1, label, rng):
    y = te["y"].to_numpy()
    d = pd.DataFrame({"fund": te["fund"].to_numpy(), "q": te["q"].to_numpy(),
                      "b": base == y, "m0": p0 == y, "m1": p1 == y})
    f = d.groupby("fund")[["b", "m0", "m1"]].mean()
    g0, g1 = (f["m0"] - f["b"]).to_numpy() * 100, (f["m1"] - f["b"]).to_numpy() * 100
    extra = g1 - g0
    d["diff"] = d["m1"].astype(float) - d["m0"]
    q = DM.quarterly(d, "diff")
    _, t, p = DM.dm(q)
    lo0, hi0 = boot(g0, rng); lo1, hi1 = boot(g1, rng); loe, hie = boot(extra, rng)
    err = lambda col: (f[col].mean() - f["b"].mean()) / (1 - f["b"].mean()) * 100
    return {"subset": label, "test_rows": len(te), "funds": len(f), "acc_base": f["b"].mean() * 100,
            "acc_model": f["m0"].mean() * 100, "acc_model_firm": f["m1"].mean() * 100,
            "gain": g0.mean(), "gain_lo": lo0, "gain_hi": hi0, "err_fixed": err("m0"),
            "gain_firm": g1.mean(), "gain_firm_lo": lo1, "gain_firm_hi": hi1, "err_fixed_firm": err("m1"),
            "extra": extra.mean(), "extra_lo": loe, "extra_hi": hie, "dm_t_extra": t, "dm_p_extra": p,
            "quarters_extra_positive": int((q > 0).sum())}


def main() -> None:
    rng = np.random.default_rng(7)
    T = pd.read_parquet(SMP.PROC / "features.parquet")
    F = pd.read_parquet(SMP.ROOT / "data" / "processed" / "firm_chars.parquet")
    T = T.merge(F, on=["cusip", "q"], how="left")
    tr, va, te = RB.train_split(T)
    feats0, feats1 = BS.FEATURES, BS.FEATURES + FC.FEATURES
    p0, p1 = fit_predict(tr, va, te, feats0), fit_predict(tr, va, te, feats1)
    base = RB.majority_pred(tr, te)
    has = te[FC.FUND_FEATS].notna().any(axis=1).to_numpy()
    rows = [evaluate(te, base, p0, p1, "all test positions", rng),
            evaluate(te[has], base[has], p0[has], p1[has], "positions with fundamentals", rng)]
    R = pd.DataFrame(rows)
    R.insert(0, "sample", SMP.SAMPLE)
    R["coverage_fundamentals"] = has.mean()
    R["coverage_price"] = te[FC.PRICE_FEATS].notna().any(axis=1).mean()
    R.to_csv(SMP.EXP / "rq1_firmchars.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).T.to_string())


if __name__ == "__main__":
    main()
