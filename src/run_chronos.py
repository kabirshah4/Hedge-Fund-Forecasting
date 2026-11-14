"""Step 7: Chronos-2, zero-shot, on each position's share history (RQ2).

For every prediction row in features.parquet (a fund held the stock at the end
of quarter t-1), give Chronos-2 the position's split-adjusted share count over
the quarters it has been held continuously, up to t-1, and ask for t.

Chronos-2 is never trained on our data ("zero-shot"). It returns 9 forecast
quantiles. From them we read:
  P(up)   = share of the forecast distribution above +1%
  P(down) = share below -1%
and turn score = P(up) - P(down) into a label with two cut-offs, chosen on the
2021 validation quarters only. That is the only thing tuned.

Input sets (adding one at a time, like the LightGBM comparison):
  A  own share history only
  B  + the stock's quarterly returns (past covariate)
  C  + news sentiment per quarter (needs FinBERT scores; skipped if missing)
  D  + policy uncertainty (EPU)

Outputs:
  experiments/rq2_chronos.csv          test scores per input set
  data/processed/chronos_preds.parquet per-row forecasts (for the hybrid)

Usage:
  .venv/bin/python src/run_chronos.py            # all input sets available
  .venv/bin/python src/run_chronos.py --sets A B
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from sklearn.metrics import accuracy_score, f1_score

import build_features as BF
import build_labels as BL

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EXP = ROOT / "experiments"
MODEL = "amazon/chronos-2"
CONTEXT = 32  # quarters of history at most
TRAIN_END = pd.Timestamp("2020-12-31")
VAL_END = pd.Timestamp("2021-12-31")
QLEVELS = np.arange(0.1, 1.0, 0.1)


def share_history() -> pd.DataFrame:
    """Split-adjusted shares per (fund, cusip, q), from the full holdings."""
    H = pd.read_parquet(PROC / "holdings.parquet", columns=["fund", "period", "cusip", "shares"])
    H = H[H["period"].dt.is_quarter_end].copy()
    H["q"] = BF.qn(H["period"])
    H = H.groupby(["fund", "cusip", "q"], as_index=False)["shares"].sum()
    # cumulative split product per stock: shares / C_q puts every quarter in the same units
    k = BL.real_split_factors().rename("k").reset_index()
    k = k.sort_values(["cusip", "q"])
    k["C"] = k.groupby("cusip")["k"].cumprod()
    H["q"] = H["q"].astype("int64")
    k["q"] = k["q"].astype("int64")
    H = pd.merge_asof(H.sort_values("q"), k[["cusip", "q", "C"]].sort_values("q"),
                      on="q", by="cusip", direction="backward")
    H["C"] = H["C"].fillna(1.0)
    H["x"] = H["shares"] / H["C"]
    H = H.sort_values(["fund", "cusip", "q"])
    H["run"] = H.groupby(["fund", "cusip"])["q"].diff().ne(1).cumsum()
    return H[["fund", "cusip", "q", "x", "run"]]


def covariate_table(sets: list[str]) -> pd.DataFrame:
    """Quarterly covariates per (cusip, q)."""
    cov = BF.stock_returns()[["cusip", "q", "ret"]]
    if "C" in sets:
        S = pd.concat([pd.read_parquet(p) for p in sorted((PROC / "news_scores").glob("*.parquet"))])
        news = S.groupby(["cusip", "quarter"])["sentiment"].mean().rename("news").reset_index()
        cov = cov.merge(news.rename(columns={"quarter": "q"}), on=["cusip", "q"], how="left")
    if "D" in sets:
        e = pd.read_excel(ROOT / "data" / "raw" / "epu" / "US_Policy_Uncertainty_Data.xlsx")
        e["Year"] = pd.to_numeric(e["Year"], errors="coerce")
        e = e.dropna(subset=["Year", "Month"])
        e["q"] = e["Year"].astype(int) * 4 + (e["Month"].astype(int) - 1) // 3
        epu = e.groupby("q")["News_Based_Policy_Uncert_Index"].mean().rename("epu").reset_index()
        cov = cov.merge(epu, on="q", how="left")
    return cov


def build_inputs(rows: pd.DataFrame, H: pd.DataFrame, cov: pd.DataFrame, cov_names: list[str]):
    """One Chronos input per prediction row: history up to t-1 in the current run.

    Works on plain numpy arrays per (fund, cusip) for speed (a pandas filter per
    row took 30 minutes for 120k rows)."""
    Hc = H.merge(cov, on=["cusip", "q"], how="left") if cov_names else H
    Hc = Hc.sort_values(["fund", "cusip", "q"])
    cols = ["q", "x", "run"] + cov_names
    arrays = {}
    for key, g in Hc.groupby(["fund", "cusip"], sort=False):
        arrays[key] = {c: g[c].to_numpy() for c in cols}
    inputs, keep = [], []
    for i, (fund, cusip, q) in enumerate(zip(rows["fund"], rows["cusip"], rows["q"])):
        a = arrays.get((fund, cusip))
        if a is None:
            continue
        end = np.searchsorted(a["q"], q - 1, side="right")  # rows with quarter <= t-1
        if end == 0 or a["q"][end - 1] != q - 1:
            continue
        run = a["run"][end - 1]
        start = end - 1
        while start > 0 and a["run"][start - 1] == run and end - start < CONTEXT:
            start -= 1
        x = a["x"][start:end].astype(np.float32)
        target = x / x[-1]  # 1.0 at t-1
        if cov_names:
            past = {c: np.nan_to_num(a[c][start:end].astype(np.float32)) for c in cov_names}
            inputs.append({"target": target, "past_covariates": past})
        else:
            inputs.append(target)
        keep.append(i)
    return inputs, np.array(keep)


def score_to_label(score: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(score > hi, 2, np.where(score < lo, 0, 1))


def tune(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best, cut = -1, (0.0, 0.0)
    grid = np.linspace(-0.9, 0.9, 37)
    for lo in grid:
        for hi in grid[grid >= lo]:
            acc = accuracy_score(y, score_to_label(score, lo, hi))
            if acc > best:
                best, cut = acc, (lo, hi)
    return cut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None)
    args = ap.parse_args()
    have_news = any((PROC / "news_scores").glob("*.parquet"))
    sets = args.sets or (["A", "B", "C", "D"] if have_news else ["A", "B", "D"])
    if "C" in sets and not have_news:
        sets.remove("C")
        print("no FinBERT scores yet: skipping input set C")

    T = pd.read_parquet(PROC / "features.parquet", columns=["fund", "cusip", "q", "period", "y"])
    rows = T[T["period"] > TRAIN_END].reset_index(drop=True)  # validation + test
    H = share_history()
    cov_all = covariate_table(sets)
    device = "cpu"  # the GPU is busy with FinBERT; Chronos-2 is small enough for the CPU
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map=device)

    cov_for = {"A": [], "B": ["ret"], "C": ["ret", "news"], "D": ["ret", "news", "epu"] if "C" in sets else ["ret", "epu"]}
    results, all_preds = [], rows[["fund", "cusip", "q", "period", "y"]].copy()
    for s in sets:
        names = cov_for[s]
        t0 = time.time()
        inputs, keep = build_inputs(rows, H, cov_all[["cusip", "q"] + names] if names else cov_all, names)
        qs, _ = pipe.predict_quantiles(inputs, prediction_length=1, quantile_levels=list(QLEVELS), batch_size=256)
        qmat = torch.stack([q[0, 0, :] for q in qs]).numpy()  # (n, 9) ratios vs t-1
        p_up = (qmat > 1.01).mean(axis=1)
        p_down = (qmat < 0.99).mean(axis=1)
        score = p_up - p_down
        sub = rows.iloc[keep].copy()
        sub["score"] = score
        val = sub[sub["period"] <= VAL_END]
        test = sub[sub["period"] > VAL_END]
        lo, hi = tune(val["score"].to_numpy(), val["y"].to_numpy())
        pred = score_to_label(test["score"].to_numpy(), lo, hi)
        y = test["y"].to_numpy()
        per_fund = pd.Series(pred == y).groupby(test["fund"].to_numpy()).mean()
        res = {
            "input_set": s, "covariates": "+".join(["own history"] + names),
            "test_rows": len(test), "coverage": len(keep) / len(rows),
            "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "mean_fund_accuracy": per_fund.mean(), "cutoffs": f"{lo:.2f}/{hi:.2f}",
            "minutes": (time.time() - t0) / 60,
        }
        results.append(res)
        all_preds.loc[sub.index, f"chronos_{s}_score"] = score
        print(f"set {s}: acc {res['accuracy']:.3f}  macroF1 {res['macro_f1']:.3f}  "
              f"fund-acc {res['mean_fund_accuracy']:.3f}  coverage {res['coverage']:.2f}  "
              f"({res['minutes']:.1f} min)")

    EXP.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(EXP / "rq2_chronos.csv", index=False)
    all_preds.to_parquet(PROC / "chronos_preds.parquet", index=False)
    print(pd.DataFrame(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
