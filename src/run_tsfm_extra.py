"""Step 7d (Phase 2): two more foundation models, so RQ2 does not rest on Chronos-2 alone.

1. TimesFM 2.5 (Google, 200M parameters), zero-shot, on exactly the inputs
   Chronos-2 got: each position's split-adjusted share history up to t-1,
   normalized to 1 at t-1. Scored for EVERY row, so it can also be a
   LightGBM feature (hybrid).
2. A finance-trained Chronos: FinText/Chronos_Small_2020_US (Rahimikia, Ni &
   Wang), pre-trained from scratch on U.S. excess returns 1990-2020 only, so it
   has never seen our validation (2021) or test (2022+) years. It learned on
   returns, so it gets the position's quarter-to-quarter log changes in shares
   and forecasts the next change. Validation + test rows only (it is slower).

Scoring is the same as for Chronos-2: score = P(up > 1%) - P(down > 1%) from
the forecast quantiles, turned into BUY/HOLD/SELL with two cut-offs chosen on
2021 only.

Outputs:
  experiments/rq2_tsfm_extra.csv
  data/processed/timesfm_scores_all.parquet  (fund, cusip, q, timesfm_score, timesfm_median)
  data/processed/fintext_scores.parquet      (validation + test rows; resumable)

Usage: .venv/bin/python src/run_tsfm_extra.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import run_chronos as RC

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EXP = ROOT / "experiments"
UP, DOWN = np.log(1.01), np.log(0.99)


def evaluate(name: str, rows: pd.DataFrame, keep: np.ndarray, score: np.ndarray, minutes: float) -> dict:
    sub = rows.iloc[keep].copy()
    sub["score"] = score
    sub = sub[sub["period"] > RC.TRAIN_END]
    val, test = sub[sub["period"] <= RC.VAL_END], sub[sub["period"] > RC.VAL_END]
    lo, hi = RC.tune(val["score"].to_numpy(), val["y"].to_numpy())
    pred = RC.score_to_label(test["score"].to_numpy(), lo, hi)
    y = test["y"].to_numpy()
    pf = pd.Series(pred == y).groupby(test["fund"].to_numpy()).mean()
    r = {"model": name, "test_rows": len(test), "accuracy": accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro"), "mean_fund_accuracy": pf.mean(),
         "cutoffs": f"{lo:.2f}/{hi:.2f}", "share_predicted_hold": float((pred == 1).mean()), "minutes": minutes}
    print(f"{name}: fund-acc {r['mean_fund_accuracy']:.3f}  acc {r['accuracy']:.3f}  macroF1 {r['macro_f1']:.3f}  "
          f"cut-offs {r['cutoffs']}  ({minutes:.1f} min)", flush=True)
    return r


def run_timesfm(rows, inputs, keep):
    path = PROC / "timesfm_scores_all.parquet"
    t0 = time.time()
    if not path.exists():  # forecasts depend only on each position's history: compute once
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
        m.compile(timesfm.ForecastConfig(max_context=RC.CONTEXT, max_horizon=128, normalize_inputs=True,
                                         use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                         per_core_batch_size=512))
        _, q = m.forecast(horizon=1, inputs=[np.asarray(x, dtype=np.float64) for x in inputs])
        dec = np.log(np.clip(q[:, 0, 1:], 1e-6, None))  # 9 deciles, log ratio vs t-1 (last input = 1)
        out = rows.iloc[keep][["fund", "cusip", "q"]].copy()
        out["timesfm_score"] = (dec > UP).mean(axis=1) - (dec < DOWN).mean(axis=1)
        out["timesfm_median"] = dec[:, 4]
        out.to_parquet(path, index=False)
        del m
    S = pd.read_parquet(path)
    sub = rows.iloc[keep][["fund", "cusip", "q"]].merge(S, on=["fund", "cusip", "q"], how="left")
    return evaluate("TimesFM 2.5, zero-shot (own history)", rows, keep, sub["timesfm_score"].fillna(0).to_numpy(),
                    (time.time() - t0) / 60)


def run_fintext(rows, inputs, keep):
    """Resumable: scores are saved every few batches to fintext_scores.parquet."""
    path = PROC / "fintext_scores.parquet"
    t0 = time.time()
    vt = np.where(rows.iloc[keep]["period"].to_numpy() > RC.TRAIN_END)[0]  # validation + test only
    ids = rows.iloc[keep[vt]][["fund", "cusip", "q"]].reset_index(drop=True)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["fund", "cusip", "q", "fintext_score"])
    have = set(zip(done["fund"], done["cusip"], done["q"]))
    todo = [i for i in range(len(vt)) if (ids.at[i, "fund"], ids.at[i, "cusip"], ids.at[i, "q"]) not in have]
    if todo:
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained("FinText/Chronos_Small_2020_US", device_map="cpu",
                                                   torch_dtype=torch.float32)
        parts = [done]
        for b in range(0, len(todo), 256):
            chunk = todo[b:b + 256]
            ctx = []
            for i in chunk:
                x = np.asarray(inputs[vt[i]], dtype=np.float64)
                ch = np.diff(np.log(np.clip(x, 1e-9, None))) if len(x) > 1 else np.array([0.0])
                ctx.append(torch.tensor(np.clip(ch, -3, 3), dtype=torch.float32))
            with torch.no_grad():
                q, _ = pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=list(RC.QLEVELS))
            qm = q[:, 0, :].numpy()
            part = ids.iloc[chunk].copy()
            part["fintext_score"] = (qm > UP).mean(axis=1) - (qm < DOWN).mean(axis=1)
            parts.append(part)
            if (b // 256) % 20 == 19 or b + 256 >= len(todo):
                done = pd.concat(parts, ignore_index=True)
                done.to_parquet(path, index=False)
                parts = [done]
                print(f"  FinText Chronos: {len(done):,}/{len(vt):,} scored", flush=True)
    S = pd.read_parquet(path)
    sub = ids.merge(S, on=["fund", "cusip", "q"], how="left")
    return evaluate("Chronos-Small, finance-trained (FinText 2020)", rows, keep[vt],
                    sub["fintext_score"].fillna(0).to_numpy(), (time.time() - t0) / 60)


def main() -> None:
    T = pd.read_parquet(PROC / "features.parquet", columns=["fund", "cusip", "q", "period", "y"]).reset_index(drop=True)
    H = RC.share_history()
    inputs, keep = RC.build_inputs(T, H, None, [])
    print(f"inputs for {len(keep):,} of {len(T):,} rows", flush=True)
    res = [run_timesfm(T, inputs, keep), run_fintext(T, inputs, keep)]
    EXP.mkdir(exist_ok=True)
    pd.DataFrame(res).to_csv(EXP / "rq2_tsfm_extra.csv", index=False)
    print(pd.DataFrame(res).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
