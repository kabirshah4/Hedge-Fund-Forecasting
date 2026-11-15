"""Add Chronos-2 forecasts for prediction rows that don't have one yet.

When the sample changes (e.g. Winton's successor SEC ID joined, Nov 13),
new rows appear. Each forecast depends only on that position's own history,
so existing forecasts stay valid; only missing rows are computed, with the
same settings as run_chronos.py (sets A, B, D; validation + test rows) and
chronos_scores_all.py (set A; all rows).
"""

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

import run_chronos as RC


def forecast(pipe, inputs):
    qs, _ = pipe.predict_quantiles(inputs, prediction_length=1, quantile_levels=list(RC.QLEVELS), batch_size=256)
    return torch.stack([q[0, 0, :] for q in qs]).numpy()


def main() -> None:
    T = pd.read_parquet(RC.PROC / "features.parquet", columns=["fund", "cusip", "q", "period", "y"]).reset_index(drop=True)
    pipe = Chronos2Pipeline.from_pretrained(RC.MODEL, device_map="cpu")
    H = RC.share_history()
    key = ["fund", "cusip", "q"]

    # 1. chronos_scores_all (hybrid feature): set A, all rows
    S = pd.read_parquet(RC.PROC / "chronos_scores_all.parquet")
    miss = T.merge(S[key], on=key, how="left", indicator=True)
    rows = T[(miss["_merge"] == "left_only").to_numpy()].reset_index(drop=True)
    if len(rows):
        inputs, keep = RC.build_inputs(rows, H, None, [])
        q = forecast(pipe, inputs)
        add = rows.iloc[keep][key].copy()
        add["chronos_score"] = (q > 1.01).mean(axis=1) - (q < 0.99).mean(axis=1)
        add["chronos_median"] = np.log(np.clip(q[:, 4], 1e-6, None))
        pd.concat([S, add], ignore_index=True).to_parquet(RC.PROC / "chronos_scores_all.parquet", index=False)
    print(f"chronos_scores_all: added {len(rows):,} rows")

    # 2. chronos_preds (zero-shot evaluation): sets A, B, D on validation + test rows
    P = pd.read_parquet(RC.PROC / "chronos_preds.parquet")
    vt = T[T["period"] > RC.TRAIN_END].reset_index(drop=True)
    miss = vt.merge(P[key], on=key, how="left", indicator=True)
    rows = vt[(miss["_merge"] == "left_only").to_numpy()].reset_index(drop=True)
    if len(rows):
        cov_all = RC.covariate_table(["A", "B", "D"])
        add = rows[["fund", "cusip", "q", "period", "y"]].copy()
        for s, names in [("A", []), ("B", ["ret"]), ("D", ["ret", "epu"])]:
            inputs, keep = RC.build_inputs(rows, H, cov_all[["cusip", "q"] + names] if names else cov_all, names)
            q = forecast(pipe, inputs)
            add.loc[add.index[keep], f"chronos_{s}_score"] = (q > 1.01).mean(axis=1) - (q < 0.99).mean(axis=1)
        pd.concat([P, add], ignore_index=True).to_parquet(RC.PROC / "chronos_preds.parquet", index=False)
    print(f"chronos_preds: added {len(rows):,} rows")


if __name__ == "__main__":
    main()
