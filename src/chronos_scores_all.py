"""Step 7b: Chronos-2 scores for EVERY prediction row (train, validation, test).

The hybrid model (step 7c) gives LightGBM the Chronos-2 forecast as one more
feature, so it needs Chronos-2 scores for the training years too. Each score
still uses only the position's history up to quarter t-1 (no leakage).
Input set A (own share history) only.

Output: data/processed/chronos_scores_all.parquet  fund, cusip, q, chronos_score, chronos_median
Usage:  .venv/bin/python src/chronos_scores_all.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

import run_chronos as RC

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    T = pd.read_parquet(PROC / "features.parquet", columns=["fund", "cusip", "q", "period"]).reset_index(drop=True)
    H = RC.share_history()
    inputs, keep = RC.build_inputs(T, H, None, [])
    pipe = Chronos2Pipeline.from_pretrained(RC.MODEL, device_map="cpu")
    qs, _ = pipe.predict_quantiles(inputs, prediction_length=1, quantile_levels=list(RC.QLEVELS), batch_size=256)
    qmat = torch.stack([q[0, 0, :] for q in qs]).numpy()
    out = T.iloc[keep][["fund", "cusip", "q"]].copy()
    out["chronos_score"] = (qmat > 1.01).mean(axis=1) - (qmat < 0.99).mean(axis=1)
    out["chronos_median"] = np.log(np.clip(qmat[:, 4], 1e-6, None))  # forecast log change
    out.to_parquet(PROC / "chronos_scores_all.parquet", index=False)
    print(f"scored {len(out):,} of {len(T):,} rows")


if __name__ == "__main__":
    main()
