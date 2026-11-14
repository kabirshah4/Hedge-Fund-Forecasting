"""Which fund sample the pipeline runs on.

Default: the hand-picked list (data/fund_list.csv -> data/processed, experiments).
Set SAMPLE=adv to run on the Form ADV list (data/fund_list_adv.csv ->
data/processed_adv, experiments_adv). Fund-independent files (universe, prices,
ticker map, implied prices, splits) are shared via symlinks.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = os.environ.get("SAMPLE", "main")
_SUFFIX = "" if SAMPLE == "main" else f"_{SAMPLE}"

PROC = ROOT / "data" / f"processed{_SUFFIX}"
EXP = ROOT / f"experiments{_SUFFIX}"
FUND_LIST = ROOT / "data" / f"fund_list{_SUFFIX}.csv"
IS_MAIN = SAMPLE == "main"
# Train / validation split. N-PORT starts in late 2019, so those samples train
# on 2020Q1-2021Q2 and validate on 2021H2; the test period (2022 on) is the same.
_SPLITS = {"nport": ("2021-06-30", "2021-12-31"), "nportparent": ("2021-06-30", "2021-12-31")}
TRAIN_END, VAL_END = _SPLITS.get(SAMPLE, ("2020-12-31", "2021-12-31"))
SHARED = ["universe.parquet", "implied_prices.parquet", "cusip_map.parquet",
          "prices_daily.parquet", "splits.parquet"]

if not IS_MAIN:
    PROC.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    for f in SHARED:
        link = PROC / f
        if not link.exists():
            link.symlink_to(ROOT / "data" / "processed" / f)
