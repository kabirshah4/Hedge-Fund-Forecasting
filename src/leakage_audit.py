"""Step 9 (Phase 2): leakage audit.

Test 1, future-scramble test (the main one).
  For a cutoff quarter T, randomize EVERY input dated in quarter T or later:
  fund labels and holdings, 13F-implied prices, universe ranks, daily prices
  and the EPU index. Then rebuild the features and check that the feature
  rows for quarter T are byte-for-byte the same as with the real data.
  If any feature changes, it was reading the future. (The target y is
  excluded, since it is supposed to come from quarter T.)

Test 2, filing timing.
  A feature at quarter t uses the fund's 13F for t-1. That filing must be
  public before we "predict", i.e. before quarter t's own 13F is due
  (t-end + 45 days = about 135 days after t-1 ends). We also keep the NEWEST
  filing for each quarter, so an amendment filed much later would be a
  data-vintage leak. We measure how often either happens.

Outputs: experiments/leakage_audit.txt

Usage: .venv/bin/python src/leakage_audit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import build_features as BF

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EXP = ROOT / "experiments"

_real_read_parquet = pd.read_parquet
_real_read_excel = pd.read_excel
_real_to_parquet = pd.DataFrame.to_parquet


def scrambler(Tq: int, rng: np.random.Generator):
    """Return read_parquet / read_excel replacements that randomize data from quarter Tq on."""
    year, qi = divmod(Tq, 4)
    cut = pd.Timestamp(year=year, month=3 * qi + 1, day=1)  # first day of quarter T

    def noise(n):
        return np.exp(rng.normal(0, 1, n))

    def read_parquet(path, *a, **k):
        df = _real_read_parquet(path, *a, **k)
        name = Path(path).name
        if name == "labels.parquet":
            fut = BF.qn(df["period"]) >= Tq
            df.loc[fut, "label"] = rng.choice(["BUY", "HOLD", "SELL"], fut.sum())
            if "shares" in df:
                df.loc[fut, "shares"] = df.loc[fut, "shares"] * noise(fut.sum())
            # shares_prev at T is the t-1 holding (known), so it stays
        elif name == "holdings.parquet":
            fut = df["period"] >= cut
            for c in ["value_usd", "shares"]:
                if c in df:
                    df.loc[fut, c] = df.loc[fut, c] * noise(fut.sum())
        elif name == "implied_prices.parquet":
            fut = df["period"] >= cut
            for c in ["implied_price", "inst_value_usd"]:
                if c in df:
                    df.loc[fut, c] = df.loc[fut, c] * noise(fut.sum())
        elif name == "universe.parquet":
            fut = BF.qn(df["period"]) >= Tq
            df.loc[fut, "rank"] = rng.permutation(df.loc[fut, "rank"].to_numpy())
        elif name == "prices_daily.parquet":
            fut = pd.to_datetime(df["date"]) >= cut
            df.loc[fut, "adj_close"] = df.loc[fut, "adj_close"] * noise(fut.sum())
        return df

    def read_excel(path, *a, **k):
        df = _real_read_excel(path, *a, **k)
        y = pd.to_numeric(df["Year"], errors="coerce")
        m = pd.to_numeric(df["Month"], errors="coerce")
        fut = (y * 4 + (m - 1) // 3) >= Tq
        col = "News_Based_Policy_Uncert_Index"
        df.loc[fut, col] = pd.to_numeric(df.loc[fut, col], errors="coerce") * noise(fut.sum())
        return df

    return read_parquet, read_excel


def build(read_parquet=None, read_excel=None) -> pd.DataFrame:
    """Run build_features.main() without writing; return the feature table."""
    captured = {}

    def capture(self, path, *a, **k):
        captured["T"] = self.copy()

    pd.read_parquet = read_parquet or _real_read_parquet
    pd.read_excel = read_excel or _real_read_excel
    pd.DataFrame.to_parquet = capture
    try:
        BF.main()
    finally:
        pd.read_parquet, pd.read_excel = _real_read_parquet, _real_read_excel
        pd.DataFrame.to_parquet = _real_to_parquet
    return captured["T"]


def scramble_test(cutoffs: list[int]) -> list[str]:
    lines = ["TEST 1: future-scramble test (features for quarter T must not change)"]
    base = build()
    for Tq in cutoffs:
        rng = np.random.default_rng(Tq)
        rp, re_ = scrambler(Tq, rng)
        alt = build(rp, re_)
        key = ["fund", "cusip", "q"]
        a = base[base["q"] == Tq].set_index(key).sort_index()
        b = alt[alt["q"] == Tq].set_index(key).sort_index()
        feats = [c for c in a.columns if c not in ("y", "label", "period")]
        same_rows = a.index.equals(b.index)
        changed = [c for c in feats if not a[c].equals(b[c])] if same_rows else feats
        # sanity: the scramble really changed later quarters' features
        a2 = base[base["q"] == Tq + 1].set_index(key).sort_index()
        b2 = alt[alt["q"] == Tq + 1].set_index(key).sort_index()
        later = sum(not a2[c].equals(b2[c].reindex(a2.index)) for c in feats)
        y_, qi = divmod(Tq, 4)
        verdict = "PASS" if same_rows and not changed else "FAIL"
        lines.append(f"  T = {y_}Q{qi + 1}: {len(a):,} rows, {len(feats)} features -> {verdict}"
                     f"{'' if not changed else ' changed: ' + ', '.join(changed)}"
                     f"  (control: {later} features of quarter T+1 did change, as they should)")
    return lines


def timing_test() -> list[str]:
    H = _real_read_parquet(PROC / "holdings.parquet", columns=["fund", "period", "filed"])
    H = H[H["period"].dt.is_quarter_end].drop_duplicates(["fund", "period"])
    H["filed"] = pd.to_datetime(H["filed"])
    lag = (H["filed"] - H["period"]).dt.days
    test = H["period"] >= "2021-12-31"  # filings used as t-1 inputs for the test period
    lines = ["", "TEST 2: filing timing (fund-quarters; filing date of the version we use)"]
    for name, m in [("all years", slice(None)), ("test inputs (2021Q4 on)", test)]:
        l_ = lag[m]
        lines.append(f"  {name}: n={len(l_):,}, median lag {l_.median():.0f} days, "
                     f"<=45 days {np.mean(l_ <= 45):.1%}, 46-135 days {np.mean((l_ > 45) & (l_ <= 135)):.1%}, "
                     f">135 days (after the next 13F is due) {np.mean(l_ > 135):.1%}")
    late = H[lag > 135].assign(lag=lag[lag > 135])
    if len(late):
        lines.append("  late fund-quarters by fund (most first):")
        for f, n in late.groupby("fund").size().sort_values(ascending=False).head(8).items():
            lines.append(f"    {f}: {n}")
    return lines


def main() -> None:
    cutoffs = [2022 * 4 + 0, 2024 * 4 + 1, 2026 * 4 + 0]  # 2022Q1, 2024Q2, 2026Q1
    out = scramble_test(cutoffs) + timing_test()
    text = "\n".join(out)
    print(text)
    EXP.mkdir(exist_ok=True)
    (EXP / "leakage_audit.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
