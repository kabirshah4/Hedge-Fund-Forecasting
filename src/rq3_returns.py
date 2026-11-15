"""Step 8: RQ3, do the trades the model can't predict earn more?

Uses the LightGBM test predictions (2022 Q1 - 2026 Q1; the last quarter has
no "next quarter" yet). A trade is "predicted" if the model's label matched
the fund's actual action, a "surprise" otherwise.

Returns (market-adjusted: minus the average universe stock that window):
  next_q        the quarter right after quarter-end t (only the fund could
                trade on this; the 13F isn't public yet)
  post_filing   3 months starting 45 days after quarter-end t, when the 13F
                is public: what an outsider could actually trade

Tests (in the spirit of Mimicking Finance, section 7):
  1. trade level   "did the action pay off": return for a BUY, minus the
                   return for a SELL. Surprise vs predicted.
  2. stock level   each quarter, rank stocks by the share of their holders'
                   trades the model predicted; compare next-quarter returns
                   of the least (Q1) and most (Q5) predictable fifths.
Significance: averages are taken per quarter first, then a t-test across
quarters (so one busy quarter can't dominate).

Fama-French alphas: see rq3_factors.py.

Outputs:
  experiments/rq3_trade_level.csv, experiments/rq3_stock_level.csv,
  experiments/fig_rq3.png

Usage:
  .venv/bin/python src/rq3_returns.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import build_features as BF
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
PROC = SMP.PROC
EXP = SMP.EXP


def tstat(per_quarter: pd.Series) -> tuple[float, float, int]:
    s = per_quarter.dropna()
    if len(s) < 3:
        return float(s.mean()) if len(s) else np.nan, np.nan, len(s)
    return float(s.mean()), float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), len(s)


def quarterly_yahoo_returns() -> pd.DataFrame:
    """Quarterly returns per (cusip, q) compounded from monthly Yahoo returns.
    A stock delisted mid-quarter keeps the months it traded."""
    from rq3_factors import monthly_returns
    R = monthly_returns()
    R["q"] = R["month"].dt.year * 4 + (R["month"].dt.quarter - 1)
    return R.groupby(["cusip", "q"])["ret"].apply(lambda s: (1 + s).prod() - 1).reset_index()


def post_filing_returns() -> pd.DataFrame:
    """Return from quarter-end + 45 days over the next ~3 months, per (cusip, q)."""
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    m = m[m["listed_us"].astype(bool) & m["ticker"].notna()].copy()
    m["yt"] = m["ticker"].str.replace("/", "-").str.replace(" ", "-")
    px = pd.read_parquet(PROC / "prices_daily.parquet", columns=["date", "ticker", "adj_close"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.dropna().sort_values("date")
    out = []
    for q in range(2022 * 4, 2026 * 4 + 2):
        year, qi = divmod(q, 4)
        qend = pd.Timestamp(year=year, month=3 * qi + 3, day=1) + pd.offsets.MonthEnd(0)
        start, end = qend + pd.Timedelta(days=45), qend + pd.Timedelta(days=45 + 91)
        a = px[px["date"] <= start].groupby("ticker")["adj_close"].last()
        b = px[(px["date"] > start) & (px["date"] <= end)].groupby("ticker")["adj_close"].last()
        r = (b / a - 1).dropna().rename("ret").reset_index()
        r["q"] = q
        out.append(r)
    R = pd.concat(out).merge(m[["cusip", "yt"]], left_on="ticker", right_on="yt")
    return R[["cusip", "q", "ret"]].rename(columns={"ret": "post_filing"})


def main() -> None:
    P = pd.read_parquet(PROC / "rq1_test_preds.parquet")
    P["correct"] = P["pred_lightgbm"] == P["y"]

    # next-quarter return (quarter t+1), market-adjusted against the universe AT t.
    # Bug fixed 2026-11-13: v1 required the stock to still be in the universe at
    # t+1, which is only known after the quarter (lookahead) and silently dropped
    # stocks that fell out of the top 500. Now: every stock in the universe at t
    # keeps its t+1 return, and only Yahoo prices are used (13F-implied prices
    # for untickered stocks are too noisy for return tests).
    R = quarterly_yahoo_returns()
    U = pd.read_parquet(PROC / "universe.parquet", columns=["period", "cusip"])
    U["q"] = BF.qn(U["period"])
    nq = U[["cusip", "q"]].merge(R.assign(q=R["q"] - 1), on=["cusip", "q"])  # in universe at t, return of t+1
    nq["next_q"] = nq["ret"] - nq.groupby("q")["ret"].transform("mean")
    P = P.merge(nq[["cusip", "q", "next_q"]], on=["cusip", "q"], how="left")

    pf = post_filing_returns()
    pf["post_filing"] = pf["post_filing"] - pf.groupby("q")["post_filing"].transform("mean")
    P = P.merge(pf, on=["cusip", "q"], how="left")

    rows = []
    # 1. trade level: did the action pay off? (BUY: +ret, SELL: -ret)
    T = P[P["label"].isin(["BUY", "SELL"])].copy()
    sign = np.where(T["label"] == "BUY", 1, -1)
    for win in ["next_q", "post_filing"]:
        T[f"payoff_{win}"] = sign * T[win]
        g = T.groupby(["q", "correct"])[f"payoff_{win}"].mean().unstack()
        for name, col in [("surprise trades", False), ("predicted trades", True)]:
            m_, t_, n_ = tstat(g[col])
            rows.append({"test": "trade level", "window": win, "group": name, "mean": m_, "t": t_, "quarters": n_})
        m_, t_, n_ = tstat(g[False] - g[True])
        rows.append({"test": "trade level", "window": win, "group": "surprise minus predicted", "mean": m_, "t": t_, "quarters": n_})
        for act in ["BUY", "SELL"]:
            a = T[T["label"] == act].groupby(["q", "correct"])[win].mean().unstack()
            m_, t_, n_ = tstat(a[False] - a[True])
            rows.append({"test": f"{act} only: surprise minus predicted (raw return)", "window": win,
                         "group": act, "mean": m_, "t": t_, "quarters": n_})
    trade = pd.DataFrame(rows)

    # 2. stock level: quintiles of how predictable the stock's holders were
    srows = []
    S = P.groupby(["cusip", "q"]).agg(pred_share=("correct", "mean"), n=("correct", "size"),
                                      next_q=("next_q", "first"), post_filing=("post_filing", "first")).reset_index()
    S = S[S["n"] >= 2]
    S["quintile"] = S.groupby("q")["pred_share"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    for win in ["next_q", "post_filing"]:
        g = S.groupby(["q", "quintile"])[win].mean().unstack()
        for k in range(1, 6):
            m_, t_, n_ = tstat(g[k])
            srows.append({"window": win, "quintile": f"Q{k}" + (" (least predictable)" if k == 1 else " (most predictable)" if k == 5 else ""),
                          "mean": m_, "t": t_, "quarters": n_})
        m_, t_, n_ = tstat(g[1] - g[5])
        srows.append({"window": win, "quintile": "Q1 minus Q5", "mean": m_, "t": t_, "quarters": n_})
    stock = pd.DataFrame(srows)

    EXP.mkdir(exist_ok=True)
    trade.to_csv(EXP / "rq3_trade_level.csv", index=False)
    stock.to_csv(EXP / "rq3_stock_level.csv", index=False)
    pd.set_option("display.width", 160)
    fmt = lambda d: d.assign(mean=(d["mean"] * 100).round(2).astype(str) + "%", t=d["t"].round(2))
    print("TRADE LEVEL (quarterly, market-adjusted; payoff = +return for BUY, -return for SELL)")
    print(fmt(trade).to_string(index=False))
    print("\nSTOCK LEVEL (quarterly, market-adjusted next-period return by predictability quintile)")
    print(fmt(stock).to_string(index=False))
    print(f"\nrows with a next-quarter return: {P['next_q'].notna().mean():.0%}; "
          f"with a post-filing return: {P['post_filing'].notna().mean():.0%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        # Left: within each action (the fair comparison), next-quarter raw-minus-market return
        T2 = P[P["label"].isin(["BUY", "SELL"])]
        g = T2.groupby(["label", "correct"])["next_q"].mean().unstack() * 100
        x = np.arange(2)
        axes[0].bar(x - 0.18, g.loc[["BUY", "SELL"], False], 0.36, label="surprise (model was wrong)", color="#C8553D")
        axes[0].bar(x + 0.18, g.loc[["BUY", "SELL"], True], 0.36, label="predicted (model was right)", color="#5B7C99")
        axes[0].set_xticks(x, ["stocks funds BOUGHT", "stocks funds SOLD"])
        axes[0].axhline(0, color="#444", lw=0.8)
        axes[0].set_ylabel("Next-quarter return vs average stock, %")
        axes[0].set_title("Surprise buys rose more; surprise sells fell more")
        axes[0].legend(frameon=False, fontsize=8)
        # Right: stock-level quintiles
        q = stock[(stock["window"] == "next_q") & stock["quintile"].str.startswith("Q") & ~stock["quintile"].str.contains("minus")]
        axes[1].bar(["Q1\nleast", "Q2", "Q3", "Q4", "Q5\nmost"], q["mean"] * 100, color="#5B7C99")
        axes[1].axhline(0, color="#444", lw=0.8)
        axes[1].set_ylabel("Next-quarter return vs average stock, %")
        axes[1].set_title("By how predictable a stock's holders were")
        fig.suptitle("RQ3 (rough): hedge funds' unpredicted trades, test 2022-2026 (17 quarters)")
        fig.tight_layout()
        fig.savefig(EXP / "fig_rq3.png", dpi=160)
        print("saved experiments/fig_rq3.png")
    except ImportError:
        print("matplotlib not installed: no figure")


if __name__ == "__main__":
    main()
