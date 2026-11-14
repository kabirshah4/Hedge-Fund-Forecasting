"""Step 9 (Phase 2): RQ3 with factor-adjusted returns (alphas).

Question: do unpredicted trades still earn more once we account for the usual
risk factors (market, size, value, profitability, investment, momentum)? If
surprise buys are just small or high-momentum stocks, their extra return
would be a known factor premium, not skill.

Method: calendar-time portfolios (the standard approach, e.g. in Mimicking
Finance section 7):
  1. At each quarter-end t, form portfolios from the test predictions:
       surprise buys, predicted buys, surprise sells, predicted sells
       (weighted by number of trades in the stock), and the stock-level
       predictability quintiles Q1 (least) and Q5 (most), equal-weighted.
  2. Hold them over a window and record MONTHLY returns:
       next_q       the 3 months of quarter t+1 (13F not yet public)
       post_filing  months 3-5 after quarter-end (13F public by day 45)
  3. Long-short spreads:
       buy spread   surprise buys - predicted buys
       sell spread  predicted sells - surprise sells  (positive = surprise
                    sells fell more, i.e. were better timed)
       trade spread average of the two
       Q1 - Q5      least minus most predictable stocks
  4. Regress each monthly spread on CAPM, Fama-French 3, and FF5 + momentum.
     The intercept (alpha) is the return the factors can't explain.
     t-stats use Newey-West standard errors (3 lags).

Only stocks with a Yahoo price are used (monthly prices needed); the rest
(mostly delisted) are dropped, which we report.

Outputs: experiments/rq3_factor_alphas.csv, experiments/rq3_monthly_spreads.csv,
         paper/figures/fig_rq3_alphas.pdf (single-column figure for the paper)

Usage: .venv/bin/python src/rq3_factors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sample as SMP

ROOT = Path(__file__).resolve().parent.parent
PROC = SMP.PROC
FF = ROOT / "data" / "raw" / "ff"
EXP = SMP.EXP


def read_ff(name: str) -> pd.DataFrame:
    """Monthly block of a Ken French CSV (stops at the annual block)."""
    rows = []
    started = False
    for line in open(FF / name):
        parts = [p.strip() for p in line.strip().split(",")]
        if parts[0] == "" and len(parts) > 1 and not started:
            cols = parts[1:]
            started = True
            continue
        if started:
            if not (parts[0].isdigit() and len(parts[0]) == 6):
                break
            rows.append([parts[0]] + [float(x) for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["ym"] + cols)
    df["month"] = pd.PeriodIndex(df["ym"].str[:4] + "-" + df["ym"].str[4:], freq="M")
    return df.drop(columns="ym").set_index("month") / 100


def monthly_returns() -> pd.DataFrame:
    """Monthly total returns per cusip from Yahoo adjusted closes."""
    m = pd.read_parquet(PROC / "cusip_map.parquet")
    m = m[m["listed_us"].astype(bool) & m["ticker"].notna()].copy()
    m["yt"] = m["ticker"].str.replace("/", "-").str.replace(" ", "-")
    px = pd.read_parquet(PROC / "prices_daily.parquet", columns=["date", "ticker", "adj_close"]).dropna()
    px["date"] = pd.to_datetime(px["date"])
    px["month"] = px["date"].dt.to_period("M")
    me = px.sort_values("date").groupby(["ticker", "month"])["adj_close"].last().reset_index()
    me = me.sort_values(["ticker", "month"])
    me["ret"] = me.groupby("ticker")["adj_close"].pct_change()
    gap = me.groupby("ticker")["month"].diff().apply(lambda x: x.n if pd.notna(x) else np.nan)
    me.loc[gap != 1, "ret"] = np.nan
    me["ret"] = me["ret"].clip(-0.95, 5)
    r = me.merge(m[["cusip", "yt"]].drop_duplicates("cusip"), left_on="ticker", right_on="yt")
    return r[["cusip", "month", "ret"]].dropna()


def quarter_end_month(q: int) -> pd.Period:
    year, qi = divmod(q, 4)
    return pd.Period(f"{year}-{3 * qi + 3:02d}", freq="M")


def portfolios(P: pd.DataFrame, quintiles: bool = True) -> dict[str, pd.DataFrame]:
    """Weights per (q, cusip) for each named portfolio."""
    out = {}
    T = P[P["label"].isin(["BUY", "SELL"])]
    for act in ["BUY", "SELL"]:
        for name, corr in [("surprise", False), ("predicted", True)]:
            w = T[(T["label"] == act) & (T["correct"] == corr)].groupby(["q", "cusip"]).size().rename("w")
            out[f"{name}_{act.lower()}"] = w.reset_index()
    if not quintiles:
        return out
    S = P.groupby(["cusip", "q"]).agg(pred_share=("correct", "mean"), n=("correct", "size")).reset_index()
    S = S[S["n"] >= 2]
    S["quintile"] = S.groupby("q")["pred_share"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    for k in [1, 5]:
        out[f"Q{k}"] = S[S["quintile"] == k][["q", "cusip"]].assign(w=1.0)
    return out


def portfolio_returns(W: pd.DataFrame, R: pd.DataFrame, offsets: list[int]) -> pd.Series:
    """Monthly weighted return; month offsets counted from quarter-end."""
    rows = []
    for q, g in W.groupby("q"):
        qe = quarter_end_month(q)
        for k in offsets:
            mth = qe + k
            r = g.merge(R[R["month"] == mth], on="cusip")
            if len(r) == 0:
                continue
            rows.append((mth, np.average(r["ret"], weights=r["w"]), r["w"].sum() / g["w"].sum()))
    s = pd.DataFrame(rows, columns=["month", "ret", "coverage"]).set_index("month")
    return s


MODELS = {
    "raw mean": [],
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5 + momentum": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


def alpha(y: pd.Series, F: pd.DataFrame, factors: list[str], lags: int = 3) -> tuple[float, float, int]:
    d = pd.concat([y.rename("y"), F], axis=1, join="inner").dropna()
    X = np.ones((len(d), 1)) if not factors else sm.add_constant(d[factors])
    fit = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params.iloc[0]), float(fit.tvalues.iloc[0]), len(d)


def main() -> None:
    P = pd.read_parquet(PROC / "rq1_test_preds.parquet")
    P["correct"] = P["pred_lightgbm"] == P["y"]
    R = monthly_returns()
    F = read_ff("F-F_Research_Data_5_Factors_2x3.csv").join(read_ff("F-F_Momentum_Factor.csv"), how="inner")

    covered = P["cusip"].isin(R["cusip"].unique()).mean()
    print(f"test positions whose stock has monthly Yahoo prices: {covered:.1%}")

    W = portfolios(P)
    windows = {"next_q": [1, 2, 3], "post_filing": [3, 4, 5]}
    spreads, rows = {}, []
    for win, offs in windows.items():
        pr = {name: portfolio_returns(w, R, offs) for name, w in W.items()}
        for name, s in pr.items():
            print(f"  {win:12s} {name:15s} months={len(s):3d}  weight with prices={s['coverage'].mean():.0%}")
        ret = {k: v["ret"] for k, v in pr.items()}
        sp = pd.DataFrame({
            "buy spread": ret["surprise_buy"] - ret["predicted_buy"],
            "sell spread": ret["predicted_sell"] - ret["surprise_sell"],
            "Q1 - Q5": ret["Q1"] - ret["Q5"],
        })
        sp["trade spread"] = (sp["buy spread"] + sp["sell spread"]) / 2
        # single legs in excess of the risk-free rate
        for leg in ["surprise_buy", "predicted_buy", "surprise_sell", "predicted_sell", "Q1", "Q5"]:
            sp[f"leg: {leg}"] = ret[leg] - F["RF"].reindex(ret[leg].index)
        spreads[win] = sp
        for col in sp.columns:
            for mname, fac in MODELS.items():
                a, t, n = alpha(sp[col], F, fac)
                rows.append({"window": win, "portfolio": col, "model": mname,
                             "alpha_monthly": a, "t_NW": t, "months": n})
    A = pd.DataFrame(rows)
    EXP.mkdir(exist_ok=True)
    A.to_csv(EXP / "rq3_factor_alphas.csv", index=False)
    pd.concat(spreads, names=["window"]).to_csv(EXP / "rq3_monthly_spreads.csv")

    show = A[~A["portfolio"].str.startswith("leg")].copy()
    show["alpha"] = (show["alpha_monthly"] * 100).map("{:+.2f}%".format) + " (t=" + show["t_NW"].map("{:.2f}".format) + ")"
    tab = show.pivot_table(index=["window", "portfolio"], columns="model", values="alpha", aggfunc="first")
    pd.set_option("display.width", 200)
    print("\nMONTHLY alphas of long-short spreads (Newey-West t, 3 lags)")
    print(tab[list(MODELS)].to_string())
    make_figure(A)

    print("\nROBUSTNESS (monthly; raw and FF5+momentum alpha, Newey-West t)")
    Rb = robustness(P, R, F)
    Rb.to_csv(EXP / "rq3_robustness.csv", index=False)
    show = Rb.copy()
    show["raw"] = (show["raw"] * 100).map("{:+.2f}%".format) + " (" + show["t_raw"].map("{:.2f}".format) + ")"
    show["FF5+Mom"] = (show["alpha_ff5mom"] * 100).map("{:+.2f}%".format) + " (" + show["t_ff5mom"].map("{:.2f}".format) + ")"
    print(show[["weighting", "horizon", "spread", "months", "raw", "FF5+Mom"]].to_string(index=False))


def value_weighted(W: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Multiply each weight by the stock's institutional value at quarter-end t
    (known at t), so big stocks count more."""
    ip = pd.read_parquet(PROC / "implied_prices.parquet", columns=["cusip", "period", "inst_value_usd"])
    ip = ip[ip["period"].dt.is_quarter_end]
    ip["q"] = ip["period"].dt.year * 4 + (ip["period"].dt.month - 1) // 3
    ip = ip.groupby(["cusip", "q"])["inst_value_usd"].max().reset_index()
    out = {}
    for k, w in W.items():
        v = w.merge(ip, on=["cusip", "q"], how="left")
        v["w"] = v["w"] * v["inst_value_usd"].fillna(v["inst_value_usd"].median())
        out[k] = v[["q", "cusip", "w"]]
    return out


def robustness(P: pd.DataFrame, R: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Value weighting and longer holding periods. Overlapping cohorts are
    averaged month by month (standard calendar-time approach)."""
    W0 = portfolios(P)
    weights = {"trade/equal-weighted": W0, "value-weighted": value_weighted(W0)}
    windows = {"3 months (main)": [1, 2, 3], "6 months": list(range(1, 7)),
               "12 months": list(range(1, 13)), "post-filing (months 3-5)": [3, 4, 5]}
    rows = []
    for wname, W in weights.items():
        for win, offs in windows.items():
            ret = {}
            for name, w in W.items():
                pr = portfolio_returns(w, R, offs)
                ret[name] = pr.groupby(level=0)["ret"].mean()
            sp = {"sells": ret["predicted_sell"] - ret["surprise_sell"],
                  "buys": ret["surprise_buy"] - ret["predicted_buy"],
                  "Q1 - Q5": ret["Q1"] - ret["Q5"]}
            lags = max(3, len(offs))  # overlapping holding periods need as many lags as months held
            for k, y in sp.items():
                a0, t0, n = alpha(y, F, [], lags)
                a6, t6, _ = alpha(y, F, MODELS["FF5 + momentum"], lags)
                rows.append({"weighting": wname, "horizon": win, "spread": k, "months": n, "nw_lags": lags,
                             "raw": a0, "t_raw": t0, "alpha_ff5mom": a6, "t_ff5mom": t6})
    return pd.DataFrame(rows)


def make_figure(A: pd.DataFrame) -> None:
    """Dot-and-whisker: monthly alpha with 95% CI, per spread and factor model.
    Sized for one ACM column (3.33 in) so text prints at ~7-8 pt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    rows = [("sell spread", "Sells: predicted\nminus surprise"),
            ("buy spread", "Buys: surprise\nminus predicted"),
            ("Q1 - Q5", "Stocks: least minus\nmost predictable")]
    models = list(MODELS)
    legend_names = {"raw mean": "raw", "CAPM": "CAPM", "FF3": "Fama-French 3", "FF5 + momentum": "Fama-French 5 + momentum"}
    colors = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp: more factors = darker
    markers = ["o", "s", "D", "^"]
    d = A[A["window"] == "next_q"]
    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    for i, (key, _) in enumerate(rows):
        for j, m in enumerate(models):
            r = d[(d["portfolio"] == key) & (d["model"] == m)].iloc[0]
            a = r["alpha_monthly"] * 100
            ci = 1.96 * abs(a / r["t_NW"])
            y = len(rows) - 1 - i + (1.5 - j) * 0.17
            ax.errorbar(a, y, xerr=ci, fmt=markers[j], color=colors[j], ms=4, lw=1.2, capsize=0,
                        label=legend_names[m] if i == 0 else None)
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows][::-1])
    ax.set_xlabel("Monthly return difference, % (95% CI)")
    ax.grid(axis="x", color="#ddd", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower center", bbox_to_anchor=(0.35, 1.0),
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    out = (ROOT / "paper" / "figures" / "fig_rq3_alphas.pdf") if SMP.IS_MAIN else (EXP / "fig_rq3_alphas.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(EXP / "fig_rq3_alphas.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
