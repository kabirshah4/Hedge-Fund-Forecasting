"""Step 18: RQ3 redesigned after the second blind review (W2a-g).

PRIMARY test, exact pre-disclosure window. For each trade of fund i in
quarter t, the stock's return from the end of quarter t to the day fund i's
13F for t became public (its first filing date, capped at the 45-day
deadline), minus the average return of all universe stocks over the same
window. Only the fund knew its trades during this window. Per quarter we
average across trades (trade-weighted), form spreads, and test the 17
quarterly spreads with Newey-West (1 lag). Primary family per sample: the
three surprise spreads (sells: predicted minus surprise; buys: surprise minus
predicted; stocks: least minus most predictable quintile); Holm-adjusted
p-values within the family.

Also reported (secondary):
  unconditional   all buys minus all sells: do the funds' trades pay at all?
  intensity       sort trades by the model's probability of the action that
                  happened; least likely third minus most likely third (buys),
                  most likely minus least likely (sells), so positive always
                  means "surprising trades did better"
  size-momentum   primary spreads with a characteristic-adjusted benchmark: the
                  average return of universe stocks in the same size tercile and
                  momentum tercile (DGTW-style), since raw sell spreads load on
                  momentum
  delisting bound Yahoo lacks most delisted stocks, which are among the trades
                  without a price; extreme bound: every such trade loses 30%

Output: experiments/rq3_v2.csv
Usage:  .venv/bin/python src/rq3_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(13)

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {"hand-picked hedge funds": "processed", "Form ADV hedge funds": "processed_adv",
           "long-only managers": "processed_longonly"}


def nw_mean(x, lags=1):
    x = pd.Series(x).dropna()
    # small sample: p-values from the t distribution (n - 1 df), not the normal
    fit = sm.OLS(x.to_numpy(), np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=True)
    return float(fit.params[0]), float(fit.tvalues[0]), float(fit.pvalues[0]), len(x)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def price_panel():
    m = pd.read_parquet(ROOT / "data/processed/cusip_map.parquet")
    m = m[m["listed_us"].astype(bool) & m["ticker"].notna()].copy()
    m["yt"] = m["ticker"].str.replace("/", "-").str.replace(" ", "-")
    px = pd.read_parquet(ROOT / "data/processed/prices_daily.parquet", columns=["date", "ticker", "adj_close"]).dropna()
    px["date"] = pd.to_datetime(px["date"])
    W = px.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    cmap = m.drop_duplicates("cusip").set_index("cusip")["yt"]
    return W, cmap


LAST = None


def window_returns(W, start, end, delist=False):
    """Total return per ticker from the last close on/before start to the last
    close on/before end. delist=True: -30% extra if the ticker stops trading
    inside the window."""
    a = W.loc[:start].ffill().iloc[-1]
    b = W.loc[:end].ffill().iloc[-1]
    r = b / a - 1
    if delist:
        stopped = (LAST > pd.Timestamp(start)) & (LAST < pd.Timestamp(end) - pd.Timedelta(days=5))
        r[stopped.reindex(r.index, fill_value=False)] = (1 + r) * 0.7 - 1
    return r


def characteristics():
    """Size and momentum tercile per (cusip, q) at quarter-end t, universe stocks only."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import build_features as BF
    U = pd.read_parquet(ROOT / "data/processed/universe.parquet", columns=["period", "cusip", "rank"])
    U["q"] = BF.qn(U["period"])
    R = BF.stock_returns()[["cusip", "q", "mom4"]]
    C = U.merge(R, on=["cusip", "q"], how="left")
    def thirds(x):
        ok = x.notna()
        out = pd.Series(np.nan, index=x.index)
        if ok.sum() >= 30:
            out[ok] = pd.qcut(x[ok].rank(method="first"), 3, labels=False)
        return out
    C["size3"] = C.groupby("q")["rank"].transform(thirds)
    C["mom3"] = C.groupby("q")["mom4"].transform(thirds)
    return C[["cusip", "q", "size3", "mom3"]]


def main():
    global LAST
    W, cmap = price_panel()
    LAST = W.apply(lambda c: c.last_valid_index())
    U = pd.read_parquet(ROOT / "data/processed/universe.parquet", columns=["period", "cusip"])
    CH = characteristics()
    rows = []
    for sname, folder in SAMPLES.items():
        proc = ROOT / "data" / folder
        P = pd.read_parquet(proc / "rq1_test_preds.parquet")
        P["correct"] = P["pred_lightgbm"] == P["y"]
        P["p_act"] = np.select([P["label"] == "BUY", P["label"] == "SELL"], [P["p_buy"], P["p_sell"]], np.nan)
        H = pd.read_parquet(proc / "holdings.parquet", columns=["fund", "period", "filed"])
        H = H[H["period"].dt.is_quarter_end]
        H["filed"] = pd.to_datetime(H["filed"])
        first = H.groupby(["fund", "period"])["filed"].min().rename("first_filed").reset_index()
        P = P.merge(first, on=["fund", "period"], how="left")
        P["end"] = np.minimum(P["first_filed"].fillna(P["period"] + pd.Timedelta(days=45)),
                              P["period"] + pd.Timedelta(days=45))
        P["ticker"] = P["cusip"].map(cmap)
        last_price = W.index.max()
        per_q, stopped_n, fund_pay = [], [], []
        for q, g in P.groupby("q"):
            qend = g["period"].iloc[0]
            if qend + pd.Timedelta(days=45) > last_price:
                continue  # no forward prices yet
            univ = U[U["period"] == qend]["cusip"].map(cmap).dropna().unique()
            g = g.copy()
            g["ret"] = np.nan
            g["ret_ch"] = np.nan
            g["raw"] = np.nan
            ch = CH[CH["q"] == q].copy()
            ch["ticker"] = ch["cusip"].map(cmap)
            g = g.merge(ch[["cusip", "size3", "mom3"]], on="cusip", how="left")
            for end, gg in g.groupby("end"):
                r = window_returns(W, qend, end)
                g.loc[gg.index, "raw"] = gg["ticker"].map(r)
                g.loc[gg.index, "ret"] = gg["ticker"].map(r) - r.reindex(univ).mean()
                ch["r"] = ch["ticker"].map(r)
                cell = ch.groupby(["size3", "mom3"])["r"].mean()
                bench = pd.MultiIndex.from_arrays([gg["size3"], gg["mom3"]]).map(lambda k: cell.get(k, np.nan))
                g.loc[gg.index, "ret_ch"] = gg["ticker"].map(r).to_numpy() - np.asarray(bench, float)
                stopped_n.append(int(((LAST.reindex(gg["ticker"]) > qend) & (LAST.reindex(gg["ticker"]) < end - pd.Timedelta(days=5))).sum()))
            # trade-quarter window: from the end of t-1 to the end of t (the trade happened inside it)
            prev_end = (qend - pd.offsets.QuarterEnd(1))
            rq = window_returns(W, prev_end, qend)
            g["ret_tq"] = g["ticker"].map(rq) - rq.reindex(univ).mean()
            # copier's window: three months starting on the day the 13F became public
            g["ret_post"] = np.nan
            for end, gg in g.groupby("end"):
                if end + pd.Timedelta(days=91) > last_price:
                    continue
                rp = window_returns(W, end, end + pd.Timedelta(days=91))
                g.loc[gg.index, "ret_post"] = gg["ticker"].map(rp) - rp.reindex(univ).mean()
            missing = g[g["label"].isin(["BUY", "SELL"]) & g["ret"].isna()]
            g["ret_bound"] = g["ret"]
            # bound: every trade without a price is a delisting losing 30% in the window
            g.loc[missing.index, "ret_bound"] = -0.30
            gb = g[g["label"].isin(["BUY", "SELL"])]
            g = g.dropna(subset=["ret"])
            T = g[g["label"].isin(["BUY", "SELL"])]
            b, s = T[T["label"] == "BUY"], T[T["label"] == "SELL"]
            out = {"q": q,
                   "unconditional": b["ret"].mean() - s["ret"].mean(),
                   "sells": s[s["correct"]]["ret"].mean() - s[~s["correct"]]["ret"].mean(),
                   "buys": b[~b["correct"]]["ret"].mean() - b[b["correct"]]["ret"].mean()}
            for lab, df in [("intensity buys", b), ("intensity sells", s)]:
                t3 = pd.qcut(df["p_act"].rank(method="first"), 3, labels=False)
                lo, hi = df[t3 == 0]["ret"].mean(), df[t3 == 2]["ret"].mean()
                out[lab] = (lo - hi) if lab == "intensity buys" else (hi - lo)
            st = g.groupby("cusip").agg(ps=("correct", "mean"), n=("correct", "size"), ret=("ret", "mean"))
            st = st[st["n"] >= 2]
            k = pd.qcut(st["ps"].rank(method="first"), 5, labels=False)
            out["stocks"] = st[k == 0]["ret"].mean() - st[k == 4]["ret"].mean()
            out["sells (size-momentum adjusted)"] = s[s["correct"]]["ret_ch"].mean() - s[~s["correct"]]["ret_ch"].mean()
            out["buys (size-momentum adjusted)"] = b[~b["correct"]]["ret_ch"].mean() - b[b["correct"]]["ret_ch"].mean()
            sb, bb = gb[gb["label"] == "SELL"], gb[gb["label"] == "BUY"]
            out["sells (missing = delisted, -30%)"] = sb[sb["correct"]]["ret_bound"].mean() - sb[~sb["correct"]]["ret_bound"].mean()
            out["buys (missing = delisted, -30%)"] = bb[~bb["correct"]]["ret_bound"].mean() - bb[bb["correct"]]["ret_bound"].mean()
            # within fund and quarter: a fund's predicted sells vs its own surprise sells
            def within(df, flip):
                d = df.groupby(["fund", "correct"])["ret"].mean().unstack()
                d = d.dropna()
                diff = (d[True] - d[False]) if flip else (d[False] - d[True])
                return diff.mean() if len(diff) else np.nan
            out["sells (within fund-quarter)"] = within(s, True)
            out["buys (within fund-quarter)"] = within(b, False)
            out["sells (after filing, copier)"] = s[s["correct"]]["ret_post"].mean() - s[~s["correct"]]["ret_post"].mean()
            out["buys (after filing, copier)"] = b[~b["correct"]]["ret_post"].mean() - b[b["correct"]]["ret_post"].mean()
            out["sells (trade quarter)"] = s[s["correct"]]["ret_tq"].mean() - s[~s["correct"]]["ret_tq"].mean()
            out["buys (trade quarter)"] = b[~b["correct"]]["ret_tq"].mean() - b[b["correct"]]["ret_tq"].mean()
            per_q.append(out)
            pay = T.assign(pay=np.where(T["label"] == "BUY", 1, -1) * T["ret"])
            fund_pay.append(pay.groupby("fund")["pay"].agg(["sum", "size"]))
        D = pd.DataFrame(per_q)
        prim = []
        for col in ["unconditional", "sells", "buys", "stocks", "intensity sells", "intensity buys",
                    "sells (size-momentum adjusted)", "buys (size-momentum adjusted)",
                    "sells (missing = delisted, -30%)", "buys (missing = delisted, -30%)",
                    "sells (trade quarter)", "buys (trade quarter)",
                    "sells (within fund-quarter)", "buys (within fund-quarter)",
                    "sells (after filing, copier)", "buys (after filing, copier)"]:
            mu, t, p, n = nw_mean(D[col])
            r = {"sample": sname, "test": col, "mean_pct": mu * 100, "t": t, "p": p, "quarters": n,
                 "family": "primary" if col in ("sells", "buys", "stocks") else "secondary"}
            rows.append(r)
            if r["family"] == "primary":
                prim.append(len(rows) - 1)
        adj = holm([rows[i]["p"] for i in prim])
        for i, a in zip(prim, adj):
            rows[i]["p_holm"] = a
        # manager level: do less predictable funds' trades pay off more before disclosure?
        FP = pd.concat(fund_pay).groupby(level=0).sum()
        FP["payoff"] = FP["sum"] / FP["size"] * 100
        acc = P.assign(c=P["pred_lightgbm"] == P["y"], b=P["pred_majority"] == P["y"]).groupby("fund")[["c", "b"]].mean()
        FP = FP.join((acc["c"] - acc["b"]).rename("gain") * 100).dropna()
        FP = FP[FP["size"] >= 50]
        rho = FP["gain"].corr(FP["payoff"], method="spearman")
        boot = []
        for _ in range(2000):
            ix = rng.integers(0, len(FP), len(FP))
            boot.append(FP["gain"].iloc[ix].reset_index(drop=True).corr(FP["payoff"].iloc[ix].reset_index(drop=True), method="spearman"))
        rows.append({"sample": sname, "test": "manager level: Spearman(gain, trade payoff)", "mean_pct": rho,
                     "ci_low": np.nanpercentile(boot, 2.5), "ci_high": np.nanpercentile(boot, 97.5),
                     "quarters": len(D), "count": len(FP), "family": "secondary"})
        rows.append({"sample": sname, "test": "trades whose stock stops trading in the window",
                     "mean_pct": np.nan, "quarters": len(D), "count": int(np.sum(stopped_n)),
                     "share_pct": 100 * np.sum(stopped_n) / max(1, int(P["label"].isin(["BUY", "SELL"]).sum())),
                     "family": "info"})
        print(f"{sname}: {len(D)} quarters with a pre-disclosure window", flush=True)
    R = pd.DataFrame(rows)
    prim = R["family"] == "primary"
    R.loc[prim, "p_holm_all"] = holm(R.loc[prim, "p"].to_numpy())  # across all samples' primary tests
    R.to_csv(ROOT / "experiments" / "rq3_v2.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
