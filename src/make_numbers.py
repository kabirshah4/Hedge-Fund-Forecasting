"""Single source of truth for every number in the paper.

Reads the result files in experiments*/ and data/processed*/ and writes
  paper/numbers.tex        one LaTeX macro per number cited in the text
  paper/tables/*.tex       every results table
so the paper never contains a hand-typed result. Rerun after any experiment:
    .venv/bin/python src/make_numbers.py && (cd paper && tectonic main.tex)

Sample suffixes in macro names: M = hand-picked hedge funds (main),
A = Form ADV hedge funds, L = long-only managers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
TAB = PAPER / "tables"
S = {"M": ("experiments", "data/processed"), "A": ("experiments_adv", "data/processed_adv"),
     "L": ("experiments_longonly", "data/processed_longonly"), "N": ("experiments_nport", "data/processed_nport"),
     "P": ("experiments_nportparent", "data/processed_nportparent")}
ALL = "MALNP"
macros: dict[str, str] = {}


def num(x, d=1, sign=False):
    """Format for LaTeX: proper minus sign, optional plus sign."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{abs(x):,.{d}f}"
    if round(x, d) < 0:
        return f"$-${s}"
    return ("+" if sign and round(x, d) > 0 else "") + s


def m(name, value):
    assert name.isalpha(), name
    macros[name] = value


def exp(k, f):
    return pd.read_csv(ROOT / S[k][0] / f)


def proc(k, f):
    return pd.read_parquet(ROOT / S[k][1] / f)


def sample_stats():
    rows = {}
    for k in S:
        L = proc(k, "labels.parquet")
        T = proc(k, "features.parquet")
        te = T[T["period"] > "2021-12-31"]
        sh = L["label"].value_counts(normalize=True) * 100
        fq = L.groupby(["fund", "period"]).size()
        r = {"funds": L["fund"].nunique(), "active": te["fund"].nunique(), "fq": len(fq), "medpos": int(fq.median()),
             "labels": len(L), "sell": sh["SELL"], "buy": sh["BUY"], "hold": sh["HOLD"],
             "new": (L["kind"] == "new position").mean() * 100, "exit": (L["kind"] == "full exit").mean() * 100,
             "cases": len(T), "test": len(te)}
        rows[k] = r
        m(f"funds{k}", str(r["funds"]))
        m(f"active{k}", str(r["active"]))
        m(f"testrows{k}", f"{r['test']:,}")
        m(f"hold{k}", num(r["hold"]))
        m(f"new{k}", num(r["new"], 0))
    lines = [r"\begin{table}[t]",
             r"  \caption{The three samples. Labels are quarterly position changes in the 500-stock universe,",
             r"  2013Q3--2026Q2; prediction cases are positions held at the end of the previous quarter.}",
             r"  \label{tab:summary}", r"  \small", r"  \setlength{\tabcolsep}{3pt}",
             r"  \begin{tabular}{@{}lccc@{}}", r"    \toprule",
             r"     & \multicolumn{2}{c}{Hedge funds} & Long-only \\ \cmidrule(lr){2-3}",
             r"     & Hand-picked & Form ADV & managers \\", r"    \midrule"]
    def row(label, f):
        lines.append(f"    {label} & " + " & ".join(f(rows[k]) for k in "MAL") + r" \\")
    row("Funds (active in test)", lambda r: f"{r['funds']} ({r['active']})")
    row("Fund-quarters", lambda r: f"{r['fq']:,}")
    row("Median positions", lambda r: str(r["medpos"]))
    row("Labels", lambda r: f"{r['labels']:,}")
    row(r"\quad SELL/BUY/HOLD (\%)", lambda r: f"{r['sell']:.0f}/{r['buy']:.0f}/{r['hold']:.0f}")
    row(r"\quad new/exits (\%)", lambda r: f"{r['new']:.0f}/{r['exit']:.0f}")
    row("Prediction cases", lambda r: f"{r['cases']:,}")
    row(r"\quad test (2022Q1--26Q2)", lambda r: f"{r['test']:,}")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TAB / "summary.tex").write_text("\n".join(lines) + "\n")


def rq1():
    for k in S:
        b = exp(k, "rq1_baselines.csv").set_index("model")
        m(f"base{k}", num(b.loc["majority", "mean_fund_accuracy"] * 100))
        m(f"lgb{k}", num(b.loc["lightgbm", "mean_fund_accuracy"] * 100))
        m(f"lgbpooled{k}", num(b.loc["lightgbm", "accuracy"] * 100))
        m(f"basepooled{k}", num(b.loc["majority", "accuracy"] * 100))
        bs = exp(k, "rq1_bootstrap.csv")
        f = bs[(bs["resampling"] == "by fund") & (bs["metric"] == "mean per-fund accuracy")].iloc[0]
        m(f"gain{k}", num(f["gain_points"], sign=True))
        m(f"gainlo{k}", num(f["ci95_low"]))
        m(f"gainhi{k}", num(f["ci95_high"]))
        q = bs[(bs["resampling"] == "by quarter") & (bs["metric"] == "mean per-fund accuracy")].iloc[0]
        m(f"gainqlo{k}", num(q["ci95_low"]))
        m(f"gainqhi{k}", num(q["ci95_high"]))
        x = exp(k, "rq1_metrics.csv").iloc[0]
        m(f"errred{k}", num(x["error_reduction_of_means"] * 100, 0))
        m(f"llskill{k}", num(x["logloss_skill_mean_fund"] * 100))
        m(f"aucbs{k}", f"{x['auc_buy_sell']:.2f}")
        pf = exp(k, "rq1_per_fund.csv")
        g = (pf["lightgbm"] - pf["majority"]) * 100
        m(f"medgain{k}", num(g.median()))
        m(f"nabove{k}", str(int((g > 1).sum())))
        m(f"nlowbase{k}", str(int((pf["majority"] < 0.5).sum())))
        m(f"gainlowbase{k}", num(g[pf["majority"] < 0.5].mean()))
        m(f"gainrest{k}", num(g[pf["majority"] >= 0.5].mean()))
        if not (ROOT / S[k][0] / "rq1_robustness.csv").exists():
            continue  # N-PORT samples: too few quarters for the rolling protocol
        r = exp(k, "rq1_robustness.csv")
        p = r[r["check"] == "Mimicking Finance protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100
        m(f"protbase{k}", num(p["majority (per fund)"]))
        m(f"protfund{k}", num(p["LightGBM per fund"]))
        m(f"protpool{k}", num(p["LightGBM pooled"]))
        m(f"protfundgain{k}", num(p["LightGBM per fund"] - p["majority (per fund)"], sign=True))
        m(f"protpoolgain{k}", num(p["LightGBM pooled"] - p["majority (per fund)"], sign=True))
        bsr = r[r["check"] == "BUY vs SELL only"].set_index("model")
        m(f"bsbase{k}", num(bsr.loc["majority (per fund)", "mean_fund_accuracy"] * 100))
        m(f"bslgb{k}", num(bsr.loc["LightGBM", "mean_fund_accuracy"] * 100))
        m(f"bsgain{k}", num((bsr.loc["LightGBM", "mean_fund_accuracy"] - bsr.loc["majority (per fund)", "mean_fund_accuracy"]) * 100, sign=True))
        hb = r[r["check"].str.startswith("HOLD band")].set_index("model")
        m(f"hbgain{k}", num((hb.loc["LightGBM", "mean_fund_accuracy"] - hb.loc["majority (per fund)", "mean_fund_accuracy"]) * 100, sign=True))
    # main-sample model table
    b = exp("M", "rq1_baselines.csv").set_index("model")
    sq = exp("M", "rq1_sequence_models.csv")
    ms = sq[sq["check"] == "main split"].set_index("model")
    ch = exp("M", "rq2_chronos.csv").set_index("input_set")
    tf = exp("M", "rq2_tsfm_extra.csv").set_index("model")
    P = pd.read_parquet(ROOT / S["M"][1] / "rq1_test_preds.parquet")
    from sklearn.metrics import f1_score
    ex = exp("M", "rq1_extra.csv").set_index("model")
    auc = ex["auc_buy_sell"].dropna()
    from sklearn.metrics import roc_auc_score
    pb = P[P["label"] != "HOLD"]
    lg = (P["label"] != "HOLD")
    def mf(name):
        r = ms.loc[name]
        return (r["accuracy"], r["macro_f1"], r["mean_fund_accuracy"])
    rule = "rule: BUY if the fund mostly bought last quarter"
    rows = [("Majority class (per fund)", b.loc["majority", "accuracy"], b.loc["majority", "macro_f1"], b.loc["majority", "mean_fund_accuracy"], np.nan),
            ("Recent majority (8 quarters)", ex.loc["recent majority (last 8 quarters)", "accuracy"], ex.loc["recent majority (last 8 quarters)", "macro_f1"],
             ex.loc["recent majority (last 8 quarters)", "mean_fund_accuracy"], np.nan),
            ("Repeat last action", b.loc["repeat", "accuracy"], b.loc["repeat", "macro_f1"], b.loc["repeat", "mean_fund_accuracy"], np.nan),
            (r"Rule: fund's net direction at $t-1$", *mf(rule), np.nan),
            ("Logistic regression", *mf("logistic regression (pooled)"), ms.loc["logistic regression (pooled)", "auc_buy_sell"]),
            ("LSTM, 8-quarter sequences", *mf("LSTM (pooled, 8-quarter sequences)"), ms.loc["LSTM (pooled, 8-quarter sequences)", "auc_buy_sell"]),
            ("LightGBM", b.loc["lightgbm", "accuracy"], b.loc["lightgbm", "macro_f1"], b.loc["lightgbm", "mean_fund_accuracy"],
             auc["LightGBM (p_buy - p_sell)"]),
            ("Chronos-2, zero-shot", ch.loc["A", "accuracy"], ch.loc["A", "macro_f1"], ch.loc["A", "mean_fund_accuracy"], auc["Chronos-2 (raw score)"]),
            (r"\quad + returns, EPU covariates", ch.loc["D", "accuracy"], ch.loc["D", "macro_f1"], ch.loc["D", "mean_fund_accuracy"], np.nan),
            ("TimesFM 2.5, zero-shot", tf.iloc[0]["accuracy"], tf.iloc[0]["macro_f1"], tf.iloc[0]["mean_fund_accuracy"], auc["TimesFM 2.5 (raw score)"]),
            ("Chronos, finance-trained", tf.iloc[1]["accuracy"], tf.iloc[1]["macro_f1"], tf.iloc[1]["mean_fund_accuracy"],
             auc["Chronos finance-trained (raw score)"])]
    best = [max(r[k] for r in rows if not np.isnan(r[k])) for k in (1, 2, 3, 4)]
    lines = [r"\begin{table}[t]",
             r"  \caption{RQ1 and RQ2, hand-picked sample: predicting hedge funds' quarterly trades, test period",
             r"  2022Q1--2026Q2 (\testrowsM{} positions). Fund acc.\ is accuracy averaged over funds; AUC ranks",
             r"  BUY against SELL from each model's continuous score. Foundation models are zero-shot on each",
             r"  position's share history. Best value in each column in bold.}",
             r"  \label{tab:rq1}", r"  \small", r"  \setlength{\tabcolsep}{3pt}",
             r"  \begin{tabular}{@{}lcccc@{}}", r"    \toprule", r"    Model & Accuracy & Macro-F1 & Fund acc. & AUC \\", r"    \midrule"]
    for i, (name, acc, f1, fa, au) in enumerate(rows):
        if i == 7:
            lines.append(r"    \midrule")
        vals = [acc, f1, fa, au]
        cells = []
        for k, (v, fmt) in enumerate(zip(vals, ["{:.1f}\\%", "{:.2f}", "{:.1f}\\%", "{:.2f}"])):
            if np.isnan(v):
                cells.append("--")
                continue
            c = fmt.format(v * 100 if "%" in fmt else v)
            if abs(v - best[k]) < 1e-12:
                c = f"\\textbf{{{c}}}"
            cells.append(c)
        lines.append(f"    {name} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TAB / "rq1.tex").write_text("\n".join(lines) + "\n")
    m("aucchronosM", f"{auc['Chronos-2 (raw score)']:.2f}"); m("auctimesfmM", f"{auc['TimesFM 2.5 (raw score)']:.2f}")
    m("aucfintextM", f"{auc['Chronos finance-trained (raw score)']:.2f}"); m("auclgbM", f"{auc['LightGBM (p_buy - p_sell)']:.2f}")
    m("recentM", num(ex.loc["recent majority (last 8 quarters)", "mean_fund_accuracy"] * 100))
    m("lstmM", num(ms.loc["LSTM (pooled, 8-quarter sequences)", "mean_fund_accuracy"] * 100))
    m("logitM", num(ms.loc["logistic regression (pooled)", "mean_fund_accuracy"] * 100))
    m("ruleM", num(ms.loc["rule: BUY if the fund mostly bought last quarter", "mean_fund_accuracy"] * 100))
    m("chronosM", num(ch.loc["A", "mean_fund_accuracy"] * 100))
    m("chronosDM", num(ch.loc["D", "mean_fund_accuracy"] * 100))
    m("chronospooledM", num(ch.loc["A", "accuracy"] * 100))
    m("chronosholdM", num(ch.loc["A", "share_predicted_hold"] * 100))
    m("timesfmM", num(tf.iloc[0]["mean_fund_accuracy"] * 100))
    m("fintextM", num(tf.iloc[1]["mean_fund_accuracy"] * 100))
    pr = sq[sq["check"] == "their protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100
    m("protlogitM", num(pr["logistic regression per fund"]))
    m("protlstmM", num(pr["LSTM per fund"]))
    m("protlogitgainM", num(pr["logistic regression per fund"] - pr["majority (per fund)"], sign=True))
    m("protlstmgainM", num(pr["LSTM per fund"] - pr["majority (per fund)"], sign=True))
    from robustness import style_groups
    st = P["fund"].map(style_groups())
    m("multiquantshareM", num(st.isin(["Multi-strategy", "Quantitative"]).mean() * 100, 0))
    hs = P.groupby("fund")["label"].apply(lambda x: (x == "HOLD").mean() * 100)
    for key, f in [("elliott", "Elliott Management"), ("icahn", "Icahn (Carl Icahn)"), ("pershing", "Pershing Square")]:
        m(f"hold{key}", num(hs[f], 0))
    m("holdpooledtestM", num((P["label"] == "HOLD").mean() * 100))
    T = pd.read_parquet(ROOT / S["M"][1] / "features.parquet", columns=["period", "holding_age"])
    m("medhistM", str(int(T.loc[T["period"] > "2021-12-31", "holding_age"].median())))
    F = pd.read_parquet(ROOT / S["M"][1] / "features.parquet", columns=["fund", "q"])
    qs = sorted(F["q"].unique())
    ntr = []
    for b in range(20, len(qs) - 7, 8):  # same blocks as robustness.py / rq1_sequence_models.py
        tr, te = F[F["q"].isin(qs[b - 20:b])], F[F["q"].isin(qs[b:b + 8])]
        ntr += [int((tr["fund"] == f).sum()) for f in te["fund"].unique()]
    m("medtrainM", f"{int(np.median(ntr)):,}")


def samples_table():
    """One full-width table: the five samples, their data, and RQ1 under the same pipeline."""
    C = pd.read_csv(ROOT / "experiments" / "compare_samples.csv").set_index("sample")
    def dif(name, tag):
        r = C.loc[name]
        m(f"diffgain{tag}", num(r["gain_points"], sign=True)); m(f"diffgainlo{tag}", num(r["gain_lo"]))
        m(f"diffgainhi{tag}", num(r["gain_hi"]))
        m(f"differr{tag}", num(r["error_reduction_pct"], sign=True)); m(f"differrlo{tag}", num(r["err_lo"]))
        m(f"differrhi{tag}", num(r["err_hi"]))
    dif("DIFF long-only minus hedge funds, hand-picked", "M")
    dif("DIFF long-only minus hedge funds, Form ADV", "A")
    dif("DIFF long-only without ETF sponsors minus hedge funds, hand-picked", "noetfM")
    dif("DIFF mutual funds, N-PORT minus hedge funds, hand-picked", "NM")
    dif("DIFF mutual funds in multi-fund families minus fund families (funds added up)", "NP")
    noetf = C.loc["long-only without ETF sponsors"]
    m("gainLnoetf", num(noetf["gain_points"], sign=True)); m("errredLnoetf", num(noetf["error_reduction_pct"], 0))
    st = pd.read_csv(ROOT / "experiments" / "short_training.csv").set_index("sample")
    stk = {"M": "hedge funds, hand-picked", "A": "hedge funds, Form ADV", "L": "long-only managers",
           "N": "mutual funds, N-PORT", "P": "fund families, N-PORT"}
    for k in ALL:
        r = st.loc[stk[k]]
        m(f"shortgain{k}", num(r["gain"], sign=True)); m(f"shortgainlo{k}", num(r["gain_lo"]))
        m(f"shortgainhi{k}", num(r["gain_hi"])); m(f"shorterr{k}", num(r["error_reduction_pct"], 0))
    m("shorterrmin", num(st["error_reduction_pct"].min(), 0)); m("shorterrmax", num(st["error_reduction_pct"].max(), 0))
    cols = []
    for k in ALL:
        L = proc(k, "labels.parquet")
        T = proc(k, "features.parquet")
        sh = L["label"].value_counts(normalize=True) * 100
        b = exp(k, "rq1_baselines.csv").set_index("model")
        bs = exp(k, "rq1_bootstrap.csv")
        f = bs[(bs["resampling"] == "by fund") & (bs["metric"] == "mean per-fund accuracy")].iloc[0]
        x = exp(k, "rq1_metrics.csv").iloc[0]
        xe = exp(k, "rq1_extra.csv").set_index("model")
        te = T[T["period"] > "2021-12-31"]
        c = {"funds": f"{L['fund'].nunique()} ({te['fund'].nunique()})", "labels": f"{len(L):,}",
             "mix": f"{sh.get('SELL', 0):.0f}/{sh.get('BUY', 0):.0f}/{sh.get('HOLD', 0):.0f}",
             "test": f"{len(te):,}",
             "train": "2013Q3--20Q4" if k in "MAL" else "2020Q1--21Q2",
             "base": f"{b.loc['majority', 'mean_fund_accuracy'] * 100:.1f}\\%",
             "lgb": f"{b.loc['lightgbm', 'mean_fund_accuracy'] * 100:.1f}\\%",
             "gain": f"{num(f['gain_points'], sign=True)} [{num(f['ci95_low'])}, {num(f['ci95_high'])}]",
             "err": f"{x['error_reduction_of_means'] * 100:.0f}\\%",
             "short": ("same as above" if k in "NP" else
                       f"{num(st.loc[stk[k], 'gain'], sign=True)} [{num(st.loc[stk[k], 'gain_lo'])}, {num(st.loc[stk[k], 'gain_hi'])}]"),
             "shorterr": ("same as above" if k in "NP" else f"{st.loc[stk[k], 'error_reduction_pct']:.0f}\\%"),
             "fw": num(xe.loc["LightGBM, fund-weighted training", "mean_fund_gain"], sign=True),
             "oracle": num(xe.loc["oracle: fund's actual net direction at t", "mean_fund_gain"], sign=True),
             "lgbor": num(xe.loc["LightGBM + oracle fund buy share at t", "mean_fund_gain"], sign=True)}
        if (ROOT / S[k][0] / "rq1_robustness.csv").exists():
            r = exp(k, "rq1_robustness.csv")
            p = r[r["check"] == "Mimicking Finance protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100
            c["prot"] = num(p["LightGBM per fund"] - p["majority (per fund)"], sign=True)
            c["pool"] = num(p["LightGBM pooled"] - p["majority (per fund)"], sign=True)
            bsr = r[r["check"] == "BUY vs SELL only"].set_index("model")["mean_fund_accuracy"]
            hbr = r[r["check"].str.startswith("HOLD band")].set_index("model")["mean_fund_accuracy"]
            c["bs"] = num((bsr["LightGBM"] - bsr["majority (per fund)"]) * 100, sign=True)
            c["hb"] = num((hbr["LightGBM"] - hbr["majority (per fund)"]) * 100, sign=True)
        else:
            c["prot"] = c["pool"] = c["bs"] = c["hb"] = "--"
        c["protlogit"] = c["protlstm"] = "--"
        if k == "M":
            sq = exp("M", "rq1_sequence_models.csv")
            pr = sq[sq["check"] == "their protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100
            c["protlogit"] = num(pr["logistic regression per fund"] - pr["majority (per fund)"], sign=True)
            c["protlstm"] = num(pr["LSTM per fund"] - pr["majority (per fund)"], sign=True)
        cols.append(c)
        m(f"labels{k}", f"{len(L):,}")
        m(f"fundsall{k}", str(L["fund"].nunique()))
        m(f"active{k}", str(te["fund"].nunique()))
        m(f"hold{k}", num(sh.get("HOLD", 0)))
    lines = [r"\begin{table*}[t]",
             r"  \caption{The five samples under the identical pipeline. Funds: in the sample (active in the test",
             r"  period 2022Q1--2026Q2). Accuracy is mean per-fund accuracy; gains are in points over each fund's",
             r"  majority action, with 95\% intervals from resampling funds. The 13F samples train on 2013Q3--2020Q4;",
             r"  N-PORT begins in late 2019, so the ``same training window'' rows retrain every sample on",
             r"  2020Q1--2021Q2 (validation 2021H2) for a like-for-like comparison. The per-fund protocol",
             r"  of~\cite{cohen2026mimicking} (20 training and 8 test quarters, four blocks 2018Q3--2026Q2, one model",
             r"  per fund, gains over the same blocks' majority baseline) needs 28 quarters and is not available for N-PORT.}",
             r"  \label{tab:samples}", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
             r"  \begin{tabular}{@{}lccccc@{}}", r"    \toprule",
             r"     & \multicolumn{2}{c}{Hedge funds (13F)} & Long-only & \multicolumn{2}{c}{Mutual funds (N-PORT)} \\",
             r"    \cmidrule(lr){2-3} \cmidrule(lr){5-6}",
             r"     & Hand-picked & Form ADV & managers (13F) & Individual funds & Fund families \\", r"    \midrule"]
    for label, key in [("Funds (active in test)", "funds"),
                       ("\\quad SELL/BUY/HOLD (\\%)", "mix"), ("Test cases", "test"), ("Training window", "train"),
                       ("\\midrule Majority baseline", "base"), ("Gain [95\\% CI]", "gain"),
                       ("Baseline errors fixed", "err"),
                       ("Fund-weighted training, gain", "fw"),
                       ("BUY vs.\\ SELL only, gain", "bs"), ("HOLD band $\\pm5\\%$, gain", "hb"),
                       ("\\midrule Per-fund protocol of~\\cite{cohen2026mimicking}: LightGBM", "prot"),
                       ("\\quad logistic regression", "protlogit"), ("\\quad LSTM", "protlstm"),
                       ("\\quad pooled LightGBM", "pool"),
                       ("\\midrule Same training window: gain", "short"), ("\\quad errors fixed", "shorterr"),
                       ("\\midrule Oracle (fund's direction at $t$), gain", "oracle"),
                       ("LightGBM + oracle, gain", "lgbor")]:
        lab = label.replace("\\midrule ", "")
        if label.startswith("\\midrule"):
            lines.append(r"    \midrule")
        lines.append(f"    {lab} & " + " & ".join(c[key] for c in cols) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    (TAB / "samples.tex").write_text("\n".join(lines) + "\n")
    xs = {k: exp(k, "rq1_extra.csv").set_index("model") for k in ALL}
    for k in ALL:
        g = lambda model: xs[k].loc[model, "mean_fund_gain"]
        m(f"fundwtgain{k}", num(g("LightGBM, fund-weighted training"), sign=True))
        m(f"fundwtmed{k}", num(xs[k].loc["LightGBM, fund-weighted training", "median_fund_gain"]))
        m(f"nopeergain{k}", num(g("LightGBM without peer features"), sign=True))
        m(f"oraclegain{k}", num(g("oracle: fund's actual net direction at t"), sign=True))
        m(f"lgboraclegain{k}", num(g("LightGBM + oracle fund buy share at t"), sign=True))
        m(f"lgboracle{k}", num(xs[k].loc["LightGBM + oracle fund buy share at t", "mean_fund_accuracy"] * 100))
        m(f"recentgain{k}", num(g("recent majority (last 8 quarters)"), sign=True))


def robust_table():
    sq = exp("M", "rq1_sequence_models.csv")
    pr = sq[sq["check"] == "their protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100
    r = exp("M", "rq1_robustness.csv")
    p = r[r["check"] == "Mimicking Finance protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100
    base = p["majority (per fund)"]
    bsr = r[r["check"] == "BUY vs SELL only"].set_index("model")
    hb = r[r["check"].str.startswith("HOLD band")].set_index("model")
    b = exp("M", "rq1_baselines.csv").set_index("model")
    rows = [("Rolling protocol: LightGBM per fund", base, p["LightGBM per fund"]),
            ("\\quad logistic regression per fund", pr["majority (per fund)"], pr["logistic regression per fund"]),
            ("\\quad LSTM per fund", pr["majority (per fund)"], pr["LSTM per fund"]),
            ("\\quad LightGBM pooled", base, p["LightGBM pooled"]),
            ("BUY vs.\\ SELL only (no HOLD)", bsr.loc["majority (per fund)", "mean_fund_accuracy"] * 100, bsr.loc["LightGBM", "mean_fund_accuracy"] * 100),
            ("HOLD band $\\pm5\\%$ instead of $\\pm1\\%$", hb.loc["majority (per fund)", "mean_fund_accuracy"] * 100, hb.loc["LightGBM", "mean_fund_accuracy"] * 100),
            ("Main result (Table~\\ref{tab:rq1})", b.loc["majority", "mean_fund_accuracy"] * 100, b.loc["lightgbm", "mean_fund_accuracy"] * 100)]
    lines = [r"\begin{table}[t]",
             r"  \caption{RQ1 robustness, hand-picked sample: mean per-fund accuracy of the per-fund majority",
             r"  baseline and each model. The rolling protocol follows~\cite{cohen2026mimicking}: 20 training and",
             r"  8 test quarters, four test blocks (2018Q3--2026Q2), one model per fund.}",
             r"  \label{tab:robust}", r"  \small", r"  \setlength{\tabcolsep}{3.5pt}",
             r"  \begin{tabular}{@{}lccc@{}}", r"    \toprule", r"    Check & Baseline & Model & Gain \\", r"    \midrule"]
    for i, (name, bb, mm) in enumerate(rows):
        if i == len(rows) - 1:
            lines.append(r"    \midrule")
        lines.append(f"    {name} & {bb:.1f}\\% & {mm:.1f}\\% & {num(mm - bb, sign=True)} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TAB / "robust.tex").write_text("\n".join(lines) + "\n")
    m("bsaucM", f"{bsr.loc['LightGBM', 'auc']:.2f}")


def style_table():
    s1 = exp("M", "rq1_by_style.csv").set_index("style")
    s3 = exp("M", "rq3_by_style.csv").set_index("style")
    sz = exp("M", "rq3_style_portfolio_sizes.csv").set_index("style")
    order = ["Long/short equity and growth", "Activist / event-driven", "Multi-strategy", "Quantitative", "Macro / value"]
    lines = [r"\begin{table}[t]",
             r"  \caption{By fund style, hand-picked sample. RQ1: gain in mean per-fund accuracy over the majority",
             r"  baseline (points). RQ3: FF5 + momentum alpha of the spreads over the next quarter (\% per month,",
             r"  Newey--West $t$); dashes where the predicted or surprise portfolio holds a median of fewer than 10 stocks per quarter.}",
             r"  \label{tab:style}", r"  \small", r"  \setlength{\tabcolsep}{3pt}",
             r"  \begin{tabular}{@{}lcccc@{}}", r"    \toprule",
             r"    Style (funds in test) & Funds & RQ1 gain & RQ3 sells & RQ3 buys \\", r"    \midrule"]
    small = []
    for st in order:
        n = int(s1.loc[st, "funds"])
        ok_s = min(sz.loc[st, "predicted_sell"], sz.loc[st, "surprise_sell"]) >= 10
        ok_b = min(sz.loc[st, "predicted_buy"], sz.loc[st, "surprise_buy"]) >= 10
        sells = f"{num(s3.loc[st, 'sells_alpha_pct'], 2, sign=True)} ({num(s3.loc[st, 'sells_t'])})" if ok_s else "--"
        buys = f"{num(s3.loc[st, 'buys_alpha_pct'], 2, sign=True)} ({num(s3.loc[st, 'buys_t'])})" if ok_b else "--"
        if not (ok_s and ok_b):
            small.append(st)
        lines.append(f"    {st.replace('Long/short equity and growth', 'Long/short equity, growth')} & {n} & "
                     f"{num(s1.loc[st, 'gain_points'], sign=True)} & {sells} & {buys} \\\\")
        key = {"Long/short equity and growth": "ls", "Activist / event-driven": "act", "Multi-strategy": "multi",
               "Quantitative": "quant", "Macro / value": "macro"}[st]
        m(f"style{key}gain", num(s1.loc[st, "gain_points"], sign=True))
        m(f"style{key}n", str(n))
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TAB / "style.tex").write_text("\n".join(lines) + "\n")
    m("multisellsalpha", num(s3.loc["Multi-strategy", "sells_alpha_pct"], 2, sign=True))
    m("multisellst", num(s3.loc["Multi-strategy", "sells_t"]))
    m("actbuystocks", num(sz.loc["Activist / event-driven", "predicted_buy"], 0))


def rq2_table():
    R = exp("M", "rq2_lightgbm.csv")
    names = {"+chronos": "Chronos-2 forecast", "+timesfm": "TimesFM forecast", "+news_prev": "News, previous quarter ($t-1$)",
             "+news_now": "News, same quarter ($t$)", "+news_late15_prev": "\\quad last 15 days of $t-1$ only",
             "+news_late15_now": "\\quad last 15 days of $t$ only", "+all": "Chronos-2 + all news",
             "+llm_prev": "LLM reading of news ($t-1$)", "+llm_now": "LLM reading of news ($t$)"}
    order = [("2022-2026Q2 (full test)", "+chronos"), ("2022-2026Q2 (full test)", "+timesfm"),
             ("2022-2023 (news window)", "+news_prev"), ("2022-2023 (news window)", "+news_late15_prev"),
             ("2022-2023 (news window)", "+news_now"), ("2022-2023 (news window)", "+news_late15_now"),
             ("2022-2023 (news window)", "+all"), ("2022-2023 (news window)", "+llm_prev"),
             ("2022-2023 (news window)", "+llm_now")]
    lines = [r"\begin{table}[t]",
             r"  \caption{RQ2: change in mean per-fund accuracy (points) when each input is added to LightGBM,",
             r"  hand-picked sample, with 95\% intervals from resampling funds (2,000 draws). News inputs are",
             r"  tested on 2022--2023, the test years with news data.}",
             r"  \label{tab:rq2}", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
             r"  \begin{tabular}{@{}llc@{}}", r"    \toprule", r"    Added input & Years & Change [95\% CI] \\", r"    \midrule"]
    for w, mdl in order:
        r = R[(R["window"] == w) & (R["model"] == mdl)]
        if not len(r):
            continue
        r = r.iloc[0]
        yrs = "2022--26" if w.startswith("2022-2026") else "2022--23"
        lines.append(f"    {names[mdl]} & {yrs} & {num(r['gain_vs_base_points'], 2, sign=True)} "
                     f"[{num(r['ci95_low'], 2)}, {num(r['ci95_high'], 2)}] \\\\")
        key = {"+chronos": "chronos", "+timesfm": "timesfm", "+news_prev": "newsprev", "+news_now": "newsnow",
               "+news_late15_prev": "lateprev", "+news_late15_now": "latenow", "+all": "all",
               "+llm_prev": "llmprev", "+llm_now": "llmnow"}[mdl]
        m(f"rqtwo{key}", num(r["gain_vs_base_points"], 2, sign=True))
        m(f"rqtwo{key}lo", num(r["ci95_low"], 2))
        m(f"rqtwo{key}hi", num(r["ci95_high"], 2))
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TAB / "rq2.tex").write_text("\n".join(lines) + "\n")
    hi = R.loc[R["model"] != "base", "ci95_high"].max()
    m("rqtwomaxhi", num(hi, 2, sign=True))
    m("rqtwomaxhiabs", f"{np.ceil(hi * 2) / 2:g}")  # rounded up to the next half point, for prose


def rq3():
    A = exp("M", "rq3_factor_alphas.csv")
    def g(win, port, model):
        r = A[(A["window"] == win) & (A["portfolio"] == port) & (A["model"] == model)].iloc[0]
        return r["alpha_monthly"] * 100, r["t_NW"]
    for key, port in [("sells", "sell spread"), ("buys", "buy spread"), ("stocks", "Q1 - Q5")]:
        a, t = g("next_q", port, "raw mean"); m(f"raw{key}", num(a, 2, sign=True)); m(f"raw{key}t", num(t, 2))
        a, t = g("next_q", port, "FF5 + momentum"); m(f"ff{key}", num(a, 2, sign=True)); m(f"ff{key}t", num(t, 2))
        a, t = g("post_filing", port, "FF5 + momentum"); m(f"post{key}", num(a, 2, sign=True)); m(f"post{key}t", num(t, 2))
    rb = exp("M", "rq3_robustness.csv")
    m("rqthreepos", str(int((rb["alpha_ff5mom"] > 0).sum())))
    m("rqthreesig", str(int((rb["t_ff5mom"].abs() >= 1.96).sum())))
    m("rqthreemin", num(rb["alpha_ff5mom"].min() * 100, 2, sign=True))
    m("rqthreemax", num(rb["alpha_ff5mom"].max() * 100, 2, sign=True))
    pw = pd.read_csv(ROOT / "experiments" / "rq3_power.csv")
    for k, test in [("M", "main sample"), ("A", "adv sample"), ("L", "longonly sample")]:
        r = pw[(pw["test"] == test) & (pw["spread"] == "Q1 - Q5")].iloc[0]
        m(f"stockcilo{k}", num(r["ci_low"], 2)); m(f"stockcihi{k}", num(r["ci_high"], 2)); m(f"stockmde{k}", num(r["mde80"], 2))
    trade = pw[pw["test"].isin(["main sample", "adv sample", "longonly sample"]) & pw["spread"].isin(["sells", "buys"])]
    m("trademdelo", num(trade["mde80"].min(), 2)); m("trademdehi", num(trade["mde80"].max(), 2))
    for key, sp in [("sells", "sells"), ("buys", "buys")]:
        r = pw[(pw["test"] == "difference: hand-picked minus Form ADV") & (pw["spread"] == sp)].iloc[0]
        m(f"diffsample{key}", num(r["alpha_pct_month"], 2, sign=True)); m(f"diffsample{key}t", num(r["t"]))
        o = pw[pw["test"].str.startswith("hand-picked, funds also") & (pw["spread"] == sp)].iloc[0]
        n = pw[pw["test"].str.startswith("hand-picked, funds not") & (pw["spread"] == sp)].iloc[0]
        m(f"overlap{key}", num(o["alpha_pct_month"], 2, sign=True)); m(f"nonoverlap{key}", num(n["alpha_pct_month"], 2, sign=True))
    m("noverlap", pw[pw["test"].str.startswith("hand-picked, funds also")].iloc[0]["test"].split("(")[1].split()[0])
    mj = pw[pw["test"].str.startswith("main, surprise = against")]
    m("majsurprisemaxabst", num(mj["t"].abs().max()))


def mechanism():
    R = pd.read_csv(ROOT / "experiments" / "mechanism.csv")
    def g(sample, test):
        return R[(R["sample"] == sample) & (R["test"] == test)].iloc[0]
    for k, smp in [("M", "hedge funds, hand-picked"), ("A", "hedge funds, Form ADV"), ("L", "long-only")]:
        r = g(smp, "small minus large (within fund)")
        m(f"sml{k}", num(r["value"], sign=True)); m(f"smllo{k}", num(r["ci_low"])); m(f"smlhi{k}", num(r["ci_high"]))
        r = g(smp, "Spearman: gain vs. fund's median holdings")
        m(f"rhoholdings{k}", num(r["value"], 2, sign=True))
    for k, smp in [("M", "hedge funds, hand-picked minus long-only"), ("A", "hedge funds, Form ADV minus long-only")]:
        r = g(smp, "small-minus-large difference")
        m(f"smldiff{k}", num(r["value"], sign=True)); m(f"smldifflo{k}", num(r["ci_low"])); m(f"smldiffhi{k}", num(r["ci_high"]))


def rq3_main():
    R = pd.read_csv(ROOT / "experiments" / "rq3_v2.csv")
    smp = {"M": "hand-picked hedge funds", "A": "Form ADV hedge funds", "L": "long-only managers"}
    def cell(k, test, holm=False):
        r = R[(R["sample"] == smp[k]) & (R["test"] == test)].iloc[0]
        c = f"{num(r['mean_pct'], 2, sign=True)} ({num(r['t'])})"
        if holm and not np.isnan(r.get("p_holm", np.nan)):
            c += f" [{r['p_holm']:.3f}]"
        return c
    lines = [r"\begin{table}[t]",
             r"  \caption{RQ3: return spreads before disclosure, from the end of quarter $t$ to the day each",
             r"  fund's 13F for $t$ became public (at most 45 days), in excess of the average universe stock;",
             r"  \% per window, averaged over \rqthreequartersM{} quarters (Newey--West $t$). Primary tests show Holm-adjusted",
             r"  $p$-values within each sample in brackets. Positive values mean surprising trades did better.}",
             r"  \label{tab:rq3main}", r"  \footnotesize", r"  \setlength{\tabcolsep}{1pt}",
             r"  \begin{tabular}{@{}lccc@{}}", r"    \toprule",
             r"     & \multicolumn{2}{c}{Hedge funds} & Long-only \\ \cmidrule(lr){2-3}",
             r"     & Hand-picked & Form ADV & managers \\", r"    \midrule",
             r"    \multicolumn{4}{@{}l}{\emph{Primary}} \\"]
    for label, test in [("Sells (pred. $-$ surpr.)", "sells"), ("Buys (surpr. $-$ pred.)", "buys"),
                        ("Stocks (Q1 $-$ Q5)", "stocks")]:
        lines.append(f"    {label} & " + " & ".join(cell(k, test, True) for k in "MAL") + r" \\")
    def holm_all(k):
        r = R[(R["sample"] == smp[k]) & (R["test"] == "sells")].iloc[0]
        return f"[{r['p_holm_all']:.3f}]"
    lines.append(r"    \quad Holm, all samples & " + " & ".join(holm_all(k) for k in "MAL") + r" \\")
    lines.append(r"    \multicolumn{4}{@{}l}{\emph{Secondary}} \\")
    for label, test in [("All buys $-$ sells", "unconditional"),
                        ("Sells, size--mom. adj.", "sells (size-momentum adjusted)"),
                        ("Sells, intensity", "intensity sells"),
                        ("Sells, within fund", "sells (within fund-quarter)"),
                        ("Sells, after filing", "sells (after filing, copier)"),
                        ("Buys, after filing", "buys (after filing, copier)")]:
        lines.append(f"    {label} & " + " & ".join(cell(k, test) for k in "MAL") + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TAB / "rq3main.tex").write_text("\n".join(lines) + "\n")
    def g(k, test):
        return R[(R["sample"] == smp[k]) & (R["test"] == test)].iloc[0]
    for k in "MAL":
        for key, test in [("pdsells", "sells"), ("pdbuys", "buys"), ("pdstocks", "stocks"), ("pduncond", "unconditional"),
                          ("pdsellsch", "sells (size-momentum adjusted)"), ("pdsellsdl", "sells (missing = delisted, -30%)")]:
            r = g(k, test)
            m(f"{key}{k}", num(r["mean_pct"], 2, sign=True)); m(f"{key}t{k}", num(r["t"]))
        m(f"pdsellsholm{k}", f"{g(k, 'sells')['p_holm']:.3f}")


def data_quality():
    q = pd.read_csv(ROOT / "experiments" / "data_quality.csv").iloc[0]
    m("splitlikepct", num(q["split_like_pct"], 1)); m("cusipchangen", str(int(q["cusip_change_n"])))
    m("cusipchangepct", num(q["cusip_change_pct"], 2))
    m("univtype", num(q["univ_removed_type"], 0)); m("univholders", num(q["univ_removed_holders"], 0))
    m("univnonequity", num(q["univ_removed_nonequity"], 0))
    m("filedthirtyone", num(q["filed_within31_pct"], 1)); m("filedfortyfive", num(q["filed_within45_pct"], 0))
    m("filinglagmedian", str(int(q["filing_lag_median"])))


def controls():
    C = pd.read_csv(ROOT / "experiments" / "positive_controls.csv")
    C = C[C["check"] != "synthetic injection"]  # superseded by the fund-level design (injection_test.py)
    I = pd.read_csv(ROOT / "experiments" / "injection_test.csv").groupby(["share_of_funds", "noise"]).mean(numeric_only=True)
    lines = [r"\begin{table}[t]",
             r"  \caption{Can the pipeline find predictability where it exists? Top: a random 10\% or 25\% of",
             r"  hand-picked funds is made to follow a visible rule (BUY if the stock's four-quarter momentum is",
             r"  positive), exactly or for 80\% of positions; mean test accuracy of those funds (3 draws). Bottom:",
             r"  gain over the majority baseline (points, 95\% interval) when testing 2017--2019 (training",
             r"  2013Q3--2015) and with the best of 20 LightGBM configurations chosen on 2021.}",
             r"  \label{tab:controls}", r"  \footnotesize", r"  \setlength{\tabcolsep}{3pt}",
             r"  \begin{tabular}{@{}lccc@{}}", r"    \toprule",
             r"    Injected funds & Baseline & Model & Rule (ceiling) \\", r"    \midrule"]
    for (sh, nz), r in I.iterrows():
        lab = f"{int(sh * 100)}\\% of funds, " + ("exact rule" if nz == 0 else "rule on 80\\%")
        lines.append(f"    {lab} & {r['acc_base']:.1f}\\% & {r['acc_model']:.1f}\\% & {r['acc_rule']:.1f}\\% \\\\")
    lines += [r"    \midrule", r"    Sample & Earlier period & Tuned & Main \\", r"    \midrule"]
    short = {"hedge funds, hand-picked": ("HF hand-picked", "M"), "hedge funds, Form ADV": ("HF Form ADV", "A"),
             "long-only managers": ("Long-only", "L")}
    for smp, (lab, kk) in short.items():
        e = C[(C["check"] == "earlier period") & (C["sample"] == smp)].iloc[0]
        t = C[(C["check"].str.startswith("tuned")) & (C["sample"] == smp)].iloc[0]
        bsm = exp(kk, "rq1_bootstrap.csv")
        f = bsm[(bsm["resampling"] == "by fund") & (bsm["metric"] == "mean per-fund accuracy")].iloc[0]
        cell = lambda r: f"{num(r['gain'], sign=True)} [{num(r['gain_lo'])}, {num(r['gain_hi'])}]"
        lines.append(f"    {lab} & {cell(e)} & {cell(t)} & {num(f['gain_points'], sign=True)} [{num(f['ci95_low'])}, {num(f['ci95_high'])}] \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TAB / "controls.tex").write_text("\n".join(lines) + "\n")
    m("injrecovered", num(I["recovered_share_pct"].min(), 0))
    m("injbaseten", num(I.loc[(0.10, 0.0), "acc_base"])); m("injmodelten", num(I.loc[(0.10, 0.0), "acc_model"]))
    m("injbasenoisy", num(I.loc[(0.10, 0.2), "acc_base"])); m("injmodelnoisy", num(I.loc[(0.10, 0.2), "acc_model"]))
    m("injrulenoisy", num(I.loc[(0.10, 0.2), "acc_rule"]))
    for k, smp in [("M", "hedge funds, hand-picked"), ("A", "hedge funds, Form ADV"), ("L", "long-only managers")]:
        e = C[(C["check"] == "earlier period") & (C["sample"] == smp)].iloc[0]
        t = C[(C["check"].str.startswith("tuned")) & (C["sample"] == smp)].iloc[0]
        m(f"early{k}", num(e["gain"], sign=True)); m(f"earlylo{k}", num(e["gain_lo"])); m(f"earlyhi{k}", num(e["gain_hi"]))
        m(f"earlyerr{k}", num(e["error_reduction_pct"], 0))
        m(f"tuned{k}", num(t["gain"], sign=True)); m(f"tunedlo{k}", num(t["gain_lo"])); m(f"tunedhi{k}", num(t["gain_hi"]))
    X = pd.read_csv(ROOT / "experiments" / "inference_extra.csv")
    for k in "MALNP":
        r = X[(X["sample"] == k) & X["test"].str.startswith("two-way")].iloc[0]
        m(f"twowaylo{k}", num(r["lo"])); m(f"twowayhi{k}", num(r["hi"]))
    r = X[X["test"] == "gain within 2022-2023"].iloc[0]
    m("newswindowgain", num(r["gain"], sign=True)); m("newswindowlo", num(r["lo"])); m("newswindowhi", num(r["hi"]))
    R = pd.read_csv(ROOT / "experiments" / "rq3_v2.csv")
    smp = {"M": "hand-picked hedge funds", "A": "Form ADV hedge funds", "L": "long-only managers"}
    for k, sname in smp.items():
        g = lambda t: R[(R["sample"] == sname) & (R["test"] == t)].iloc[0]
        for key, t in [("tqsells", "sells (trade quarter)"), ("tqbuys", "buys (trade quarter)")]:
            m(f"{key}{k}", num(g(t)["mean_pct"], 2, sign=True)); m(f"{key}t{k}", num(g(t)["t"]))
        mg = g("manager level: Spearman(gain, trade payoff)")
        m(f"mgrrho{k}", num(mg["mean_pct"], 2, sign=True)); m(f"mgrlo{k}", num(mg["ci_low"], 2)); m(f"mgrhi{k}", num(mg["ci_high"], 2))
        m(f"mgrn{k}", str(int(mg["count"])))
        m(f"pdsellsholmall{k}", f"{g('sells')['p_holm_all']:.3f}")
        for key, t in [("wfsells", "sells (within fund-quarter)"), ("cpsells", "sells (after filing, copier)"),
                       ("cpbuys", "buys (after filing, copier)")]:
            m(f"{key}{k}", num(g(t)["mean_pct"], 2, sign=True)); m(f"{key}t{k}", num(g(t)["t"]))
        m(f"rqthreequarters{k}", str(int(g("sells")["quarters"])))
    sz = exp("M", "rq3_style_portfolio_sizes.csv").set_index("style")
    nest = 0
    for st in sz.index:
        nest += int(min(sz.loc[st, "predicted_sell"], sz.loc[st, "surprise_sell"]) >= 10)
        nest += int(min(sz.loc[st, "predicted_buy"], sz.loc[st, "surprise_buy"]) >= 10)
    m("nstyletests", str(nest))


def extras():
    L = pd.read_csv(ROOT / "experiments" / "learning_curves.csv")
    tag = {"hedge funds, hand-picked": "M", "hedge funds, Form ADV": "A", "long-only managers": "L"}
    words = {6: "six", 12: "twelve", 20: "twenty"}
    for _, r in L.iterrows():
        k = tag[r["sample"]]
        w = words.get(int(r["train_quarters"]), "all")
        m(f"lc{w}{k}", num(r["gain"], sign=True)); m(f"lc{w}lo{k}", num(r["gain_lo"])); m(f"lc{w}hi{k}", num(r["gain_hi"]))
    m("lcallquarters", str(int(L["train_quarters"].max())))
    G = pd.read_csv(ROOT / "experiments" / "graded_injection.csv").groupby("rule_strength").mean(numeric_only=True)
    for st, w in [(0.55, "ff"), (0.60, "sixty"), (0.70, "seventy")]:
        m(f"grrec{w}", num(G.loc[st, "recovered_pct"], 0)); m(f"gratt{w}", num(G.loc[st, "attainable_gain"], 1))
        m(f"grgain{w}", num(G.loc[st, "model_gain"], 1, sign=True))
    W = pd.read_csv(ROOT / "experiments" / "rq1_weights.csv").set_index("check")
    r = W.loc["excluding multi-strategy and quant"]
    m("exmqgain", num(r["gain"], sign=True)); m("exmqlo", num(r["lo"])); m("exmqhi", num(r["hi"])); m("exmqn", str(int(r["funds"])))
    r = W.loc["dollar-weighted within fund"]
    m("dwgain", num(r["gain"], sign=True)); m("dwlo", num(r["lo"])); m("dwhi", num(r["hi"]))
    Mc = pd.read_csv(ROOT / "experiments" / "mechanism.csv")
    for k, smp in [("M", "hedge funds, hand-picked"), ("A", "hedge funds, Form ADV"), ("L", "long-only")]:
        r = Mc[(Mc["sample"] == smp) & (Mc["test"] == "Spearman: gain vs. fund's median holdings")].iloc[0]
        m(f"rhoholdingslo{k}", num(r["ci_low"], 2)); m(f"rhoholdingshi{k}", num(r["ci_high"], 2))


def write():
    TAB.mkdir(exist_ok=True)
    out = ["% Generated by src/make_numbers.py from the result files. Do not edit by hand."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (PAPER / "numbers.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {len(macros)} macros and {len(list(TAB.glob('*.tex')))} tables")


def main():
    TAB.mkdir(exist_ok=True)
    sample_stats(); rq1(); samples_table(); robust_table(); style_table(); rq2_table(); rq3(); rq3_main(); mechanism(); data_quality(); controls(); extras()
    write()


if __name__ == "__main__":
    main()
