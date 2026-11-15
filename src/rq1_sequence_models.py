"""Step 10d: stronger and more comparable RQ1 models (answers reviewer W1, W4c, W4d).

1. Simple baselines on the main split (train 2013-2020, val 2021, test 2022+):
     rule       BUY if the fund bought more than it sold last quarter, else SELL
     logistic   pooled multinomial logistic regression on the same features
                (fund identity one-hot)
2. An LSTM, the model family of Cohen, Lu & Nguyen: each position's last 8
   quarters (action, log change in shares, log portfolio weight, held flag)
   go through an LSTM, then join the static features and a fund embedding.
   Pooled across funds; early stopping on 2021.
3. Their rolling, per-fund protocol (20 quarters train, next 8 test, four
   blocks) with per-fund LOGISTIC REGRESSION and per-fund LSTM, alongside the
   per-fund LightGBM of robustness.py. Also reports how many training
   examples the median fund has, since per-fund models may simply overfit.

Output: <experiments>/rq1_sequence_models.csv     (SAMPLE=... for other samples)
Usage:  .venv/bin/python src/rq1_sequence_models.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn

import baselines as BS
import robustness as RB
import sample as SMP

SEQ = 8
torch.manual_seed(7)
np.random.seed(7)
NUM = [f for f in BS.FEATURES if f != "fund_id"]


def sequences(T: pd.DataFrame) -> np.ndarray:
    """(n, SEQ, 7): per lag k = SEQ..1 (oldest first): one-hot action (3), missing flag,
    log(1 + change), log weight, held flag."""
    L = pd.read_parquet(SMP.PROC / "labels.parquet", columns=["fund", "cusip", "period", "label", "change_pct"])
    L["q"] = L["period"].dt.year * 4 + (L["period"].dt.month - 1) // 3
    L["a"] = L["label"].map(BS_CODE)
    L["lc"] = np.log1p(L["change_pct"].clip(-0.99, 9))
    H = pd.read_parquet(SMP.PROC / "holdings.parquet", columns=["fund", "period", "cusip", "value_usd"])
    H = H[H["period"].dt.is_quarter_end]
    H["q"] = H["period"].dt.year * 4 + (H["period"].dt.month - 1) // 3
    H = H.groupby(["fund", "q", "cusip"], as_index=False)["value_usd"].sum()
    H["lw"] = np.log((H["value_usd"] / H.groupby(["fund", "q"])["value_usd"].transform("sum")).clip(lower=1e-7))
    key = T[["fund", "cusip", "q"]].reset_index(drop=True)
    X = np.zeros((len(T), SEQ, 7), dtype=np.float32)
    for j, k in enumerate(range(SEQ, 0, -1)):
        a = key.merge(L[["fund", "cusip", "q", "a", "lc"]].assign(q=L["q"] + k), on=["fund", "cusip", "q"], how="left")
        w = key.merge(H[["fund", "cusip", "q", "lw"]].assign(q=H["q"] + k), on=["fund", "cusip", "q"], how="left")
        act = a["a"].to_numpy()
        for c in range(3):
            X[:, j, c] = (act == c)
        X[:, j, 3] = np.isnan(act)
        X[:, j, 4] = np.nan_to_num(a["lc"].to_numpy())
        X[:, j, 5] = np.nan_to_num(w["lw"].to_numpy(), nan=-16.0) / 16.0
        X[:, j, 6] = ~np.isnan(w["lw"].to_numpy())
    return X


BS_CODE = {"SELL": 0, "HOLD": 1, "BUY": 2}


def static_matrix(T: pd.DataFrame, fit=None):
    """Standardized static features + missing flags. `fit` = (medians, scaler)
    from the training rows, reused for every other row."""
    Z = T[NUM].astype(float)
    miss = Z.isna().astype(np.float32).to_numpy()
    if fit is None:
        med = Z.median()
        scaler = StandardScaler().fit(Z.fillna(med).to_numpy())
        fit = (med, scaler)
    med, scaler = fit
    Z = np.clip(np.nan_to_num(scaler.transform(Z.fillna(med).to_numpy())), -6, 6)
    return np.hstack([Z, miss]).astype(np.float32), fit


class Net(nn.Module):
    def __init__(self, n_static, n_funds, hidden=32, emb=8):
        super().__init__()
        self.lstm = nn.LSTM(7, hidden, batch_first=True)
        self.emb = nn.Embedding(n_funds, emb) if n_funds > 1 else None
        self.head = nn.Sequential(nn.Linear(hidden + n_static + (emb if n_funds > 1 else 0), 64), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(64, 3))

    def forward(self, seq, stat, fid):
        _, (h, _) = self.lstm(seq)
        parts = [h[-1], stat] + ([self.emb(fid)] if self.emb is not None else [])
        return self.head(torch.cat(parts, dim=1))


def train_lstm(Xs, Zs, F, y, val=None, epochs=30, hidden=32, n_funds=1, lr=2e-3, bs=512):
    net = Net(Zs.shape[1], n_funds, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
    Xs, Zs, F, y = t(Xs), t(Zs), t(F, torch.long), t(y, torch.long)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y))
        for b in range(0, len(y), bs):
            i = perm[b:b + bs]
            opt.zero_grad()
            loss_fn(net(Xs[i], Zs[i], F[i]), y[i]).backward()
            opt.step()
        if val is not None:
            net.eval()
            with torch.no_grad():
                vl = loss_fn(net(t(val[0]), t(val[1]), t(val[2], torch.long)), t(val[3], torch.long)).item()
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 3:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict(net, Xs, Zs, F, proba=False):
    with torch.no_grad():
        out = net(torch.as_tensor(Xs), torch.as_tensor(Zs), torch.as_tensor(F, dtype=torch.long))
        return torch.softmax(out, 1).numpy() if proba else out.argmax(1).numpy()


def auc_bs(sub, pr):
    """BUY-vs-SELL AUC from p(BUY) - p(SELL), HOLD rows dropped."""
    from sklearn.metrics import roc_auc_score
    k = (sub["label"] != "HOLD").to_numpy()
    return roc_auc_score((sub["label"].to_numpy()[k] == "BUY").astype(int), pr[k, 2] - pr[k, 0])


def main() -> None:
    T = pd.read_parquet(SMP.PROC / "features.parquet").reset_index(drop=True)
    X = sequences(T)
    fid = T["fund"].astype("category").cat.codes.to_numpy()
    y = T["y"].to_numpy()
    tr = (T["period"] <= BS.TRAIN_END).to_numpy()
    va = ((T["period"] > BS.TRAIN_END) & (T["period"] <= BS.VAL_END)).to_numpy()
    te = (T["period"] > BS.VAL_END).to_numpy()
    test = T[te]
    rows = []

    def add(check, model, pred, sub=test, block="test 2022-2026", auc=np.nan):
        rows.append({"check": check, "block": block, "model": model, "auc_buy_sell": auc,
                     "mean_fund_accuracy": RB.fund_mean_acc(sub, pred), "accuracy": float((pred == sub["y"]).mean()),
                     "macro_f1": f1_score(sub["y"], pred, average="macro")})

    maj = RB.majority_pred(T[tr], test)
    add("main split", "majority (per fund)", maj)
    rule = np.where(test["fund_buy_share"].fillna(0.5) > 0.5, 2, 0)
    add("main split", "rule: BUY if the fund mostly bought last quarter", rule)
    Ztr, sc = static_matrix(T[tr])
    Zall, _ = static_matrix(T, sc)
    onehot = np.eye(fid.max() + 1, dtype=np.float32)[fid]
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(np.hstack([Zall[tr], onehot[tr]]), y[tr])
    plr = lr.predict_proba(np.hstack([Zall[te], onehot[te]]))
    add("main split", "logistic regression (pooled)", plr.argmax(1), auc=auc_bs(test, plr))
    net = train_lstm(X[tr], Zall[tr], fid[tr], y[tr], val=(X[va], Zall[va], fid[va], y[va]),
                     epochs=40, hidden=32, n_funds=fid.max() + 1)
    pls = predict(net, X[te], Zall[te], fid[te], proba=True)
    add("main split", "LSTM (pooled, 8-quarter sequences)", pls.argmax(1), auc=auc_bs(test, pls))
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

    # rolling per-fund protocol of Cohen, Lu & Nguyen
    qs = sorted(T["q"].unique())
    ntrain = []
    for b in range(20, len(qs) - 7, 8):
        trq, teq = qs[b - 20:b], qs[b:b + 8]
        mtr, mte = T["q"].isin(trq).to_numpy(), T["q"].isin(teq).to_numpy()
        sub = T[mte]
        y0, q0 = divmod(int(teq[0]), 4)
        y1, q1 = divmod(int(teq[-1]), 4)
        block = f"{y0}Q{q0 + 1}-{y1}Q{q1 + 1}"
        majb = RB.majority_pred(T[mtr], sub)
        p_lr, p_ls = majb.copy(), majb.copy()
        sub_idx = np.where(mte)[0]
        for f in sub["fund"].unique():
            itr = np.where(mtr & (T["fund"] == f).to_numpy())[0]
            ite = np.where(mte & (T["fund"] == f).to_numpy())[0]
            ntrain.append(len(itr))
            if len(itr) < 200 or len(np.unique(y[itr])) < 2:
                continue
            Zf, scf = static_matrix(T.iloc[itr])
            Zt, _ = static_matrix(T.iloc[ite], scf)
            pos = np.searchsorted(sub_idx, ite)
            p_lr[pos] = LogisticRegression(max_iter=2000, C=0.5).fit(Zf, y[itr]).predict(Zt)
            netf = train_lstm(X[itr], Zf, np.zeros(len(itr)), y[itr], epochs=15, hidden=16)
            p_ls[pos] = predict(netf, X[ite], Zt, np.zeros(len(ite)))
        add("their protocol", "majority (per fund)", majb, sub, block)
        add("their protocol", "logistic regression per fund", p_lr, sub, block)
        add("their protocol", "LSTM per fund", p_ls, sub, block)
        print(f"  block {block} done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(SMP.EXP / "rq1_sequence_models.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))
    print("\nprotocol, averaged over blocks:",
          (R[R["check"] == "their protocol"].groupby("model")["mean_fund_accuracy"].mean() * 100).round(1).to_dict())
    print(f"training examples per fund-window: median {int(np.median(ntrain))}, "
          f"25th pct {int(np.percentile(ntrain, 25))}, 75th pct {int(np.percentile(ntrain, 75))}")


if __name__ == "__main__":
    main()
