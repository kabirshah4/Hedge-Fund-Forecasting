"""Step 6b: score each headline's tone with FinBERT.

FinBERT (ProsusAI/finbert) is a transformer language model fine-tuned on
financial text. For each headline it gives three probabilities:
positive / negative / neutral. We store all three, and a single score
  sentiment = P(positive) - P(negative)     (from -1 to +1)

Runs on the Mac's GPU (Apple "MPS") when available. Takes a year range so
the work can be split between two Macs, and saves after every chunk so it
can be stopped and resumed.

Usage:
  .venv/bin/python src/score_news.py 2013 2018     # this Mac
  .venv/bin/python src/score_news.py 2019 2023     # the other Mac
  .venv/bin/python src/score_news.py 2013 2013 --limit 2000   # speed test

Output (one file per year):
  data/processed/news_scores/YYYY.parquet  cusip, date, quarter, p_pos, p_neg, p_neu, sentiment
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = PROC / "news_scores"
MODEL = "ProsusAI/finbert"
BATCH = 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("--limit", type=int, default=0, help="score only this many (speed test)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(3)]
    print(f"device {device}; label order {labels}")

    N = pd.read_parquet(PROC / "news_headlines.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    for year in range(args.start, args.end + 1):
        dest = OUT / f"{year}.parquet"
        if dest.exists() and not args.limit:
            print(f"{year}: already done")
            continue
        Y = N[N["date"].dt.year == year].reset_index(drop=True)
        if args.limit:
            Y = Y.head(args.limit)
        probs, t0 = [], time.time()
        with torch.inference_mode():
            for i in range(0, len(Y), BATCH):
                enc = tok(list(Y["headline"].iloc[i : i + BATCH]), padding=True, truncation=True,
                          max_length=64, return_tensors="pt").to(device)
                probs.append(torch.softmax(model(**enc).logits, dim=-1).float().cpu())
                if (i // BATCH) % 200 == 0 and i:
                    rate = (i + BATCH) / (time.time() - t0)
                    print(f"  {year}: {i + BATCH:,}/{len(Y):,}  ({rate:,.0f} headlines/s)")
        P = torch.cat(probs).numpy()
        S = Y[["cusip", "date", "quarter"]].copy()
        for j, name in enumerate(labels):
            S[f"p_{name[:3]}"] = P[:, j]
        S["sentiment"] = S["p_pos"] - S["p_neg"]
        rate = len(Y) / (time.time() - t0)
        if args.limit:
            print(f"speed test: {len(Y):,} headlines at {rate:,.0f}/s")
            print(pd.concat([Y["headline"].str[:70], S[["p_pos", "p_neg", "sentiment"]].round(2)], axis=1).head(8).to_string())
            return
        S.to_parquet(dest, index=False)
        print(f"{year}: {len(S):,} headlines scored at {rate:,.0f}/s")


if __name__ == "__main__":
    main()
