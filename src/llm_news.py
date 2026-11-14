"""Step 12 (Thanksgiving): an LLM reads each stock's news, one quarter at a time.

Why: RQ2 found that FinBERT sentiment, averaged over a quarter, adds nothing.
One explanation is that averaging headline sentiment is too blunt. Here a
local LLM (Llama 3.1 8B via Ollama, training cutoff December 2023) reads up to
15 of the latest headlines for a stock in a quarter and gives two judgements:
  outlook    -2..2, how good the news is for the stock next quarter
  surprise    0..2, how much genuinely new information the news contains
These become features for LightGBM (src/rq2_models.py), exactly like the
FinBERT features: quarter t-1 news for a forecast, quarter t for a nowcast.

Leakage controls
  - Headlines reporting an investor's stake or a 13F/13D/13G filing are
    removed first (same filter as rq2_models.STAKE).
  - Memorization check (--anonymize): the model was trained on text up to
    Dec 2023, which covers all our news. It might "know" what happened to a
    company next. So we also run it with the ticker and company name replaced
    by "the company". If named runs predict next-quarter returns much better
    than anonymized runs, the model is recalling outcomes, not reading news
    (the idea behind the lookahead test of Gao, Jiang & Yan).

Two Macs: the second Mac only needs Ollama with the model, started so it
listens on the home network (OLLAMA_HOST=0.0.0.0 ollama serve). Then run both
shards from THIS Mac, pointing shard 1 at the other Mac with --url. No data
or code has to be copied.
Runs are resumable: finished stock-quarters are skipped on restart.

One-Mac-elsewhere mode (what we use: the LLM runs only on Mom's Mac):
  .venv/bin/python src/llm_news.py --export     # writes tools/llm_runner/jobs.jsonl
  (AirDrop the tools/llm_runner folder to the other Mac, run run_jobs.py there,
   AirDrop results.jsonl back, then:)
  .venv/bin/python src/llm_news.py --import ~/Downloads/results.jsonl

Usage:
  .venv/bin/python src/llm_news.py --limit 30                 # quick test
  .venv/bin/python src/llm_news.py --shard 0 --nshards 2      # Mac 1, all years
  .venv/bin/python src/llm_news.py --shard 1 --nshards 2 --url http://<other-mac>.local:11434   # Mac 2 does the work
  .venv/bin/python src/llm_news.py --anonymize --from-year 2021   # memorization check
Output: data/processed/llm/{named|anon}_shard{i}.jsonl
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = PROC / "llm"
MODEL = "llama3.1:8b"
URL = "http://localhost:11434/api/chat"
MAX_HEADLINES = 15
STAKE = (r"\bstake\b|\bstakes\b|13F|13D|13G|takes position|boosts position|cuts position|"
         r"raises position|lowers position|new position|holdings in|buys shares of|sells shares of|"
         r"increases stake|decreases stake|exits position")
PROMPT = ("You are an equity analyst. Below are news headlines about {t} from one calendar quarter. "
          "Based only on these headlines, reply with JSON only: "
          '{{"outlook": <integer -2 to 2, very negative to very positive for the stock next quarter>, '
          '"surprise": <integer 0 to 2, how much genuinely new information>}}.\nHeadlines:\n{h}')
SUFFIX = r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|PLC|LLC|HOLDINGS?|GROUP|CLASS [A-C]|CL [A-C]|NEW|THE|COM)\b"


def company_names() -> dict[str, list[str]]:
    """cusip -> name variants to hide (from 13F issuer names), e.g. ['APPLE']."""
    H = pd.read_parquet(PROC / "holdings.parquet", columns=["cusip", "issuer"])
    top = H.groupby("cusip")["issuer"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    out = {}
    for c, n in top.items():
        core = re.sub(r"\s+", " ", re.sub(SUFFIX, " ", re.sub(r"[^A-Z0-9& ]", " ", str(n).upper()))).strip()
        if core:
            out[c] = [core] + ([core.split()[0]] if len(core.split()[0]) >= 4 else [])
    return out


def anonymize(text: str, ticker: str, names: list[str]) -> str:
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(n)}\b", "the company", text, flags=re.I)
    if ticker:
        text = re.sub(rf"\b{re.escape(ticker)}\b", "the company", text)
    return text


def ask(prompt: str) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "format": "json", "options": {"temperature": 0, "num_predict": 40, "seed": 7}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    try:
        d = json.loads(r["message"]["content"])
        return {"outlook": int(d.get("outlook")), "surprise": int(d.get("surprise"))}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"outlook": None, "surprise": None, "raw": r["message"]["content"][:200]}


def stock_quarters(from_year: int):
    N = pd.read_parquet(PROC / "news_headlines.parquet", columns=["cusip", "ticker", "quarter", "date", "headline"])
    N = N[(N["quarter"] >= from_year * 4) & ~N["headline"].str.contains(STAKE, case=False, regex=True)]
    return N.sort_values("date").groupby(["cusip", "quarter"], sort=True)


def make_prompt(g: pd.DataFrame, anon: bool, names: dict) -> tuple[str, int]:
    g = g.tail(MAX_HEADLINES)
    ticker = str(g["ticker"].iloc[0])
    heads = list(g["headline"])
    label = ticker
    if anon:
        heads = [anonymize(h, ticker, names.get(g["cusip"].iloc[0], [])) for h in heads]
        label = "a company"
    return PROMPT.format(t=label, h="\n".join("- " + h for h in heads)), len(heads)


def done_keys(kind: str) -> set:
    out = set()
    for f in OUT.glob(f"{kind}_*.jsonl"):
        for line in open(f):
            d = json.loads(line)
            if d.get("outlook") is not None:
                out.add((d["cusip"], d["quarter"]))
    return out


def export_jobs() -> None:
    """All prompts in priority order: the memorization check first (2021-2023,
    named and anonymized), then the remaining named stock-quarters."""
    groups = stock_quarters(2013)
    names = company_names()
    jobs = []
    for kind in ["named", "anon"]:
        skip = done_keys(kind)
        for (cusip, q), g in groups:
            if (cusip, q) in skip or (kind == "anon" and q < 2021 * 4):
                continue
            prompt, n = make_prompt(g, kind == "anon", names)
            jobs.append({"id": f"{kind}|{cusip}|{int(q)}", "kind": kind, "cusip": cusip, "quarter": int(q),
                         "n_headlines": n, "prompt": prompt,
                         "priority": 0 if q >= 2021 * 4 else 1})
    jobs.sort(key=lambda j: (j["priority"], j["quarter"], j["kind"], j["cusip"]))
    d = ROOT / "tools" / "llm_runner"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "jobs.jsonl", "w") as f:
        for j in jobs:
            j.pop("priority")
            f.write(json.dumps(j) + "\n")
    n_mem = sum(1 for j in jobs if j["quarter"] >= 2021 * 4)
    print(f"wrote {len(jobs):,} jobs to tools/llm_runner/jobs.jsonl "
          f"({n_mem:,} for the memorization check first, about {n_mem * 2.9 / 3600:.0f} h; "
          f"all about {len(jobs) * 2.9 / 3600:.0f} h at 2.9 s each)")


def import_results(path: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(path)]
    for kind in ["named", "anon"]:
        sel = [r for r in rows if r["kind"] == kind]
        with open(OUT / f"{kind}_othermac.jsonl", "w") as f:
            for r in sel:
                f.write(json.dumps({k: r[k] for k in r if k not in ("kind", "id")}) + "\n")
        bad = sum(r.get("outlook") is None for r in sel)
        print(f"{kind}: {len(sel):,} results imported ({bad} unreadable answers)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--from-year", type=int, default=2013)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", default="http://localhost:11434", help="Ollama server (another Mac on the home network works)")
    ap.add_argument("--export", action="store_true", help="write all prompts to tools/llm_runner/jobs.jsonl instead of running")
    ap.add_argument("--import", dest="import_path", help="bring results.jsonl from the other Mac into data/processed/llm/")
    a = ap.parse_args()
    if a.export:
        export_jobs()
        return
    if a.import_path:
        import_results(Path(a.import_path))
        return
    global URL
    URL = a.url.rstrip("/") + "/api/chat"

    N = pd.read_parquet(PROC / "news_headlines.parquet", columns=["cusip", "ticker", "quarter", "date", "headline"])
    N = N[(N["quarter"] >= a.from_year * 4) & ~N["headline"].str.contains(STAKE, case=False, regex=True)]
    N = N.sort_values("date")
    groups = N.groupby(["cusip", "quarter"], sort=True)
    keys = [k for i, k in enumerate(groups.groups) if i % a.nshards == a.shard]
    if a.limit:
        keys = keys[:a.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{'anon' if a.anonymize else 'named'}_shard{a.shard}.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            done.add((d["cusip"], d["quarter"]))
    names = company_names() if a.anonymize else {}
    todo = [k for k in keys if k not in done]
    print(f"{len(keys):,} stock-quarters in this shard, {len(done):,} done, {len(todo):,} to go -> {path.name}")

    t0 = time.time()
    with open(path, "a") as f:
        for i, (cusip, q) in enumerate(todo, 1):
            g = groups.get_group((cusip, q)).tail(MAX_HEADLINES)
            ticker = str(g["ticker"].iloc[0])
            heads = list(g["headline"])
            label = ticker
            if a.anonymize:
                heads = [anonymize(h, ticker, names.get(cusip, [])) for h in heads]
                label = "a company"
            res = ask(PROMPT.format(t=label, h="\n".join("- " + h for h in heads)))
            f.write(json.dumps({"cusip": cusip, "quarter": int(q), "n_headlines": len(heads), **res}) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i:,}/{len(todo):,}  {rate:.1f}s each, about {rate * (len(todo) - i) / 3600:.1f} h left", flush=True)


if __name__ == "__main__":
    main()
