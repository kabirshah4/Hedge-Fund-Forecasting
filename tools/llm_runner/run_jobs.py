"""Run the LLM prompts on this Mac and save the answers.

Needs only: Ollama (with `ollama pull llama3.1:8b` done) and the python3 that
comes with macOS. No other installs, no project files.

In Terminal, in this folder:
    caffeinate -i python3 run_jobs.py
    caffeinate -i python3 run_jobs.py --until 07:00     # stop by itself at 7 AM

It reads jobs.jsonl, asks the local Ollama, and appends each answer to
results.jsonl. Safe to stop (Ctrl-C) and restart: finished jobs are skipped.
When it prints "ALL DONE", tell Claude (results.jsonl is in this folder).
"""

import datetime as dt
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs.jsonl")
RESULTS = os.path.join(HERE, "results.jsonl")
URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"


def ask(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "format": "json", "options": {"temperature": 0, "num_predict": 40, "seed": 7}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            break
        except Exception as e:  # Ollama restarting, sleep/wake, etc.
            print(f"  Ollama not answering ({e}); retrying in 30 s...", flush=True)
            time.sleep(30)
    else:
        sys.exit("Ollama is not running. Open the Ollama app, then run this again.")
    try:
        d = json.loads(r["message"]["content"])
        return {"outlook": int(d.get("outlook")), "surprise": int(d.get("surprise"))}
    except (ValueError, TypeError):
        return {"outlook": None, "surprise": None, "raw": r["message"]["content"][:200]}


def stop_time():
    """--until HH:MM -> the next time the clock shows HH:MM (tonight or tomorrow)."""
    if "--until" not in sys.argv:
        return None
    hh, mm = map(int, sys.argv[sys.argv.index("--until") + 1].split(":"))
    now = dt.datetime.now()
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t if t > now else t + dt.timedelta(days=1)


def main():
    stop = stop_time()
    if stop:
        print(f"will stop at {stop:%a %H:%M}", flush=True)
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    with open(JOBS) as f:
        jobs = [json.loads(l) for l in f]
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs):,} jobs, {len(done):,} already done, {len(todo):,} to go", flush=True)
    t0 = time.time()
    with open(RESULTS, "a") as out:
        for i, j in enumerate(todo, 1):
            if stop and dt.datetime.now() >= stop:
                left = len(todo) - i + 1
                print(f"Stopping at {stop:%H:%M} as asked. {left:,} jobs left "
                      f"(about {left * 2.9 / 3600:.1f} hours). Run the same command tonight to continue.")
                return
            res = ask(j["prompt"])
            rec = {k: j[k] for k in ("id", "kind", "cusip", "quarter", "n_headlines")}
            rec.update(res)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i:,}/{len(todo):,} done, {rate:.1f} s each, about {rate * (len(todo) - i) / 3600:.1f} hours left", flush=True)
    print("ALL DONE. Tell Claude: the results are in tools/llm_runner/results.jsonl")


if __name__ == "__main__":
    main()
