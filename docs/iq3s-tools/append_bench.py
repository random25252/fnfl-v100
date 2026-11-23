#!/usr/bin/env python3
"""Append TTFT bench: short additions to a long resident context, against a running FNFL server.

Usage: append_bench.py PORT LABEL [DEPTH] [LENGTHS] [REPS] [--corpus-dir DIR]

Builds a context of DEPTH tokens (real text from this repo, as ttft_bench.py), prefills it once, then for each
length sends context + a fresh suffix of that many tokens (n_predict 1, streamed). Each suffix diverges right after
the context, so the server restores the context's end checkpoint and prefills only the suffix: the agent-turn
pattern. The context is re-sent between appends so its checkpoint stays among the slot's eight.
TTFT is the client's time to the first streamed token; prompt_ms the server's.
"""
import json, os, random, statistics, sys, time, urllib.request

args = list(sys.argv[1:])
corpus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if "--corpus-dir" in args:
    i = args.index("--corpus-dir"); corpus_dir = args[i + 1]; del args[i:i + 2]
PORT, LABEL = args[0], args[1]
DEPTH = int(args[2]) if len(args) > 2 else 30000
LENGTHS = [int(x) for x in (args[3] if len(args) > 3 else "32,64,128,256,384,512,640,1024").split(",")]
REPS = int(args[4]) if len(args) > 4 else 3
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as f:
        return json.loads(f.read())


def corpus_tokens(need):
    toks = []
    for root, _, files in sorted(os.walk(corpus_dir)):
        if any(p in root for p in ("/build", "/.git", "/models", "/node_modules")):
            continue
        for fn in sorted(files):
            if fn.endswith((".cpp", ".h", ".cu", ".cuh", ".md", ".py")):
                try:
                    toks += post("/tokenize", {"content": open(os.path.join(root, fn), errors="replace").read()})["tokens"]
                except OSError:
                    pass
                if len(toks) >= need:
                    return toks
    return toks


def send(prompt):
    body = {"prompt": prompt, "n_predict": 1, "temperature": 0, "stream": True, "cache_prompt": True}
    req = urllib.request.Request(BASE + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; timings = {}
    with urllib.request.urlopen(req, timeout=3600) as f:
        for line in f:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ttft is None and (ev.get("content") or ev.get("tokens")):
                ttft = time.perf_counter() - t0
            if "timings" in ev:
                timings = ev["timings"]
    return (ttft if ttft is not None else time.perf_counter() - t0), timings


toks = corpus_tokens(DEPTH + (max(LENGTHS) + 10) * (len(LENGTHS) * REPS + 2) + 1000)
rng = random.Random(777)
ctx = toks[:DEPTH]
pool = toks[DEPTH:]
t, tm = send(ctx)
print(f"# context {DEPTH} tokens: first read {t:.3f} s (server prompt_n {tm.get('prompt_n')})", flush=True)
rows = []
off = 0
for n in [64] + [n for n in LENGTHS for _ in range(REPS)]:  # the first 64 is a warm-up
    suffix = pool[off:off + n]; off += n
    t, tm = send(ctx + suffix)
    r = {"label": LABEL, "depth": DEPTH, "n": n, "ttft_s": round(t, 4), "srv_prompt_ms": round(tm.get("prompt_ms", 0), 1),
         "srv_prompt_n": tm.get("prompt_n")}
    print(json.dumps(r), flush=True)
    rows.append(r)
    send(ctx)  # keep the context's checkpoint fresh (and the slot back at the context)
rows = rows[1:]
print(f"\n== {LABEL}: appends after {DEPTH} tokens, medians over {REPS}")
print(f"{'append':>7} {'TTFT s':>8} {'server ms':>10} {'server n':>9}")
for n in LENGTHS:
    g = [r for r in rows if r["n"] == n]
    print(f"{n:>7} {statistics.median(r['ttft_s'] for r in g):>8.3f} {statistics.median(r['srv_prompt_ms'] for r in g):>10.1f} "
          f"{statistics.median(r['srv_prompt_n'] or 0 for r in g):>9.0f}")
print("BENCH-DONE", flush=True)
