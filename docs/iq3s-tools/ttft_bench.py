#!/usr/bin/env python3
"""Cache-busted prefill / TTFT bench against a running FNFL server.

Usage: ttft_bench.py PORT LABEL [LENGTHS] [REPS] [--corpus-dir DIR] [--max-total N]

Every request is a token-exact prompt (/completion with a token array) cut from a different offset of a corpus of
real text (this repo's sources and docs by default), so no two requests share a prefix and no checkpoint can be
reused. n_predict is 1 and the reply streams: TTFT is the client's time from sending to the first streamed token;
the server's own prompt_ms comes with the final event. Prints one JSON line a request and a median table.
LENGTHS: comma list (default 28,256,1024,2048,4096,8192,16384,32768).
"""
import json, os, random, statistics, sys, time, urllib.request

args = [a for a in sys.argv[1:]]
corpus_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if "--corpus-dir" in args:
    i = args.index("--corpus-dir"); corpus_dir = args[i + 1]; del args[i:i + 2]
PORT, LABEL = args[0], args[1]
LENGTHS = [int(x) for x in (args[2] if len(args) > 2 else "28,256,1024,2048,4096,8192,16384,32768").split(",")]
REPS = int(args[3]) if len(args) > 3 else 3
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def corpus_tokens(need):
    texts = []
    for root, _, files in sorted(os.walk(corpus_dir)):
        if any(p in root for p in ("/build", "/.git", "/models", "/node_modules")):
            continue
        for fn in sorted(files):
            if fn.endswith((".cpp", ".h", ".cu", ".cuh", ".md", ".py")):
                try:
                    texts.append(open(os.path.join(root, fn), errors="replace").read())
                except OSError:
                    pass
    toks = []
    for t in texts:  # tokenize file by file (the endpoint is happier with modest bodies)
        toks += post("/tokenize", {"content": t})["tokens"]
        if len(toks) >= need:
            break
    return toks


def run(n, toks, rng):
    off = rng.randrange(0, len(toks) - n - 1)
    prompt = toks[off:off + n]
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
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    out = {"label": LABEL, "n": n, "off": off, "ttft_s": round(ttft, 4), "total_s": round(total, 4),
           "srv_prompt_ms": round(timings.get("prompt_ms", float("nan")), 1),
           "srv_prompt_n": timings.get("prompt_n"), "ttft_tps": round(n / ttft, 1)}
    print(json.dumps(out), flush=True)
    return out


rng = random.Random(12345)
toks = corpus_tokens(max(LENGTHS) * 3 + 10000)
print(f"# corpus {len(toks)} tokens", flush=True)
if len(toks) < max(LENGTHS) * 2:
    sys.exit("corpus too small for the longest length")
run(256, toks, rng)  # warm-up, not counted
rows = [run(n, toks, rng) for n in LENGTHS for _ in range(REPS)]
print(f"\n== {LABEL}: medians over {REPS} reps")
print(f"{'tokens':>7} {'TTFT s':>8} {'TTFT tok/s':>11} {'server ms':>10} {'server tok/s':>13} {'client-server ms':>17}")
for n in LENGTHS:
    g = [r for r in rows if r["n"] == n]
    t = statistics.median(r["ttft_s"] for r in g); s = statistics.median(r["srv_prompt_ms"] for r in g)
    print(f"{n:>7} {t:>8.3f} {n / t:>11.0f} {s:>10.1f} {n / (s / 1000):>13.0f} {t * 1000 - s:>17.1f}")
print("BENCH-DONE", flush=True)
