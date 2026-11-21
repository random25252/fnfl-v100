#!/usr/bin/env python3
"""FNFL IQ3_S baseline: prefill and decode separately, by prompt length and output kind.

Usage: fnfl_iq3s_bench.py <port> <server log> <label> [reps]
Prints one JSON line per request and a summary table; the server's log gives each request's draft acceptance
("FNFL request: ... accepted drafts a speculative step X").
"""
import json, re, statistics, sys, time, urllib.request

PORT, LOG, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
BASE = f"http://127.0.0.1:{PORT}"

PARA = ("Gateway power: The fog came in off the water and the keeper counted the seconds between the horn and the "
        "echo, a habit that had outlived the reason for it. ")
TASKS = {
    "tech": "Write a detailed technical explanation of speculative decoding. Be thorough.",
    "code": "Write a complete, well-commented Python implementation of an LRU cache with TTL expiry, plus unit tests.",
    "story": "Write an original short story about a lighthouse keeper who discovers something unexpected.",
}
# filler repeats per target prompt size (~41 tokens per PARA)
SIZES = {"30": 0, "2k": 48, "8k": 195, "25k": 610}
MAXTOK = 400


def accepted_lines():
    try:
        return [l for l in open(LOG, errors="replace") if l.startswith("FNFL request:")]
    except OSError:
        return []


def run(size, task):
    filler = (PARA * SIZES[size]).strip()
    prompt = (filler + "\n\n" if filler else "") + TASKS[task]
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    before = len(accepted_lines())
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as f:
        r = json.loads(f.read())
    wall = time.time() - t0
    t, u = r.get("timings", {}), r.get("usage", {})
    acc = None
    for _ in range(20):  # the request's log line lands just after the response
        lines = accepted_lines()
        if len(lines) > before:
            m = re.search(r"accepted drafts a speculative step ([\d.]+)", lines[-1])
            acc = float(m.group(1)) if m else None
            break
        time.sleep(0.1)
    out = {"label": LABEL, "size": size, "task": task, "prompt_tok": u.get("prompt_tokens"),
           "gen_tok": u.get("completion_tokens"), "prefill_tps": round(t.get("prompt_per_second", 0), 1),
           "decode_tps": round(t.get("predicted_per_second", 0), 1), "accepted_per_step": acc,
           "wall_s": round(wall, 2), "finish": r["choices"][0].get("finish_reason")}
    print(json.dumps(out), flush=True)
    return out


run("30", "tech")  # warm-up, not counted
rows = [run(size, task) for size in SIZES for task in TASKS for _ in range(REPS)]
print(f"\n== {LABEL}: medians over {REPS} reps", flush=True)
print(f"{'size':>5} {'task':>6} {'prompt':>7} {'prefill t/s':>12} {'decode t/s':>11} {'accepted/step':>14}")
for size in SIZES:
    for task in TASKS:
        g = [r for r in rows if r["size"] == size and r["task"] == task]
        med = lambda k: statistics.median([r[k] for r in g if r[k] is not None]) if any(r[k] is not None for r in g) else float("nan")
        print(f"{size:>5} {task:>6} {g[0]['prompt_tok']:>7} {med('prefill_tps'):>12.0f} {med('decode_tps'):>11.1f} "
              f"{med('accepted_per_step'):>14.2f}")
print("BENCH-DONE", flush=True)
