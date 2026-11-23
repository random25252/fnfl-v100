#!/usr/bin/env python3
"""Edit-heavy decode bench: agent-style turns that rewrite a file already in the context, against a running FNFL server.

Usage: edit_bench.py PORT SERVER_LOG LABEL [REPS] [--temp T]

bench.py's tasks write new text, so they show almost nothing of lookup (n-gram) drafts. Here each request gives a
source file from this repo and asks for the whole file back with one small change (an agent applying an edit), plus
one control task that writes new prose. Chat completions, thinking off, n_predict up to 1,600. Prints one JSON line a
request (decode tok/s, tokens a speculative step and the lookup steps from the server log) and a median table.
"""
import hashlib, json, os, re, statistics, sys, time, urllib.request

args = list(sys.argv[1:])
TEMP = 0.0
if "--temp" in args:
    i = args.index("--temp"); TEMP = float(args[i + 1]); del args[i:i + 2]
PORT, LOG, LABEL = args[0], args[1], args[2]
REPS = int(args[3]) if len(args) > 3 else 2
BASE = f"http://127.0.0.1:{PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))

# (name, file, instruction): each answer copies most of the file
EDITS = [
    ("rename", "ttft_bench.py", "Rename the variable `toks` to `tokens` everywhere it appears."),
    ("docstr", "append_bench.py", "Add a one-line docstring to every function that lacks one."),
    ("guard", "gen_instances.py", "Make the script exit with a clear error message when an input file is missing."),
    ("const", "glu_check.py", "Move every numeric literal threshold into a named constant at the top of the file."),
]
CONTROL = ("prose", None, "Write a detailed technical explanation of how a CPU branch predictor works. Be thorough.")
MAXTOK = 1600


def log_lines():
    try:
        return [l for l in open(LOG, errors="replace") if l.startswith("FNFL request:")]
    except OSError:
        return []


def run(name, fn, instr):
    if fn is None:
        prompt = instr
    else:
        src = open(os.path.join(HERE, fn)).read()
        prompt = (f"Here is `{fn}`:\n\n```python\n{src}```\n\n{instr} Output the complete updated file in one "
                  f"```python code block and nothing else.")
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK, "temperature": TEMP,
            "chat_template_kwargs": {"enable_thinking": False}}
    before = len(log_lines())
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as f:
        r = json.loads(f.read())
    wall = time.time() - t0
    t, u = r.get("timings", {}), r.get("usage", {})
    line = ""
    for _ in range(20):
        ls = log_lines()
        if len(ls) > before:
            line = ls[-1]
            break
        time.sleep(0.1)
    num = lambda pat: (float(m.group(1)) if (m := re.search(pat, line)) else None)
    out = {"label": LABEL, "task": name, "temp": TEMP, "prompt_tok": u.get("prompt_tokens"),
           "gen_tok": u.get("completion_tokens"), "decode_tps": round(t.get("predicted_per_second", 0), 1),
           "tok_per_spec_step": num(r"tokens a speculative step ([\d.]+)"),
           "step_ms": num(r"step wall ms median ([\d.]+)"),
           "lookup_steps": num(r"lookup steps (\d+)"), "spec_steps": num(r"speculative (\d+)"),
           "lookup_tok": num(r"lookup steps \d+ \(proposals a step [\d.]+, accepted [\d.]+, tokens ([\d.]+)"),
           "wall_s": round(wall, 2), "finish": r["choices"][0].get("finish_reason"),
           "content_md5": hashlib.md5(r["choices"][0]["message"]["content"].encode()).hexdigest()}
    print(json.dumps(out), flush=True)
    return out


rows = [run(*task) for _ in range(REPS) for task in EDITS + [CONTROL]]
print(f"\n== {LABEL} (T {TEMP}): medians over {REPS} reps")
print(f"{'task':>7} {'gen':>5} {'decode t/s':>10} {'tok/step':>9} {'step ms':>8} {'lookup steps':>13}")
for name in [e[0] for e in EDITS] + [CONTROL[0]]:
    g = [r for r in rows if r["task"] == name]
    med = lambda k: statistics.median((r[k] or 0) for r in g)
    print(f"{name:>7} {med('gen_tok'):>5.0f} {med('decode_tps'):>10.1f} {med('tok_per_spec_step'):>9.2f} "
          f"{med('step_ms'):>8.2f} {med('lookup_steps'):>6.0f}/{med('spec_steps'):<6.0f}")
ed = [r for r in rows if r["task"] != CONTROL[0]]
print(f"edits, mean decode t/s: {statistics.mean(r['decode_tps'] for r in ed):.1f}")
print("BENCH-DONE", flush=True)
