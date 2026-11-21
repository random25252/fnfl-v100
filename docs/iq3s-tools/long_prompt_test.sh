#!/bin/bash
# a ~2,000-token chat prompt (a numbered fact list, then a question about it), and a short follow-up
M=$1; LOG=$2; PORT=18573
cd ${FNFL_DIR:-/home/dom/fnfl}
./build-fnfl/bin/llama-server --host 127.0.0.1 --port $PORT -np 1 --jinja -m "$M" \
  -md /home/dom/fnfl-models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --fnfl-draft-subset /home/dom/fnfl/models/draft-vocab-qwen3.8-flash-next.txt -c 8192 > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 900); do kill -0 $PID 2>/dev/null || { echo EXITED; tail -3 "$LOG"; exit 1; }; curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break; sleep 1; done
python3 -I - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
facts = "\n".join(f"Fact {i}: the locker numbered {i} holds a {['red','green','blue','yellow','violet'][i%5]} ball and {i*7%13} coins." for i in range(1, 121))
q = facts + "\n\nQuestion: what colour is the ball in locker 42, and how many coins are in locker 42? Answer in one short sentence."
def ask(msgs, n=96):
    body = json.dumps({"messages": msgs, "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    if "error" in r: print("ERROR", r["error"]); return None
    t = r.get("timings", {}); u = r.get("usage", {})
    print(f"prompt {u.get('prompt_tokens')} tok, {t.get('prompt_per_second', 0):.0f} tok/s prefill; gen {u.get('completion_tokens')} tok at {t.get('predicted_per_second', 0):.1f} tok/s")
    return r["choices"][0]["message"]["content"]
a = ask([{"role": "user", "content": q}]); print("ANSWER:", repr(a))
b = ask([{"role": "user", "content": "Write a haiku about the sea, then explain its imagery in two sentences."}], 160); print("HAIKU:", repr(b))
PY
grep -E "engine error|error word" "$LOG" | head -3
kill $PID; wait $PID 2>/dev/null
