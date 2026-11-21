#!/bin/bash
# reference llama.cpp fork run: one completion with top-5 probabilities per position
M=$1; LOG=$2; PORT=18590
cd /data/qwenfork/llama.cpp
./build/bin/llama-server --host 127.0.0.1 --port $PORT -m "$M" -ngl 99 --n-cpu-moe ${NCMOE:-20} -c 512 -np 1 > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 1200); do
  if ! kill -0 $PID 2>/dev/null; then echo "REF SERVER EXITED"; tail -8 "$LOG"; exit 1; fi
  curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break
  sleep 1
done
echo "ref ready after ${i}s"
curl -s -m 1200 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":8,"temperature":0,"n_probs":5}' > "$LOG.resp"
python3 -I -c "
import json,sys; r=json.load(open('$LOG.resp'))
print('TEXT:', repr(r.get('content')))
for p in r.get('completion_probabilities', [])[:4]:
    print('  ', repr(p.get('token')), [(repr(t.get('token')), round(t.get('prob', t.get('logprob', 0)), 4)) for t in p.get('top_probs', p.get('top_logprobs', []))])
"
kill $PID; wait $PID 2>/dev/null
