#!/bin/bash
# Draft acceptance: base IQ3_XXS vs Swift IQ3_S, at draft depth 2 and 4. Same prompts (bench.py), reps 2.
S=${1:?usage: accept_compare.sh <output dir>}; mkdir -p "$S"; PORT=18575
BASE=/data/fnfl-models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
SWIFT=/home/dom/Strata-data/models/swift/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
cd /home/dom/fnfl
for cfg in "base $BASE 2" "base $BASE 4" "swift $SWIFT 2" "swift $SWIFT 4"; do
  set -- $cfg; LABEL=$1-d$3; OUT=$S/$LABEL.txt; LOG=$S/$LABEL.server.log
  echo "== $LABEL start $(date +%T)"
  ./build-fnfl/bin/llama-server --host 127.0.0.1 --port $PORT -np 1 --jinja -m $2 \
    -md /home/dom/fnfl-models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
    --fnfl-draft-subset /home/dom/fnfl/models/draft-vocab-qwen3.8-flash-next.txt \
    -c 32768 --fnfl-spec-d $3 --fnfl-draft-attention sparse > $LOG 2>&1 &
  PID=$!
  for i in $(seq 1 1200); do kill -0 $PID 2>/dev/null || { echo "SERVER EXITED"; tail -5 $LOG; continue 2; }
    curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break; sleep 1; done
  echo "ready after ${i}s"
  python3 -I /home/dom/fnfl/docs/iq3s-tools/bench.py $PORT $LOG "$LABEL" 2 > $OUT 2>&1
  tail -14 $OUT
  kill $PID; wait $PID 2>/dev/null; sleep 3
done
echo ALL-DONE
