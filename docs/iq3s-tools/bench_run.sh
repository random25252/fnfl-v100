#!/bin/bash
# start FNFL on the Swift IQ3_S file, run the baseline bench, stop the server
LABEL=$1; OUT=$2; PORT=18575; LOG=$OUT.server.log
cd /home/dom/fnfl
./build-fnfl/bin/llama-server --host 127.0.0.1 --port $PORT -np 1 --jinja \
  -m /home/dom/Strata-data/models/swift/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -md /home/dom/fnfl-models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --fnfl-draft-subset /home/dom/fnfl/models/draft-vocab-qwen3.8-flash-next.txt -c 32768 ${EXTRA} > $LOG 2>&1 &
PID=$!
for i in $(seq 1 900); do kill -0 $PID 2>/dev/null || { echo "SERVER EXITED"; tail -5 $LOG; exit 1; }; curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"ok"' && break; sleep 1; done
echo "ready after ${i}s"; grep "PLE warm" $LOG | cut -c1-200
python3 -I /home/dom/fnfl/docs/iq3s-tools/bench.py $PORT $LOG "$LABEL" 3 > $OUT 2>&1
grep "PLE" $LOG | tail -2 | cut -c1-200
kill $PID; wait $PID 2>/dev/null
