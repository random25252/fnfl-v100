import json, urllib.request, statistics, sys
B="http://127.0.0.1:8190"
def post(p, b):
    r=urllib.request.Request(B+p, json.dumps(b).encode(), {"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())
src=open('/home/dom/fnfl/docs/iq3s-tools/bench.py').read()+open('/home/dom/fnfl/docs/iq3s-tools/edit_bench.py').read()
ids=post("/tokenize",{"content":src})["tokens"]
print("pool tokens",len(ids))
base=ids[:2000]
res={}
for rep in range(3):
  for n in [1,2,4,5,8,12,16,24,32,48,64,96,128]:
    # fresh base each time: a distinct first token avoids reuse beyond the base
    ctx=[ids[2000+rep]]+base
    post("/completion",{"prompt":ctx,"n_predict":1,"cache_prompt":True,"temperature":0})
    ext=ctx+ids[2100:2100+n]
    j=post("/completion",{"prompt":ext,"n_predict":1,"cache_prompt":True,"temperature":0})
    t=j["timings"]; res.setdefault(n,[]).append((t["prompt_n"],t["prompt_ms"]))
for n,v in res.items():
    pn=[a for a,_ in v]; ms=statistics.median(b for _,b in v)
    print(f"n={n:4d} prompt_n={pn} median ms {ms:7.2f}  ms/row {ms/n:6.2f}")
