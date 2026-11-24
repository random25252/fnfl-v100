import json, urllib.request, os, sys
B="http://127.0.0.1:8190"; H="/home/dom/fnfl/docs/iq3s-tools"
def post(p,b):
    r=urllib.request.Request(B+p,json.dumps(b).encode(),{"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=1800).read())
EDITS=[("rename","ttft_bench.py","Rename the variable `toks` to `tokens` everywhere it appears."),
 ("docstr","append_bench.py","Add a one-line docstring to every function that lacks one."),
 ("guard","gen_instances.py","Make the script exit with a clear error message when an input file is missing."),
 ("const","glu_check.py","Move every numeric literal threshold into a named constant at the top of the file."),
 ("refactor","bench.py","Refactor the script so the summary table printing lives in its own function, keeping behaviour identical."),
 ("argparse","edit_bench.py","Replace the manual sys.argv parsing with argparse, keeping the same arguments and defaults.")]
GEN=[("tech",None,"Write a detailed technical explanation of speculative decoding. Be thorough."),
 ("code",None,"Write a complete, well-commented Python implementation of an LRU cache with TTL expiry, plus unit tests."),
 ("story",None,"Write an original short story about a lighthouse keeper who discovers something unexpected.")]
out=[]
for name,fn,instr in EDITS+GEN:
    if fn: prompt=f"Here is `{fn}`:\n\n```python\n{open(os.path.join(H,fn)).read()}```\n\n{instr} Output the complete updated file in one ```python code block and nothing else."
    else: prompt=instr
    msgs=[{"role":"user","content":prompt}]
    r=post("/v1/chat/completions",{"messages":msgs,"max_tokens":1600,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}})
    c=r["choices"][0]["message"]["content"]
    tp=post("/apply-template",{"messages":msgs,"chat_template_kwargs":{"enable_thinking":False}})["prompt"]
    pid=post("/tokenize",{"content":tp,"add_special":False,"parse_special":True})["tokens"]
    oid=post("/tokenize",{"content":c,"add_special":False})["tokens"]
    kind="edit" if fn else "gen"
    out.append({"name":name,"kind":kind,"prompt":pid,"out":oid,"tps":r["timings"]["predicted_per_second"]})
    print(name,len(pid),len(oid),round(r["timings"]["predicted_per_second"],1),flush=True)
json.dump(out,open(sys.argv[1],"w"))
