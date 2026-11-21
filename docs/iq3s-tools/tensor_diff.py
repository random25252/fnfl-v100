import sys, re, glob
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
def load(pat):
    d = {}
    for p in sorted(glob.glob(pat)):
        for t in GGUFReader(p).tensors: d[t.name] = (t.tensor_type.name, list(t.shape))
    return d
a = load('/home/dom/Strata-data/models/swift/*IQ3_S-0000*.gguf')
b = load('/data/fnfl-models/IQ3_XXS/*.gguf')
print("only in S:", sorted(set(a)-set(b))[:20]); print("only in XXS:", sorted(set(b)-set(a))[:20])
xt = set(v[0] for v in b.values()); st = set(v[0] for v in a.values())
print("XXS types:", sorted(xt)); print("S types:", sorted(st)); print("types in S never in XXS:", sorted(st-xt))
# per (tensor kind, type) pairs in S not seen in XXS for same kind
def kind(n): return re.sub(r'blk\.\d+\.', '', n)
xp = set((kind(n), v[0]) for n, v in b.items())
new = {}
for n, v in a.items():
    if (kind(n), v[0]) not in xp: new.setdefault((kind(n), v[0]), []).append(n)
print("\n(kind,type) combos in S never exercised by XXS:")
for k, ns in sorted(new.items()):
    ls = sorted(int(re.match(r'blk\.(\d+)', n).group(1)) for n in ns if n.startswith('blk.'))
    print("  %-28s %-8s layers=%s" % (k[0], k[1], ls if ls else ns))
print("\nper-layer expert types S | XXS:")
for L in range(64):
    k = f'blk.{L}.ffn_gate_exps.weight'
    if k not in a: continue
    f = lambda d, s: d.get(f'blk.{L}.{s}', ('-',))[0]
    print(f"  {L:2d} g/u/d S={f(a,'ffn_gate_exps.weight')}/{f(a,'ffn_up_exps.weight')}/{f(a,'ffn_down_exps.weight')}  XXS={f(b,'ffn_gate_exps.weight')}/{f(b,'ffn_up_exps.weight')}/{f(b,'ffn_down_exps.weight')}  ssm={'ssm_a' if f'blk.{L}.ssm_a' in a else 'attn'}")
print("token_embd", a['token_embd.weight'], b['token_embd.weight'])
print("ple", a.get('per_layer_token_embd.weight'), b.get('per_layer_token_embd.weight'))
