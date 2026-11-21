import sys, glob, re, collections
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
ty = {}
for p in sorted(glob.glob('/home/dom/Strata-data/models/swift/*IQ3_S-0000*.gguf')):
    for t in GGUFReader(p).tensors: ty[t.name] = t.tensor_type.name.lower()
src = open('/home/dom/fnfl/fn/kernels/experts/experts_inst.h').read()
dep = src[src.find('#define FN_EXPERTS_GATE_UP_INSTANCES'):src.find('#define FN_EXPERTS_Q4_GATE_UP')]
have_gu = set(re.findall(r'X\((\w+), (\w+), (\w+), \d+\)', dep))
have_dn = set(re.findall(r'X\((\w+), (\w+), \d+\)', dep))
gu = collections.OrderedDict(); dn = collections.OrderedDict()
for L in range(48):
    g = lambda s: ty[f'blk.{L}.{s}.weight']
    gu.setdefault((g('ffn_gate_exps'), g('ffn_gate_shexp'), g('ffn_up_shexp')), []).append(L)
    dn.setdefault((g('ffn_down_exps'), g('ffn_down_shexp')), []).append(L)
def mac(name, items, n):
    w = 120
    out = [f"#define {name}(X)".ljust(w) + " \\"]
    for i, (k, ls) in enumerate(items):
        body = f"    X({', '.join(k)}, {ls[0]})"
        cmt = f" /* layers {','.join(map(str, ls))} */"
        line = (body + cmt)
        out.append(line.ljust(w) + (" \\" if i < len(items) - 1 else ""))
    return "\n".join(out)
full_gu = list(gu.items()); full_dn = list(dn.items())
extra_gu = [(k, ls) for k, ls in full_gu if k not in have_gu and k[0] != 'iq4_xs']
iq4 = [(k, ls) for k, ls in full_gu if k[0] == 'iq4_xs']
extra_dn = [(k, ls) for k, ls in full_dn if k not in have_dn]
print("// IQ3S: the Swift IQ3_S file's expert instances (its GGUF headers, layers 0-47): its profile's full lists")
print(mac("FN_EXPERTS_IQ3S_GATE_UP_INSTANCES", full_gu, 0)); print(mac("FN_EXPERTS_IQ3S_DOWN_INSTANCES", full_dn, 0))
print("// IQ3S: of those, the entries the deployed lists lack (instantiated beside them; the routed IQ4_XS one apart)")
print(mac("FN_EXPERTS_IQ3S_EXTRA_GATE_UP_INSTANCES", extra_gu, 0)); print(mac("FN_EXPERTS_IQ3S_EXTRA_DOWN_INSTANCES", extra_dn, 0))
print("// IQ3S: the routed IQ4_XS triple (layer 47), on the K-quant path as the Q4 candidate's")
print(mac("FN_EXPERTS_IQ4XS_GATE_UP_INSTANCES", iq4, 0))
print("// IQ3S: the target layers' instantiated lists (the deployed and the IQ3_S file's extras)")
print("#define FN_EXPERTS_TARGET_GATE_UP_INSTANCES(X) FN_EXPERTS_GATE_UP_INSTANCES(X) FN_EXPERTS_IQ3S_EXTRA_GATE_UP_INSTANCES(X)")
print("#define FN_EXPERTS_TARGET_DOWN_INSTANCES(X) FN_EXPERTS_DOWN_INSTANCES(X) FN_EXPERTS_IQ3S_EXTRA_DOWN_INSTANCES(X)")
print(f"// counts: full gu {len(full_gu)} dn {len(full_dn)} extra gu {len(extra_gu)} dn {len(extra_dn)} iq4 {len(iq4)}", file=sys.stderr)
