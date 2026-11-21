# usage: glucheck.py <dump dir> <layer>   -- the GPU's routed GLU vs ggml-exact dequant (gguf-py) on the same x, ids
import sys, glob, os, numpy as np
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
d, L = sys.argv[1], int(sys.argv[2])
T = {}
for p in sorted(glob.glob('/home/dom/Strata-data/models/swift/*IQ3_S-0000*.gguf')):
    for t in GGUFReader(p).tensors:
        if t.name in (f'blk.{L}.ffn_gate_exps.weight', f'blk.{L}.ffn_up_exps.weight'): T[t.name] = t
G, U = T[f'blk.{L}.ffn_gate_exps.weight'], T[f'blk.{L}.ffn_up_exps.weight']
print(f"layer {L}: gate/up {G.tensor_type.name}, data shape {G.data.shape}")
cache = {}
def W(t, e):
    k = (t.name, e)
    if k not in cache: cache[k] = dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float64)  # [640][2560]
    return cache[k]
for xf in sorted(glob.glob(os.path.join(d, f'glu-L{L}-k*-x.f16'))):
    base = xf[:-6]; k = int(base.split('-k')[1].split('-')[0]); n = int(base.split('-n')[1])
    x = np.fromfile(xf, np.float16).astype(np.float64).reshape(n, 2560)
    ids = np.fromfile(base + '-ids.i32', np.int32).reshape(n, 10)
    glu = np.fromfile(base + '-glu.f16', np.float16).astype(np.float64).reshape(n * 10, 320)
    for off_name, off in (("card rows 320k", 320 * k), ("card rows 0", 0)):
        ref = np.zeros_like(glu)
        # the GLU rows are in route.sorted order: pairs p = 10 b + i sorted by expert (stable)
        order = sorted(range(n * 10), key=lambda q: (ids.ravel()[q], q))
        for j, q in enumerate(order):
            b, e = q // 10, ids.ravel()[q]
            g = W(G, e)[off:off + 320] @ x[b]; u = W(U, e)[off:off + 320] @ x[b]
            ref[j] = g / (1 + np.exp(-g)) * u
        err = np.abs(glu - ref); scale = np.abs(ref).max()
        rel = np.linalg.norm(glu - ref) / max(np.linalg.norm(ref), 1e-30)
        print(f"  card {k} [{off_name}]: n={n} max|ref|={scale:.4g} max|err|={err.max():.4g} rel-L2={rel:.3e} "
              f"nonfinite={(~np.isfinite(glu)).sum()} ids[0]={ids[0].tolist()}")
