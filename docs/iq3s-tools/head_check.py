import sys, glob, numpy as np
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
d, tag = sys.argv[1], sys.argv[2]
r = [p for p in sorted(glob.glob('/home/dom/Strata-data/models/swift/*IQ3_S-0000*.gguf'))]
W = None
for p in r:
    for t in GGUFReader(p).tensors:
        if t.name == 'output.weight': W = t
raw = np.asarray(W.data)  # [248320][row bytes]
rows = int(tag.split('rows')[1])
x = np.fromfile(f'{d}/{tag}-x.f32', np.float32).reshape(-1, 2560)[:rows].astype(np.float64)
lg = np.fromfile(f'{d}/{tag}-logits.f32', np.float32).reshape(rows, 124160).astype(np.float64)
print(f"{tag}: W {W.tensor_type.name} {raw.shape}; x row norms {np.linalg.norm(x, axis=1).round(3)}; "
      f"x nonfinite {(~np.isfinite(x)).sum()}; logits range {lg.min():.3g}..{lg.max():.3g}")
def ref_rows(idx):
    out = np.empty((len(idx), rows))
    for s in range(0, len(idx), 8192):
        w = dequantize(raw[idx[s:s + 8192]], W.tensor_type).astype(np.float64)
        out[s:s + 8192] = w @ x.T
    return out
V = 248320
for name, idx in (("first half", np.arange(0, V // 2)), ("interleaved even", np.arange(0, V, 2))):
    ref = ref_rows(idx)  # [124160][rows]
    for b in range(rows):
        a, c = lg[b], ref[:, b]
        corr = np.corrcoef(a, c)[0, 1]
        print(f"  {name:16s} row {b}: corr {corr:.6f}  argmax gpu {int(idx[a.argmax()])} ref {int(idx[c.argmax()])}"
              f"  rel-L2 {np.linalg.norm(a - c) / np.linalg.norm(c):.3e}")
