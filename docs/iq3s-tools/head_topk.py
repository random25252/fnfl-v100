import sys, glob, numpy as np
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
d, model = sys.argv[1], sys.argv[2]
r = GGUFReader(sorted(glob.glob(model))[0]); f = r.fields['tokenizer.ggml.tokens']
tok = lambda i: bytes(f.parts[f.data[i]]).decode('utf-8', 'replace').replace('Ġ', ' ').replace('Ċ', '\\n')
l0 = np.fromfile(glob.glob(f'{d}/head-k0-i0-rows*-logits.f32')[0], np.float32)
l1 = np.fromfile(glob.glob(f'{d}/head-k1-i0-rows*-logits.f32')[0], np.float32)
rows = l0.size // 124160
for b in range(rows):
    lg = np.concatenate([l0.reshape(rows, -1)[b], l1.reshape(rows, -1)[b]]).astype(np.float64)
    lp = lg - (lg.max() + np.log(np.exp(lg - lg.max()).sum()))
    top = np.argsort(-lp)[:5]
    print(f"row {b} (FNFL):", [(repr(tok(i)), round(float(lp[i]), 4)) for i in top])
