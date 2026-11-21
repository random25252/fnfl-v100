import sys, glob, numpy as np
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
p = sorted(glob.glob('/home/dom/Strata-data/models/swift/*IQ3_S-00001*.gguf'))[0]
t = [t for t in GGUFReader(p).tensors if t.name == 'token_embd.weight'][0]
raw = np.asarray(t.data).view(np.uint8).reshape(248320, 1360)
V = 248320
kv = np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113], np.float32)
def iq4xs_row(r):  # correct decode of one full 2560-row
    out = []
    for b in range(10):
        blk = r[b*136:(b+1)*136]
        d = blk[0:2].view(np.float16)[0].astype(np.float32)
        sh = int(blk[2]) | (int(blk[3]) << 8); sl = blk[4:8]; qs = blk[8:136]
        for ib in range(8):
            ls = ((sl[ib//2] >> (4*(ib%2))) & 0xf) | (((sh >> (2*ib)) & 3) << 4)
            dl = d * (int(ls) - 32); q = qs[ib*16:(ib+1)*16]
            out += list(dl*kv[q & 0xf]) + list(dl*kv[q >> 4])
    return np.array(out)
for card in (0, 1):
    slab = np.ascontiguousarray(raw[:, card*680:(card+1)*680]).reshape(-1)  # the card's [V][680] slice
    ids = np.arange(V)
    # what the kernel reads: super-block sb of token id at id*550 + sb*110, d = its first two bytes (fp16)
    d = np.stack([slab[ids*550 + sb*110 : ][:0] if False else slab[(ids*550 + sb*110)[:,None] + np.arange(2)] for sb in range(5)], 1)
    dv = d.copy().view(np.float16).reshape(V, 5).astype(np.float32)
    bad = ~np.isfinite(dv); big = np.abs(dv) > 1.0
    print(f"card {card}: tokens with a non-finite d in any of the 5 IQ3_S-misread blocks: {bad.any(1).sum()} / {V}"
          f"  ({100*bad.any(1).mean():.2f}%);  with |d|>1: {big.any(1).sum()};  median |d| {np.median(np.abs(dv[np.isfinite(dv)])):.3g}")
    # correct d of the real IQ4_XS blocks, for scale
    dc = np.ascontiguousarray(raw[:, card*680:(card+1)*680]).reshape(V,5,136)[:,:,0:2].copy().view(np.float16).astype(np.float32)
    print(f"        real IQ4_XS d: median |d| {np.median(np.abs(dc)):.3g}, max {np.abs(dc).max():.3g}")
for tok in (0, 1, 9707, 151643):
    v = iq4xs_row(raw[tok]); print(f"correct embedding tok {tok}: max|x| {np.abs(v).max():.4g}, rms {np.sqrt((v*v).mean()):.4g}")
