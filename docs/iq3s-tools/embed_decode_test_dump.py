import sys, glob, numpy as np
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
p = sorted(glob.glob('/home/dom/Strata-data/models/swift/*IQ3_S-00001*.gguf'))[0]
t = [t for t in GGUFReader(p).tensors if t.name == 'token_embd.weight'][0]
raw = np.asarray(t.data).view(np.uint8).reshape(248320, 1360)
rng = np.random.default_rng(0)
toks = np.concatenate([[0, 1, 9707, 151643, 248319], rng.integers(0, 248320, 251)]).astype(np.int64)
kv = np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113], np.float32)
rows, ref = [], []
for card in (0, 1):
    for tk in toks:
        r = raw[tk, card*680:(card+1)*680]; rows.append(r)
        out = np.empty(1280, np.float32)
        for b in range(5):
            blk = r[b*136:(b+1)*136]; d = np.float32(blk[0:2].view(np.float16)[0])
            sh = int(blk[2]) | (int(blk[3]) << 8)
            for ib in range(8):
                ls = ((int(blk[4+ib//2]) >> (4*(ib%2))) & 0xf) | (((sh >> (2*ib)) & 3) << 4)
                dl = np.float32(d * np.float32(ls - 32)); q = blk[8+16*ib:24+16*ib]
                o = b*256 + ib*32
                out[o:o+16] = dl * kv[q & 0xf]; out[o+16:o+32] = dl * kv[q >> 4]
        ref.append(out)
np.array(rows, np.uint8).tofile('rows.bin'); np.array(ref, np.float32).tofile('ref.bin')
print(len(rows), "rows dumped")
