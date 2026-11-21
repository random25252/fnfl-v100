import sys, re, collections
sys.path.insert(0, '/data/qwenfork/llama.cpp/gguf-py')
from gguf import GGUFReader
for path in sys.argv[1:]:
    r = GGUFReader(path)
    print("==", path.split('/')[-1], len(r.tensors), "tensors")
    hist = collections.Counter(); per = collections.defaultdict(collections.Counter)
    for t in r.tensors:
        ty = t.tensor_type.name
        hist[ty] += 1
        m = re.match(r'blk\.(\d+)\.(.*)', t.name)
        kind = m.group(2) if m else t.name
        per[kind][ty] += 1
        if not m: print("   ", t.name, ty, list(t.shape))
    print(" types:", dict(hist))
    for k in sorted(per): print("  %-32s %s" % (k, dict(per[k])))
