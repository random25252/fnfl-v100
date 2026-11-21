import re, sys, os, numpy as np
log, d = sys.argv[1], sys.argv[2]
txt = open(log).read().split('common_debug_cb_eval:')
ref = []
for blk in txt:
    m = re.match(r'\s*hc_mixed-(\d+) = .*?= \{2560, (\d+), 1, 1\}', blk)
    if not m: continue
    rows = re.findall(r'\[\s*([-\d.e+ ,]+?)\s*,\s*\.\.\.\s*,\s*([-\d.e+ ,]+?)\s*\]', blk)
    s = float(re.search(r'sum = ([-\d.e+naninf]+)', blk).group(1))
    ref.append((int(m.group(1)), [(list(map(float, a.split(','))), list(map(float, b.split(',')))) for a, b in rows], s))
seen = {}
print(f"{'site':14s} {'ref sum':>12s} {'fnfl sum':>12s}  first values row0 ref | fnfl")
for L, rows, s in ref:
    sub = seen.get(L, 0); seen[L] = sub + 1
    name = 'attn' if sub == 0 else 'ffn'
    p = os.path.join(d, f'c0-L{L}-{name}-h0-mixed.f32')
    if not os.path.exists(p): print(L, name, 'no dump'); continue
    a = np.fromfile(p, np.float32).reshape(-1, 2560)
    fs = float(a.sum()); f3 = a[0, :3]; l3 = a[0, -3:]
    dev = max(max(abs(x - y) for x, y in zip(rows[b][0], a[b, :3])) for b in range(len(rows)))
    flag = '  <-- DIVERGES' if dev > 0.05 * max(1, max(abs(v) for v in rows[0][0])) else ''
    print(f"L{L:2d} {name:4s}      {s:12.4f} {fs:12.4f}  {np.round(rows[0][0], 4)} | {np.round(f3, 4)}  maxdev3={dev:.4f}{flag}")
