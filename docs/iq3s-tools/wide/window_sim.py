import json, sys
seqs=json.load(open(sys.argv[1]))
MTP_TOK=2.8      # tokens a step on MTP steps (general text measured ~2.8-3.4)
def lookup(ctx, W, nmax=6, nmin=3, need=4):
    L=len(ctx)
    for n in range(nmax, nmin-1, -1):
        if L<n+1: continue
        key=tuple(ctx[-n:])
        for s in range(L-n-1, -1, -1):          # most recent earlier occurrence
            if tuple(ctx[s:s+n])==key:
                fol=ctx[s+n:s+n+W]
                if len(fol)>=min(need,W): return fol
                break
    return None
def V(rows, slope_lo, slope_hi):    # verify ms, draft step excluded
    base=13.45                        # 4 rows today
    if rows<=4: return base-0.85*(4-rows)
    x=min(rows,16)-4; y=max(rows-16,0)
    return base+slope_lo*x+slope_hi*y
def run(seq, W, slope_lo, slope_hi, adapt=False):
    ctx=list(seq["prompt"]); out=seq["out"]; i=0; t=0.0; steps=0; lk=0; prev=W
    while i<len(out):
        w=W if not adapt else max(4,min(W,2*prev))
        ii=int(i); p=lookup(ctx+out[:ii], w)
        if p:
            a=0
            while a<len(p) and ii+a<len(out) and p[a]==out[ii+a]: a+=1
            adv=min(a+1,len(out)-ii); i=ii; t+=V(len(p)+1,slope_lo,slope_hi); lk+=1; prev=a if a>0 else 4
        else:
            adv=min(MTP_TOK,len(out)-i); t+=14.6
        i+=adv; steps+=1
    return len(out)/(t/1000), lk/steps, len(out)/steps
for kind in ("edit","gen"):
    S=[s for s in seqs if s["kind"]==kind]
    print(f"== {kind} ({len(S)} seqs)")
    for slopes in ((0.45,0.6),(0.9,0.9)):
        for W in (4,8,12,16,24,32,48):
            for ad in (False,True):
                r=[run(s,W,*slopes,ad) for s in S]
                tps=sum(x[0] for x in r)/len(r); lf=sum(x[1] for x in r)/len(r); tk=sum(x[2] for x in r)/len(r)
                print(f" slope {slopes} W={W:2d} {'adapt' if ad else 'fixed'}: tok/s {tps:6.1f}  lookup share {lf:4.2f}  tokens/step {tk:5.2f}")
