"""Gateway scheduler runtime versus network size (5 seeds x 200 epochs). Usage: python scaling.py"""
import os
import json, numpy as np
from sim import run_one
out={}
for N in [40,80,120,200,300]:
    R=[run_one(({'N':N,'T':200},'RISK',s)) for s in range(5)]
    out[N]=dict(rt=float(np.mean([r['rt_ms'] for r in R])), rt95=float(np.mean([r['rt_p95_ms'] for r in R])), ont=float(np.mean([r['ev_ontime'] for r in R])))
    print(N,out[N],flush=True)
os.makedirs('results', exist_ok=True)
json.dump(out,open('results/scal.json','w'))
