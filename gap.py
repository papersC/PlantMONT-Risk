"""Optimality gap of the risk-weighted greedy solver against exhaustive enumeration on small instances.

Usage: python gap.py   (writes results/gap.json)
"""
import os
os.makedirs('results', exist_ok=True)
import json, time, itertools, numpy as np
from sim import Sim, greedy_solve, eval_assignment

snaps = []
orig = Sim.pol_risk


def cap(self, t, pend, alive):
    self.last_problem = None
    orig(self, t, pend, alive)
    if self.last_problem is not None:
        Ks = self.last_problem[0]
        if 2 <= len(Ks) <= 4 and any(K['crit'] for K in Ks):
            snaps.append((self.last_problem, self.P, self.topo))


Sim.pol_risk = cap
rng = np.random.default_rng(0)
for seed in range(12):
    Sim({'N': 24, 'T': 300, 'lam0': 0.15, 'R_u': 40}, 'RISK', seed).run()
Sim.pol_risk = orig
print('snapshots', len(snaps))
idx = rng.choice(len(snaps), min(60, len(snaps)), replace=False)
CMAX = 9
out = []
for ii in idx:
    (Ks, avail, nS, lam, rho, alpha, mu), P, topo = snaps[ii]
    sub = []
    for K in Ks:
        C = len(K['cands'])
        sel = np.sort(rng.choice(C, min(CMAX, C), replace=False))
        K2 = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == (C,) else v) for k, v in K.items()}
        K2['cands'] = [K['cands'][c] for c in sel]
        sub.append(K2)
    t0 = time.perf_counter()
    Jg = greedy_solve(sub, P, topo, avail, nS, lam, rho, alpha, mu)
    tg = time.perf_counter() - t0
    t0 = time.perf_counter()
    best = np.inf
    for assign in itertools.product(*[range(-1, len(K['cands'])) for K in sub]):
        J = eval_assignment(sub, assign, P, topo, avail, nS, lam, rho, alpha, mu)
        best = min(best, J)
    te = time.perf_counter() - t0
    gap = (Jg - best) / max(abs(best), 1e-9)
    out.append(dict(nK=len(sub), Jg=Jg, Jopt=best, gap=gap, tg=tg, te=te))
    print(len(sub), round(Jg, 4), round(best, 4), round(gap, 4), round(tg * 1000, 1), round(te, 2), flush=True)
json.dump(out, open('results/gap.json', 'w'))
g = np.array([o['gap'] for o in out])
print('mean gap', g.mean(), 'median', np.median(g), 'p95', np.percentile(g, 95), 'max', g.max(),
      'zero-gap frac', np.mean(g < 1e-6))
