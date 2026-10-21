"""Run the simulation campaigns reported in the paper (30 seeds per configuration).

Usage: python run_experiments.py [default|moist|harv|load|pareto|stress|sens|all] [--workers N]
Results are cached in results/<campaign>.json; finished configurations are skipped on rerun.
"""
import sys, json, os, time, numpy as np
from multiprocessing import Pool
from sim import run_one
SEEDS = list(range(30))
def job(a):
    cfg, pol, seed = a
    r = run_one((cfg, pol, seed))
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in r.items()}
EXPS = {
 'default': [({}, p) for p in ['TDMA','PAFS','ERA','EAMAC','ETEV','RISK','RISK_NOM']],
 'moist': [({'mbar': m}, p) for m in [0.2,0.35,0.65,0.8] for p in ['PAFS','ERA','EAMAC','ETEV','RISK','RISK_NOM']],
 'harv': [({'h_mult': h}, p) for h in [0.25,1.0,1.5,2.0] for p in ['ERA','EAMAC','ETEV','RISK']],
 'load': [({'lam0': l}, p) for l in [0.06,0.24,0.48] for p in ['ERA','EAMAC','ETEV','RISK']],
 'pareto': [({'lam': l}, 'RISK') for l in [0.0,0.5,1.0,2.0,4.0]] + [({'R_tdma': R}, 'TDMA') for R in [2,3,5,10]]
           + [({'R_tdma': R}, 'PAFS') for R in [2,3,5,10]] + [({'q_ra': q}, 'ERA') for q in [0.1,0.25,0.5]] + [({}, 'ETEV'), ({}, 'EAMAC')],
 'stress': [({'stress': st}, p) for st in ['burst_rain','zero_harvest','sink_loss'] for p in ['PAFS','ERA','EAMAC','ETEV','RISK']],
 'sens': [({'sig_w': w}, p) for w in [0.0,0.2,0.3] for p in ['ETEV','RISK']] + [({'alpha': a}, 'RISK') for a in [0.8,0.95]]
         + [({'n_scen': n}, 'RISK') for n in [16,96]],
}
def run(name, workers):
    os.makedirs('results', exist_ok=True)
    out = f'results/{name}.json'
    res = json.load(open(out)) if os.path.exists(out) else {}
    with Pool(workers) as pool:
        for cfg, pol in EXPS[name]:
            key = pol + '|' + json.dumps(cfg, sort_keys=True)
            if key in res: continue
            t0 = time.time()
            R = pool.map(job, [(cfg, pol, s) for s in SEEDS])
            res[key] = R
            json.dump(res, open(out, 'w'))
            print(name, key, f'{time.time()-t0:.0f}s', round(np.mean([r['ev_ontime'] for r in R]),3), flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else os.cpu_count()
    if '--workers' in sys.argv:
        args = [a for a in args if a != str(workers)]
    names = list(EXPS) if (not args or args[0] == 'all') else args
    for n in names:
        run(n, workers)
