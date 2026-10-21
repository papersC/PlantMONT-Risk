"""Smoke and invariant tests for the PlantMONT-Risk simulator. Run with: python -m pytest -q"""
import numpy as np
import sim
from sim import Sim, Occ, run_one, POLICIES


def test_deterministic():
    a = run_one(({'T': 80}, 'RISK', 7))
    b = run_one(({'T': 80}, 'RISK', 7))
    keys = [k for k in a if not k.startswith('rt')]  # runtimes are wall-clock
    assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in keys)


def test_common_random_numbers():
    """Every policy must face the same stress bursts for a given seed."""
    bursts = []
    for pol in ['ERA', 'ETEV']:
        s = Sim({'T': 120}, pol, 3)
        s.run()
        bursts.append([(b['t'], b['leaf'], round(b['w'], 9)) for b in s.bursts])
    assert bursts[0] == bursts[1]


def test_energy_bounds_all_policies():
    for pol in POLICIES:
        s = Sim({'T': 100}, pol, 1)
        orig = s.step_policy
        lows = []

        def wrapped(t, pend, alive, orig=orig, s=s):
            orig(t, pend, alive)
            lows.append(s.B[s.topo.M:].min())
        s.step_policy = wrapped
        m = s.run()
        assert min(lows) >= -1e-9, pol
        assert s.B[s.topo.M:].max() <= s.P['Bmax'] + 1e-9, pol
        assert 0.0 <= m['ev_ontime'] <= 1.0


def test_gateway_schedule_conflict_free():
    s = Sim({'T': 120, 'lam0': 0.3}, 'RISK', 2)
    orig = sim.greedy_solve
    checked = []

    def wrapped(Ks, P, topo, *a, **k):
        J = orig(Ks, P, topo, *a, **k)
        occ = Occ(P, topo)
        for K in Ks:
            c = K['cur']
            if c < 0:
                continue
            v = K['val'][c]
            assert occ.feasible(K['ui'][c:c + 1], K['uj'][c:c + 1], K['uf'][c:c + 1], K['us'][c:c + 1],
                                K['val'][c:c + 1])[0]
            occ.add(K['ui'][c][v], K['uj'][c][v], K['uf'][c][v], K['us'][c][v], 1)
            checked.append(1)
        return J
    sim.greedy_solve = wrapped
    try:
        s.run()
    finally:
        sim.greedy_solve = orig
    assert len(checked) > 0
