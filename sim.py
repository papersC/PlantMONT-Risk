"""PlantMONT-Risk slot-level simulator.

Discrete-epoch, slot-level simulation of a hierarchical plant nanosensor network
(leaf nanosensors -> stem macrosinks) with a parametric THz link abstraction,
moisture/blockage dynamics, intermittent harvesting, correlated VOC stress bursts,
and seven scheduling policies. All energies are in normalised energy units (EU).
"""
import numpy as np
import time
from math import erf, sqrt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import ndtr

# ----------------------------------------------------------------------------- params
DEFAULT = dict(
    N=120, M=3, S=30, F=3, T=500, warm=30,
    dmax=2.0, dI=2.5,                         # cm
    M0=12.0, g=(0.0, 3.0, 5.0), kappa=(10.0, 5.0, 2.5), sig_fast=2.0,
    sig_X=2.0, Lb=5.0, p01=0.02, p10=0.2,
    mbar=0.5, m_amp=0.10, m_period=200, rain_rate=1 / 150, rain_jump=(0.2, 0.4), rain_tau=30.0,
    est_noise=0.03, scen_sigma=0.06,
    Bmax=60.0, e_tx=1.0, e_rx=0.5, e_base=0.04, e_req=0.1, reserve=3.0,
    h_on=1.0, h_mult=0.5, p_onoff=0.05, p_offon=0.05,
    lam0=0.12, burst_m_gain=3.0, r_burst=1.5, p_det=0.9, p_onset=0.6,
    w_range=(0.6, 1.0), sig_w=0.10, p_fa=2e-4,
    D_c=3.0, R_u=20, D_u=20.0, w_u=0.05,
    R_tdma=3, q_ra=0.25, ra_max_att=8, p_req=0.95,
    n_scen=48, lam=1.0, alpha=0.9, mu=0.005, rho=2.0, eps=0.1,
    n_paths=3, tau_set=(None, 0.9, 0.97, 0.995), r_max=5, s0_step=3, max_routine=8, mu_cap=0.02, beta_r=0.5,
    stress=None, foliage='uniform', relay_z0=3.0, relay_dz=1.6, petiole_x=(0.9, 1.8),
)

POLICIES = ['TDMA', 'PAFS', 'ERA', 'EAMAC', 'ETEV', 'RISK', 'RISK_NOM']


# ----------------------------------------------------------------------------- topology
class Topology:
    def __init__(self, P, rng):
        N, M = P['N'], P['M']
        zs = {1: [15.0], 2: [9.0, 21.0], 3: [7.0, 15.0, 23.0], 4: [5.0, 11.0, 17.0, 23.0]}[M]
        leaf_z = [7.0, 15.0, 23.0]
        leaves = []
        for z in leaf_z:
            for sgn in (1, -1):
                leaves.append((sgn * 3.8, 0.0, z + 0.6))
        L = len(leaves)
        if P['foliage'] == 'uniform':
            wts = np.ones(L) / L
        else:
            wts = rng.dirichlet(np.ones(L) * 1.5)
        counts = rng.multinomial(N, wts)
        pos = [np.array([0.0, 0.0, z]) for z in zs]
        leaf_of = [-1] * M
        for z in np.arange(P['relay_z0'], 27.01, P['relay_dz']):  # stem nanorouters
            pos.append(np.array([0.25, 0.0, z]))
            leaf_of.append(-2)
        for (cx, cy, cz) in leaves:  # two petiole nanorouters per leaf
            sg = np.sign(cx)
            for x in P['petiole_x']:
                pos.append(np.array([sg * x, 0.0, cz - 0.3]))
                leaf_of.append(-2)
        self.n_relay = len(pos) - M
        for li, (cx, cy, cz) in enumerate(leaves):
            for _ in range(counts[li]):
                while True:
                    u, v = rng.uniform(-1, 1, 2)
                    if u * u + v * v <= 1:
                        break
                pos.append(np.array([cx + 3.0 * u, cy + 1.25 * v, cz + 0.15 * rng.standard_normal()]))
                leaf_of.append(li)
        self.pos = np.array(pos)
        self.leaf_of = np.array(leaf_of)
        self.M, self.N, self.n = M, N, len(pos)
        self.sensors = np.where(self.leaf_of >= 0)[0]
        self.leaves = leaves
        D = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        self.D = D
        ii, jj = np.where(np.triu(D <= P['dmax'], 1))
        keep = ~((ii < M) & (jj < M))
        ii, jj = ii[keep], jj[keep]
        self.la, self.lb = ii, jj
        self.ld = np.maximum(D[ii, jj], 0.2)
        la_, lb_ = self.leaf_of[ii], self.leaf_of[jj]
        stemlike = (la_ < 0) & (lb_ < 0)
        self.lcls = ((la_ != lb_) & ~stemlike).astype(float)  # blockage-susceptible
        self.nl = len(ii)
        self.lid = -np.ones((self.n, self.n), dtype=int)
        self.lid[ii, jj] = np.arange(self.nl)
        self.lid[jj, ii] = np.arange(self.nl)
        self.nbrI = (D <= P['dI'])  # interference neighbourhood (includes self)
        self.is_sink = np.zeros(self.n, bool)
        self.is_sink[:M] = True


def link_p(P, topo, m, blk, X):
    """Success probability per link and band. m scalar or (S,), X (nl,) or (S,nl)."""
    g = np.array(P['g'])
    kap = np.array(P['kappa'])
    m = np.asarray(m, float)
    base = P['M0'] - 20 * np.log10(topo.ld)  # (nl,)
    if m.ndim == 0:
        marg = base[:, None] - g[None, :] - kap[None, :] * m * topo.ld[:, None] \
            - P['Lb'] * blk * topo.lcls[:, None] + np.asarray(X)[..., None]
    else:  # scenarios: m (S,), blk (S,), X (S,nl) -> (S,nl,F)
        marg = base[None, :, None] - g[None, None, :] \
            - kap[None, None, :] * m[:, None, None] * topo.ld[None, :, None] \
            - P['Lb'] * blk[:, None, None] * topo.lcls[None, :, None] + X[..., None]
    return np.minimum(ndtr(marg / P['sig_fast']), 0.995)


# ----------------------------------------------------------------------------- environment
class Env:
    def __init__(self, P, topo, rng):
        self.P, self.topo, self.rng = P, topo, rng
        self.rain = 0.0
        self.eta = 0.0
        self.blk = 0
        self.m_hist = [P['mbar'], P['mbar']]
        n = topo.n
        self.hon = rng.random(n) < P['p_offon'] / (P['p_onoff'] + P['p_offon'])
        self.t = 0

    def step(self, t):
        P, rng = self.P, self.rng
        self.t = t
        self.rain *= np.exp(-1.0 / P['rain_tau'])
        st = P['stress']
        if rng.random() < P['rain_rate']:
            self.rain += rng.uniform(*P['rain_jump'])
        if st == 'burst_rain' and t % 100 == 50:
            self.rain += 0.35
        self.eta = 0.9 * self.eta + 0.02 * rng.standard_normal()
        m = P['mbar'] + P['m_amp'] * np.sin(2 * np.pi * t / P['m_period']) + self.rain + self.eta
        self.m = float(np.clip(m, 0.0, 1.0))
        self.m_hist.append(self.m)
        if self.blk == 0:
            self.blk = int(rng.random() < P['p01'])
        else:
            self.blk = int(rng.random() >= P['p10'])
        self.X = P['sig_X'] * rng.standard_normal(self.topo.nl)
        self.p_true = link_p(P, self.topo, self.m, self.blk, self.X)  # (nl,F)
        # harvesting
        sw = rng.random(self.topo.n)
        self.hon = np.where(self.hon, sw >= P['p_onoff'], sw < P['p_offon'])
        H = self.hon * P['h_on'] * P['h_mult']
        H[:self.topo.M] = 0.0
        if st == 'zero_harvest':
            H[self.zero_set] = 0.0 if 100 <= t < 400 else H[self.zero_set]
        self.H = H

    def gateway_estimate(self, blind=False):
        P = self.P
        if blind:
            return P['mbar'], 0
        mh = self.m_hist[-2] + P['est_noise'] * self.grng.standard_normal()
        return float(np.clip(mh, 0, 1)), self.blk_prev

    def scenarios(self, mhat, blk_prev, blind=False):
        P, rng = self.P, self.grng
        S = P['n_scen']
        if blind:
            ms = np.full(S, P['mbar'])
            bs = np.zeros(S)
        else:
            ms = mhat + P['scen_sigma'] * rng.standard_normal(S)
            jump = rng.random(S) < P['rain_rate'] * 2
            ms = ms + jump * rng.uniform(*P['rain_jump'], S)
            ms = np.clip(ms, 0, 1)
            if blk_prev == 0:
                bs = (rng.random(S) < P['p01']).astype(float)
            else:
                bs = (rng.random(S) >= P['p10']).astype(float)
        X = P['sig_X'] * rng.standard_normal((S, self.topo.nl))
        return link_p(P, self.topo, ms, bs, X)  # (S,nl,F)


# ----------------------------------------------------------------------------- alerts
class Alert:
    __slots__ = ('id', 'src', 'holder', 't_det', 'deadline', 'w', 'what', 'e', 'kind', 'burst',
                 'req', 'done', 't_del', 'att', 'path', 'grp')

    def __init__(self, aid, src, t_det, deadline, w, what, e, kind, burst):
        self.id, self.src, self.holder = aid, src, src
        self.t_det, self.deadline = t_det, deadline
        self.w, self.what, self.e, self.kind, self.burst = w, what, e, kind, burst
        self.req = False
        self.done = False
        self.t_del = None
        self.att = 0
        self.path = None
        self.grp = None


# ----------------------------------------------------------------------------- occupancy
class Occ:
    def __init__(self, P, topo):
        S, F, n = P['S'], P['F'], topo.n
        self.occ = np.zeros((S, n), np.int16)
        self.rxb = np.zeros((S, F, n), np.int16)
        self.txb = np.zeros((S, F, n), np.int16)
        self.nbr = topo.nbrI.astype(np.int16)

    def feasible(self, ui, uj, uf, us, valid):
        ok = (self.occ[us, ui] == 0) & (self.occ[us, uj] == 0) & \
             (self.rxb[us, uf, uj] == 0) & (self.txb[us, uf, ui] == 0)
        return np.all(ok | ~valid, axis=-1)

    def add(self, ui, uj, uf, us, sign=1):
        for i, j, f, s in zip(ui, uj, uf, us):
            self.occ[s, i] += sign
            self.occ[s, j] += sign
            self.rxb[s, f] += sign * self.nbr[i]
            self.txb[s, f] += sign * self.nbr[j]


# ----------------------------------------------------------------------------- simulator
class Sim:
    def __init__(self, P, policy, seed):
        self.P = dict(DEFAULT)
        self.P.update(P)
        P = self.P
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        topo_rng = np.random.default_rng(seed + 7919)
        self.topo = Topology(P, topo_rng)
        self.env = Env(P, self.topo, np.random.default_rng(seed + 104729))
        self.erng = np.random.default_rng(seed + 15485863)  # event process (common across policies)
        self.env.blk_prev = 0
        self.env.grng = np.random.default_rng(seed + 31337)
        n = self.topo.n
        self.B = topo_rng.uniform(0.5, 1.0, n) * P['Bmax']
        self.B[:self.topo.M] = 1e9
        if P['stress'] == 'zero_harvest':
            sens = self.topo.sensors
            self.env.zero_set = topo_rng.choice(sens, len(sens) // 4, replace=False)
        self.alerts = []
        self.aid = 0
        self.bursts = []
        self.phase = topo_rng.integers(0, 10 ** 6, n)
        self.flags = {}  # TDMA pending detection flags per sensor
        self.dead_sink_after = None
        # fixed tree for baselines (band 1, nominal moisture)
        p_nom = link_p(P, self.topo, P['mbar'], 0, np.zeros(self.topo.nl))
        self.fixed_band = 1
        self.tree_fixed = self._tree(-np.log(p_nom[:, self.fixed_band]) + 0.02)
        self.tree_minhop = self._tree(np.ones(self.topo.nl))
        self.dead_sink_after = None
        # metrics
        self.loss_epoch = np.zeros(P['T'])
        self.e_comm = 0.0
        self.e_harv = 0.0
        self.e_spent = 0.0
        self.depl = 0
        self.node_epochs = 0
        self.occupied = 0
        self.slotbands = 0
        self.rt = []
        self.slack_n = 0
        self.crit_sched_n = 0
        self.req_lost = 0
        self.gopen, self.gcount, self.gdone = {}, 0, set()

    # ---------------------------------------------------------------- routing helpers
    def _tree(self, w, excl=None):
        topo = self.topo
        w = np.array(w, float)
        if excl is not None and excl.any():
            bad = excl[topo.la] | excl[topo.lb]
            w = w.copy()
            w[bad] = 1e6
        G = csr_matrix((np.r_[w, w], (np.r_[topo.la, topo.lb], np.r_[topo.lb, topo.la])),
                       shape=(topo.n, topo.n))
        sinks = np.arange(topo.M)
        if self.dead_sink_after is not None:
            sinks = sinks[sinks != self.dead_sink]
        dist, pred, src = dijkstra(G, directed=False, indices=sinks, min_only=True,
                                   return_predecessors=True)
        return pred

    @staticmethod
    def _path(pred, v, maxlen=12):
        path = [v]
        while pred[path[-1]] >= 0 and len(path) <= maxlen:
            path.append(pred[path[-1]])
        return path

    def _sink_ok(self, v):
        if v >= self.topo.M:
            return False
        return not (self.dead_sink_after is not None and v == self.dead_sink)

    # ---------------------------------------------------------------- event generation
    def gen_events(self, t, alive):
        P, rng, topo = self.P, self.erng, self.topo
        M = topo.M
        m = self.env.m
        lam = P['lam0'] * (1 + P['burst_m_gain'] * max(0.0, m - P['mbar']))
        if P['stress'] == 'burst_rain' and 50 <= t % 100 < 56:
            lam = max(lam, 0.6)
        new = []
        n_b = rng.poisson(lam)
        for _ in range(n_b):
            li = rng.integers(len(topo.leaves))
            sens = np.where(topo.leaf_of == li)[0]
            if len(sens) == 0:
                continue
            epi = topo.pos[rng.choice(sens)]
            aff = sens[np.linalg.norm(topo.pos[sens] - epi, axis=1) <= P['r_burst']]
            w = rng.uniform(*P['w_range'])
            bid = len(self.bursts)
            self.bursts.append(dict(t=t, n_alert=0, ontime=0, w=w, leaf=int(li), epi=epi.copy()))
            for s in aff:
                delay = rng.geometric(P['p_onset']) - 1
                if delay > 2 or rng.random() > P['p_det']:
                    rng.random(); rng.random()
                    continue
                frac = rng.random()
                wn = float(np.clip(w + 0.05 * rng.standard_normal(), 0.05, 1))
                wh = float(np.clip(wn + P['sig_w'] * rng.standard_normal(), 0.05, 1))
                new.append((t + delay, s, frac, wn, wh, 1, 'crit', bid))
        # false alarms
        nfa = rng.binomial(topo.N, P['p_fa'])
        for s in rng.choice(topo.sensors, nfa, replace=False) if nfa else []:
            wh = float(rng.uniform(0.5, 0.9))
            new.append((t, s, rng.random(), 0.0, wh, 0, 'fa', -1))
        self.future = getattr(self, 'future', []) + new
        out = []
        keep = []
        for ev in self.future:
            if ev[0] == t:
                tt, s, frac, wn, wh, e, kind, bid = ev
                if not alive[s]:
                    continue  # dead node cannot sense
                dl = P['D_c']
                a = Alert(self.aid, s, t + frac, t + frac + dl, wn, wh, e, kind, bid)
                self.aid += 1
                out.append(a)
                if bid >= 0:
                    self.bursts[bid]['n_alert'] += 1
            elif ev[0] > t:
                keep.append(ev)
        self.future = keep
        # routine updates
        due = np.where(((self.phase + t) % P['R_u'] == 0) & alive)[0]
        for s in due:
            if topo.leaf_of[s] < 0:
                continue
            a = Alert(self.aid, s, t + 0.0, t + P['D_u'], P['w_u'], P['w_u'], 1, 'routine', -1)
            self.aid += 1
            out.append(a)
        return out

    # ---------------------------------------------------------------- energy helpers
    def spend(self, v, e):
        self.B[v] -= e
        self.e_comm += e

    # ---------------------------------------------------------------- execution of a scheduled route
    def execute(self, a, hops, t):
        """hops: list of (i, j, f, [slots]). Packet moves hop by hop with ARQ inside allotted slots."""
        P, env = self.P, self.env
        for (i, j, f, slots) in hops:
            l = self.topo.lid[i, j]
            ok = False
            for s in slots:
                self.spend(i, P['e_tx'])
                self.spend(j, P['e_rx'])
                self.occupied += 1
                if self.rng.random() < env.p_true[l, f]:
                    ok = True
                    a.holder = j
                    last = s
                    break
            if not ok:
                return False
        if self._sink_ok(a.holder):
            a.done = True
            a.t_del = t + (last + 1) / P['S']
            return True
        return False

    # ---------------------------------------------------------------- policies
    def step_policy(self, t, pend, alive):
        pol = self.policy
        if pol in ('ETEV', 'RISK', 'RISK_NOM'):
            self.pol_risk(t, pend, alive)
        elif pol == 'EAMAC':
            self.pol_eamac(t, pend, alive)
        elif pol in ('TDMA', 'PAFS'):
            self.pol_periodic(t, pend, alive)
        elif pol == 'ERA':
            self.pol_era(t, pend, alive)

    # .......... periodic TDMA / plant-aware frequency selection
    def pol_periodic(self, t, pend, alive):
        P, topo = self.P, self.topo
        occ = Occ(P, topo)
        if self.policy == 'PAFS':
            mhat, bprev = self.env.gateway_estimate()
            p_hat = link_p(P, topo, mhat, bprev, np.zeros(topo.nl))
            band_of = np.argmax(p_hat, axis=1)
            excl = ~alive | (self.B < P['e_tx'] + P['e_rx'])
            excl[:topo.M] = False
            pred = self._tree(-np.log(p_hat.max(1)) + 0.02, excl)
        else:
            band_of = np.full(topo.nl, self.fixed_band)
            pred = self.tree_fixed
        # report packets: one per due sensor, carrying all its pending detections
        due = set(np.where(((self.phase + t) % P['R_tdma'] == 0) & alive & (topo.leaf_of >= 0))[0])
        pkts = getattr(self, 'pkts', [])
        bysrc = {}
        for a in pend:
            bysrc.setdefault(a.src, []).append(a)
        for s in due:
            members = [a for a in bysrc.get(s, []) if a.holder == a.src and not a.req]
            for a in members:
                a.req = True  # bound to this report
            pkts.append(dict(holder=s, members=members, born=t))
        newpk = []
        for pk in pkts:
            pk['members'] = [a for a in pk['members'] if not a.done and a.deadline > t]
            if t - pk['born'] > P['R_u']:
                for a in pk['members']:
                    a.req = False
                continue
            v = pk['holder']
            path = self._path(pred, v)
            s_next = 0
            moved = True
            while moved and not self._sink_ok(v):
                moved = False
                if len(path) < 2:
                    break
                j = path[1]
                l = topo.lid[v, j]
                if l < 0:
                    break
                f = band_of[l]
                if self.B[v] < P['e_tx'] or self.B[j] < P['e_rx'] or not alive[j] or not alive[v]:
                    break
                for s in range(s_next, P['S']):
                    if occ.feasible(np.array([v]), np.array([j]), np.array([f]), np.array([s]),
                                    np.array([True])):
                        occ.add([v], [j], [f], [s])
                        self.spend(v, P['e_tx']); self.spend(j, P['e_rx'])
                        self.occupied += 1
                        if self.rng.random() < self.env.p_true[l, f]:
                            v = j
                            path = path[1:]
                            s_next = s + 1
                            moved = True
                        else:
                            v = None
                        break
                if v is None:
                    break
            if v is None:  # lost
                for a in pk['members']:
                    a.req = False
                continue
            if self._sink_ok(v):
                for a in pk['members']:
                    a.done = True
                    a.t_del = t + s_next / P['S']
                    a.holder = v
                continue
            pk['holder'] = v
            newpk.append(pk)
        self.pkts = newpk

    # .......... event-only random access
    def pol_era(self, t, pend, alive):
        P, topo, rng = self.P, self.topo, self.rng
        pred = self.tree_fixed
        f = self.fixed_band
        act = [a for a in pend if a.kind != 'routine' and not a.done]
        for a in pend:
            if a.kind == 'routine':
                a.deadline = -1  # event-only: routine updates are not sent
        start = {}
        for a in act:
            start[a.id] = int(np.ceil((a.t_det - t) * P['S'])) if a.t_det > t else 0
        for s in range(P['S']):
            tx = []
            for a in act:
                if a.done or a.att > P['ra_max_att'] or s < start[a.id]:
                    continue
                v = a.holder
                if not alive[v] or self.B[v] < P['e_tx']:
                    continue
                if rng.random() < P['q_ra']:
                    path = self._path(pred, v)
                    if len(path) < 2:
                        continue
                    tx.append((a, v, path[1]))
            if not tx:
                continue
            txn = np.array([v for _, v, _ in tx])
            for a, v, j in tx:
                self.spend(v, P['e_tx'])
                self.occupied += 1
                a.att += 1
                if not alive[j] or (j >= topo.M and self.B[j] < P['e_rx']):
                    continue
                self.spend(j, P['e_rx'])
                others = txn[txn != v]
                if j in txn:
                    continue  # half duplex
                if others.size and topo.nbrI[others, j].any():
                    continue  # collision
                l = topo.lid[v, j]
                if rng.random() < self.env.p_true[l, f]:
                    a.holder = j
                    a.att = 0
                    if self._sink_ok(j):
                        a.done = True
                        a.t_del = t + (s + 1) / P['S']

    # .......... energy-aware event-triggered MAC (FIFO, min-hop, single attempt)
    def pol_eamac(self, t, pend, alive):
        P, topo = self.P, self.topo
        occ = Occ(P, topo)
        f = self.fixed_band
        excl = ~alive | (self.B < P['reserve'] + P['e_tx'] + P['e_rx'])
        excl[:topo.M] = False
        pred = self._tree(np.ones(topo.nl), excl)
        used = np.zeros(topo.n)
        avail = self.B - P['reserve']
        q = sorted([a for a in pend if a.req and not a.done], key=lambda a: (a.kind == 'routine', a.t_det))
        for a in q:
            path = self._path(pred, a.holder)
            if len(path) < 2 or not self._sink_ok(path[-1]):
                continue
            if a.kind == 'routine' and np.any(self.B[path[:-1]] < P['beta_r'] * P['Bmax']):
                continue
            hops = []
            s = 0
            okall = True
            for i, j in zip(path[:-1], path[1:]):
                if used[i] + P['e_tx'] > avail[i] or (j >= topo.M and used[j] + P['e_rx'] > avail[j]):
                    okall = False
                    break
                while s < P['S'] and not occ.feasible(np.array([i]), np.array([j]), np.array([f]),
                                                      np.array([s]), np.array([True])):
                    s += 1
                if s >= P['S']:
                    okall = False
                    break
                hops.append((i, j, f, [s]))
                s += 1
            if not okall or not hops:
                continue
            if t + (hops[-1][3][0] + 1) / P['S'] > a.deadline:
                continue
            for i, j, ff, ss in hops:
                occ.add([i], [j], [ff], ss)
                used[i] += P['e_tx']
                used[j] += P['e_rx']
            self.execute(a, hops, t)

    # .......... proposed risk-aware scheduler and its ablations
    def build_K(self, a, preds, band_of, t, pS, price, p_hat):
        P, topo = self.P, self.topo
        S = P['S']
        routine = a.kind == 'routine'
        taus = (None,) if routine else P['tau_set']
        step = 2 * P['s0_step'] if routine else P['s0_step']
        blocks = []
        seen = set()
        for pred in (preds[:1] if routine else preds):
            path = self._path(pred, a.holder)
            tp = tuple(path)
            if len(path) < 2 or not self._sink_ok(path[-1]) or tp in seen:
                continue
            seen.add(tp)
            pa = np.array(path)
            if routine and np.any(self.B[pa[:-1]] < P['beta_r'] * P['Bmax']):
                continue
            links = topo.lid[pa[:-1], pa[1:]]
            if links.min() < 0:
                continue
            bands = band_of[links]
            H = len(links)
            ph = pS[:, links, bands]  # (nS,H)
            phat = p_hat[links, bands]
            rvecs = []
            for tau in taus:
                if tau is None:
                    rv = np.ones(H, int)
                else:
                    rv = np.clip(np.ceil(np.log(1 - tau) / np.log(np.maximum(1 - phat, 1e-6))), 1, P['r_max']).astype(int)
                if not any(np.array_equal(rv, x) for x in rvecs):
                    rvecs.append(rv)
            for rv in rvecs:
                need = int(rv.sum())
                s0 = np.arange(0, S - need + 1, step)
                s0 = s0[t + (s0 + need) / S <= a.deadline + 1e-9]
                if s0.size == 0:
                    continue
                ui = np.repeat(pa[:-1], rv); uj = np.repeat(pa[1:], rv); uf = np.repeat(bands, rv)
                offs = np.arange(need)
                uvec = np.prod(1 - (1 - ph) ** rv[None, :], axis=1)  # (nS,)
                E = np.zeros(topo.n)
                np.add.at(E, pa[:-1], rv * P['e_tx'])
                np.add.at(E, pa[1:], rv * P['e_rx'])
                E[:topo.M] = 0
                blocks.append((path, links, bands, rv, s0, ui, uj, uf, offs, uvec, E))
        if not blocks:
            return None
        C = sum(len(bl[4]) for bl in blocks)
        U = max(len(bl[5]) for bl in blocks)
        nS = pS.shape[0]
        Ui = np.zeros((C, U), int); Uj = np.zeros((C, U), int); Uf = np.zeros((C, U), int)
        Us = np.zeros((C, U), int); val = np.zeros((C, U), bool)
        u = np.zeros((C, nS)); Em = np.zeros((C, topo.n)); cands = []
        c = 0
        for (path, links, bands, r, s0, ui, uj, uf, offs, uvec, E) in blocks:
            n_ = len(s0); L_ = len(ui)
            sl = slice(c, c + n_)
            Ui[sl, :L_] = ui; Uj[sl, :L_] = uj; Uf[sl, :L_] = uf
            Us[sl, :L_] = s0[:, None] + offs[None, :]
            val[sl, :L_] = True
            u[sl] = uvec[None, :]
            Em[sl] = E[None, :]
            for x in s0:
                cands.append((path, links, bands, r, int(x)))
            c += n_
        return dict(a=a, cands=cands, ui=Ui, uj=Uj, uf=Uf, us=Us, val=val, u=u, E=Em,
                    Etot=Em @ price, crit=a.what >= 0.5 and not routine, cur=-1)

    def pol_risk(self, t, pend, alive):
        P, topo = self.P, self.topo
        t0 = time.perf_counter()
        blind = self.policy == 'RISK_NOM'
        lam = 0.0 if self.policy == 'ETEV' else P['lam']
        rho = 0.0 if self.policy == 'ETEV' else P['rho']
        alpha = P['alpha']
        mhat, bprev = self.env.gateway_estimate(blind)
        p_hat = link_p(P, topo, mhat, bprev, np.zeros(topo.nl))
        band_of = np.argmax(p_hat, axis=1)
        excl = ~alive | (self.B < P['reserve'] + P['e_tx'] + P['e_rx'])
        excl[:topo.M] = False
        w = 1.0 / np.maximum(p_hat.max(1), 1e-3)  # ETX
        preds = []
        vv = np.arange(topo.M, topo.n)
        for _ in range(P['n_paths']):
            pr = self._tree(w, excl)
            preds.append(pr)
            uu = pr[vv]
            mk = uu >= 0
            on = np.zeros(topo.nl, bool)
            on[topo.lid[vv[mk], uu[mk]]] = True
            w = np.where(on, w * 2.5 + 0.3, w)
        q = [a for a in pend if a.req and not a.done]
        for a in q:
            if a.grp is None:
                if a.kind == 'routine':
                    a.grp = ('r', a.id)
                else:
                    lf = topo.leaf_of[a.src]
                    cand = [g for g in self.gopen.get(lf, [])
                            if a.t_det - g[1] <= P['D_c'] and topo.D[a.src, g[2]] <= 2 * P['r_burst']]
                    if cand:
                        g = cand[0]
                    else:
                        g = (('e', self.gcount), a.t_det, a.src)
                        self.gcount += 1
                        self.gopen[lf] = [x for x in self.gopen.get(lf, []) if a.t_det - x[1] <= P['D_c']] + [g]
                    a.grp = g[0]
        q = [a for a in q if a.grp not in self.gdone]
        qc = [a for a in q if a.kind != 'routine']
        qr = sorted([a for a in q if a.kind == 'routine'], key=lambda a: a.t_det)[:P['max_routine']]
        q = qc + qr
        if not q:
            self.rt.append(time.perf_counter() - t0)
            return
        pS = self.env.scenarios(mhat, bprev, blind)  # (S,nl,F)
        nS = pS.shape[0]
        avail = self.B - P['reserve']
        price = np.minimum(P['mu'] * P['Bmax'] / np.maximum(self.B, 1.0), P['mu_cap'])
        price[:topo.M] = 0.0
        Ks = []
        for a in q:
            K = self.build_K(a, preds, band_of, t, pS, price, p_hat)
            if K is not None:
                K['g'] = a.grp
                Ks.append(K)
        if not Ks:
            self.rt.append(time.perf_counter() - t0)
            return
        sol = greedy_solve(Ks, P, topo, avail, nS, lam, rho, alpha, 1.0)
        self.rt.append(time.perf_counter() - t0)
        self.last_problem = (Ks, avail, nS, lam, rho, alpha, 1.0)
        seen_g = set()
        for K in Ks:
            if K['crit'] and K['g'] not in seen_g:
                seen_g.add(K['g'])
                self.crit_sched_n += 1
                if K['gmiss'] > P['eps'] + 1e-9:
                    self.slack_n += 1
            if K['cur'] < 0:
                continue
            path, links, bands, rv, s0 = K['cands'][K['cur']]
            hops = []
            k = 0
            for h, (l, f) in enumerate(zip(links, bands)):
                hops.append((path[h], path[h + 1], f, [s0 + k + x for x in range(rv[h])]))
                k += rv[h]
            if self.execute(K['a'], hops, t):
                self.gdone.add(K['g'])

    # ---------------------------------------------------------------- main loop
    def run(self):
        P, topo, env = self.P, self.topo, self.env
        T = P['T']
        all_alerts = []
        for t in range(T):
            env.blk_prev = env.blk
            env.step(t)
            if P['stress'] == 'sink_loss' and t == 120:
                self.dead_sink_after = t
                self.dead_sink = 1 if topo.M > 1 else 0
                p_nom = link_p(P, topo, P['mbar'], 0, np.zeros(topo.nl))
                self.tree_fixed = self._tree(-np.log(p_nom[:, self.fixed_band]) + 0.02)
            alive = self.B >= P['e_base']
            alive[:topo.M] = True
            if self.dead_sink_after is not None:
                alive[self.dead_sink] = False
            # base cost
            sens = np.arange(topo.M, topo.n)
            self.B[sens] -= np.where(alive[sens], P['e_base'], 0)
            self.e_spent += np.where(alive[sens], P['e_base'], 0).sum()
            new = self.gen_events(t, alive)
            all_alerts += new
            self.alerts += new
            pend = [a for a in self.alerts if not a.done and a.deadline > t]
            # request phase for event-triggered centralised policies
            if self.policy in ('EAMAC', 'ETEV', 'RISK', 'RISK_NOM'):
                for a in pend:
                    if not a.req and a.t_det < t:  # detected in an earlier epoch
                        if self.B[a.src] >= P['e_req']:
                            self.spend(a.src, P['e_req'])
                            if self.rng.random() < P['p_req']:
                                a.req = True
                            else:
                                self.req_lost += 1
            self.step_policy(t, pend, alive)
            # harvest
            sensH = env.H[sens]
            before = self.B[sens].copy()
            self.B[sens] = np.minimum(P['Bmax'], np.maximum(self.B[sens], 0) + sensH)
            self.e_harv += (self.B[sens] - np.maximum(before, 0)).sum()
            self.slotbands += P['S'] * P['F']
            if t >= P['warm']:
                self.depl += int((self.B[sens] < P['e_base']).sum())
                self.node_epochs += len(sens)
            # expire
            keep = []
            for a in self.alerts:
                if a.done:
                    continue
                if a.deadline <= t + 1 or a.deadline < 0:
                    if a.e and t < T:
                        self.loss_epoch[t] += a.w
                    continue
                keep.append(a)
            self.alerts = keep
        self.all_alerts = all_alerts
        return self.metrics()

    def metrics(self):
        P = self.P
        W = P['warm']
        T = P['T']
        crit = [a for a in self.all_alerts if a.kind == 'crit']
        first_det, first_del = {}, {}
        for a in crit:
            first_det[a.burst] = min(first_det.get(a.burst, 1e9), a.t_det)
            if a.done and a.t_del <= a.deadline:
                first_del[a.burst] = min(first_del.get(a.burst, 1e9), a.t_del)
        # merge ground-truth bursts that overlap in space and time into one event
        nb = len(self.bursts)
        par = list(range(nb))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i in range(nb):
            for j in range(i + 1, nb):
                bi, bj = self.bursts[i], self.bursts[j]
                if bj['t'] - bi['t'] > P['D_c']:
                    break
                if bi['leaf'] == bj['leaf'] and np.linalg.norm(bi['epi'] - bj['epi']) <= 2 * P['r_burst']:
                    par[find(j)] = find(i)
        ev = {}
        for i in range(nb):
            ev.setdefault(find(i), []).append(i)
        b_ids = [r for r in ev if W <= self.bursts[r]['t'] < T - 4]
        okl, dl_, lossv = [], [], np.zeros(T)
        for r in b_ids:
            mem = ev[r]
            fd = [first_del[i] for i in mem if i in first_del]
            okl.append(1.0 if fd else 0.0)
            if fd:
                dl_.append(min(fd) - min(first_det.get(i, 1e9) for i in mem))
            else:
                lossv[min(self.bursts[r]['t'] + 3, T - 1)] += max(self.bursts[i]['w'] for i in mem)
        ok = np.array(okl)
        dly = np.array(dl_)
        lv = lossv[W:]
        k = max(1, int(np.ceil(0.05 * len(lv))))
        cvar = np.sort(lv)[-k:].mean()
        crit_w = [a for a in crit if a.t_det >= W]
        rout = [a for a in self.all_alerts if a.kind == 'routine' and W <= a.t_det < T - P['D_u']]
        nsens = self.topo.n - self.topo.M
        tx_crit = sum(1 for a in crit_w if a.done)
        return dict(
            ev_ontime=ok.mean() if len(ok) else np.nan,
            n_ev=len(b_ids),
            missed_ev=1 - ok.mean() if len(ok) else np.nan,
            alert_ontime=np.mean([a.done and a.t_del <= a.deadline for a in crit_w]) if crit_w else np.nan,
            p95=np.percentile(dly, 95) if len(dly) else np.nan,
            p99=np.percentile(dly, 99) if len(dly) else np.nan,
            mean_delay=dly.mean() if len(dly) else np.nan,
            cvar=cvar, mean_loss=lv.mean(),
            e_comm=self.e_comm / (nsens * T),
            e_net=(self.e_harv - self.e_spent - self.e_comm) / (nsens * T),
            depl=self.depl / max(1, self.node_epochs),
            ev_per_kEU=len(dly) / max(1e-9, self.e_comm) * 1000,
            occ=self.occupied / max(1, self.slotbands),
            routine=(np.mean([a.done for a in rout]) if rout and self.policy not in ('TDMA', 'PAFS') else np.nan),
            slack=self.slack_n / max(1, self.crit_sched_n) if self.policy in ('RISK', 'RISK_NOM', 'ETEV') else np.nan,
            rt_ms=1000 * np.mean(self.rt) if self.rt else np.nan,
            rt_p95_ms=1000 * np.percentile(self.rt, 95) if self.rt else np.nan,
        )


# ----------------------------------------------------------------------------- objective & greedy
def cvar_rows(L, alpha):
    """Empirical CVaR (Rockafellar-Uryasev) of each row of L (..., S)."""
    S = L.shape[-1]
    k = (1 - alpha) * S
    Ls = -np.sort(-L, axis=-1)
    kf = int(np.floor(k))
    tail = Ls[..., :kf].sum(-1)
    if kf < S:
        tail = tail + (k - kf) * Ls[..., kf]
    return tail / k


def objective(L, lam, alpha):
    return L.mean(-1) + (lam * cvar_rows(L, alpha) if lam > 0 else 0.0)


def greedy_solve(Ks, P, topo, avail, nS, lam, rho, alpha, mu, max_iter=None):
    """Risk-weighted greedy with group (event-level) loss.

    L_sigma = sum_g w_g prod_{k in g} (1 - u_{k,sigma}); objective = mean(L) + lam*CVaR_alpha(L)
    + sum price*E + rho * sum_{critical g} max(0, mean(prod_g) - eps).
    """
    occ = Occ(P, topo)
    used = np.zeros(topo.n)
    eps = P['eps']
    gids = sorted(set(K['g'] for K in Ks), key=str)
    gix = {g: i for i, g in enumerate(gids)}
    G = len(gids)
    members = [[] for _ in range(G)]
    wg = np.zeros(G)
    cg = np.zeros(G, bool)
    for kk, K in enumerate(Ks):
        gi = gix[K['g']]
        K['gi'] = gi
        K['cur'] = -1
        members[gi].append(kk)
        wg[gi] = max(wg[gi], K['a'].what)
        cg[gi] |= K['crit']
    miss = [np.ones(nS) for _ in Ks]
    prod = np.ones((G, nS))
    L = wg @ prod

    def pen(gi, pm):  # pm: mean miss probability of group (array)
        if rho == 0 or not cg[gi]:
            return 0.0 * pm
        return rho * np.maximum(0.0, pm - eps)

    def excl(kk):
        gi = Ks[kk]['gi']
        e = np.ones(nS)
        for o in members[gi]:
            if o != kk:
                e = e * miss[o]
        return e

    def toggle(K, c, sign):
        v = K['val'][c]
        occ.add(K['ui'][c][v], K['uj'][c][v], K['uf'][c][v], K['us'][c][v], sign)

    if max_iter is None:
        max_iter = 2 * len(Ks) + 5
    for _ in range(max_iter):
        best = (0.0, None, None)
        J0 = objective(L, lam, alpha)
        for kk, K in enumerate(Ks):
            gi = K['gi']
            cur = K['cur']
            if cur >= 0:
                toggle(K, cur, -1)
                used -= K['E'][cur]
                e_cur = K['Etot'][cur]
            else:
                e_cur = 0.0
            feas = occ.feasible(K['ui'], K['uj'], K['uf'], K['us'], K['val'])
            feas &= np.all(used[None, :] + K['E'] <= avail[None, :] + 1e-9, axis=1)
            idx = np.where(feas)[0]
            if cur >= 0:
                toggle(K, cur, 1)
                used += K['E'][cur]
            if idx.size == 0:
                continue
            ex = excl(kk)
            newprod = ex[None, :] * (1 - K['u'][idx])
            Lnew = L[None, :] + wg[gi] * (newprod - prod[gi][None, :])
            dJ = objective(Lnew, lam, alpha) - J0 + mu * (K['Etot'][idx] - e_cur)
            dJ += pen(gi, newprod.mean(1)) - pen(gi, np.array([prod[gi].mean()]))[0]
            b = int(np.argmin(dJ))
            if dJ[b] < best[0] - 1e-12 and idx[b] != cur:
                best = (dJ[b], kk, idx[b])
        if best[1] is None:
            break
        _, kk, c = best
        K = Ks[kk]
        gi = K['gi']
        cur = K['cur']
        if cur >= 0:
            toggle(K, cur, -1)
            used -= K['E'][cur]
        toggle(K, c, 1)
        used += K['E'][c]
        K['cur'] = c
        miss[kk] = 1 - K['u'][c]
        newp = excl(kk) * miss[kk]
        L = L + wg[gi] * (newp - prod[gi])
        prod[gi] = newp
    J = objective(L, lam, alpha) + mu * sum(K['Etot'][K['cur']] for K in Ks if K['cur'] >= 0)
    J += sum(pen(gi, np.array([prod[gi].mean()]))[0] for gi in range(G))
    for gi in range(G):
        for kk in members[gi]:
            Ks[kk]['gmiss'] = prod[gi].mean()
    return J


def eval_assignment(Ks, assign, P, topo, avail, nS, lam, rho, alpha, mu):
    """Objective of a full assignment (list of candidate index or -1); inf if infeasible."""
    occ = Occ(P, topo)
    used = np.zeros(topo.n)
    eps = P['eps']
    gids = sorted(set(K['g'] for K in Ks), key=str)
    gix = {g: i for i, g in enumerate(gids)}
    prod = np.ones((len(gids), nS))
    wg = np.zeros(len(gids)); cg = np.zeros(len(gids), bool)
    J = 0.0
    for K, c in zip(Ks, assign):
        gi = gix[K['g']]
        wg[gi] = max(wg[gi], K['a'].what); cg[gi] |= K['crit']
        if c < 0:
            continue
        v = K['val'][c]
        if not occ.feasible(K['ui'][c:c+1], K['uj'][c:c+1], K['uf'][c:c+1], K['us'][c:c+1], K['val'][c:c+1])[0]:
            return np.inf
        occ.add(K['ui'][c][v], K['uj'][c][v], K['uf'][c][v], K['us'][c][v], 1)
        used += K['E'][c]
        if np.any(used > avail + 1e-9):
            return np.inf
        prod[gi] *= 1 - K['u'][c]
        J += mu * K['Etot'][c]
    L = wg @ prod
    J += objective(L, lam, alpha)
    if rho > 0:
        J += rho * np.sum(np.maximum(0.0, prod.mean(1) - eps) * cg)
    return float(J)


def run_one(args):
    P, pol, seed = args
    return Sim(P, pol, seed).run()
