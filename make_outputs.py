"""Regenerate every figure, table and statistic reported in the paper from results/*.json.

Writes pgfplots/table snippets to generated/ and prints the paired statistics quoted in the text.
Usage: python make_outputs.py
"""
import json, os
import numpy as np
from scipy import stats

os.makedirs('generated', exist_ok=True)
NAMES = {'TDMA': 'TDMA', 'PAFS': 'PA-FS', 'ERA': 'E-RA', 'EAMAC': 'EA-MAC', 'ETEV': 'ET-EV',
         'RISK': 'PlantMONT-Risk', 'RISK_NOM': 'RISK-NM'}
STY = {'PAFS': 'color=gray,mark=diamond*', 'ERA': 'color=blue!70!black,mark=square*',
       'EAMAC': 'color=green!50!black,mark=triangle*', 'ETEV': 'color=violet,mark=o,dashed',
       'RISK': 'color=orange!85!black,mark=*,thick', 'RISK_NOM': 'color=brown,mark=x,densely dotted',
       'TDMA': 'color=black,mark=+'}
AX = ('width=\\columnwidth,height=5.0cm,grid=major,grid style={gray!25},tick label style={font=\\scriptsize},'
      'label style={font=\\footnotesize},legend style={font=\\tiny,at={(0.5,1.03)},anchor=south,legend columns=3,draw=none}')


def load(name):
    return json.load(open(f'results/{name}.json'))


def vals(R, k):
    return np.array([r[k] for r in R if r[k] is not None], float)


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, 0.0
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def key(pol, cfg):
    return pol + '|' + json.dumps(cfg, sort_keys=True)


def sweep_plot(res, default, par, dval, metric, pols, xlabel, ylabel, legend=True, extra=''):
    out = ['\\begin{tikzpicture}\n\\begin{axis}[%s,xlabel={%s},ylabel={%s},%s]' % (AX, xlabel, ylabel, extra)]
    for pol in pols:
        pts = []
        for k, R in list(res.items()) + [(key(pol, {par: dval}), default[key(pol, {})])]:
            p, cfg = k.split('|', 1)
            cfg = json.loads(cfg)
            if p == pol and par in cfg:
                m, h = ci(vals(R, metric))
                pts.append((cfg[par], m, h))
        coords = ' '.join('(%g,%.4f) +- (0,%.4f)' % t for t in sorted(pts) if not np.isnan(t[1]))
        out.append('\\addplot+[%s,error bars/.cd,y dir=both,y explicit] coordinates {%s};' % (STY[pol], coords))
        if legend:
            out.append('\\addlegendentry{%s}' % NAMES[pol])
    out.append('\\end{axis}\n\\end{tikzpicture}')
    return '\n'.join(out)


def pareto_plot(pa):
    groups = {}
    for k, R in pa.items():
        p = k.split('|', 1)[0]
        e = ci(vals(R, 'e_comm'))[0]
        o, h = ci(vals(R, 'ev_ontime'))
        groups.setdefault(p, []).append((e, o, h))
    out = ['\\begin{tikzpicture}\n\\begin{axis}[%s,height=5.2cm,xlabel={Communication energy (EU per node and epoch)},'
           'ylabel={On-time event delivery},xmin=0.03,xmax=0.31,scaled x ticks=false,'
           'xticklabel style={/pgf/number format/fixed},ymin=0,ymax=1.02]' % AX]
    for p in ['TDMA', 'PAFS', 'ERA', 'EAMAC', 'ETEV', 'RISK']:
        pts = sorted(groups[p])
        coords = ' '.join('(%.4f,%.4f) +- (0,%.4f)' % t for t in pts)
        only = ',only marks' if len(pts) == 1 else ''
        out.append('\\addplot+[%s%s,error bars/.cd,y dir=both,y explicit] coordinates {%s};' % (STY[p], only, coords))
        out.append('\\addlegendentry{%s}' % NAMES[p])
    out.append('\\end{axis}\n\\end{tikzpicture}')
    return '\n'.join(out)


def fmt(R, k, d=3, pct=False):
    m, h = ci(vals(R, k))
    if np.isnan(m):
        return 'n/a'
    return ('%.1f$\\pm$%.1f' % (100 * m, 100 * h)) if pct else ('%.*f$\\pm$%.*f' % (d, m, d, h))


def paired(res, a, b, k, cfg):
    A = np.array([r[k] if r[k] is not None else np.nan for r in res[key(a, cfg)]])
    B = np.array([r[k] if r[k] is not None else np.nan for r in res[key(b, cfg)]])
    d = A - B
    d = d[~np.isnan(d)]
    h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), h, stats.ttest_1samp(d, 0).pvalue


def main():
    d, mo, ha, lo = load('default'), load('moist'), load('harv'), load('load')
    P4 = ['ERA', 'EAMAC', 'ETEV', 'RISK']
    xt = 'xmode=log,log basis x=2,xtick={0.06,0.12,0.24,0.48},xticklabels={0.06,0.12,0.24,0.48}'
    figs = {
        'moist': sweep_plot(mo, d, 'mbar', 0.5, 'ev_ontime', ['PAFS', 'ERA', 'EAMAC', 'ETEV', 'RISK', 'RISK_NOM'],
                            'Mean canopy moisture $\\bar m$', 'On-time event delivery', extra='ymin=0,ymax=1.05'),
        'moist_p95': sweep_plot(mo, d, 'mbar', 0.5, 'p95', ['ERA', 'EAMAC', 'ETEV', 'RISK', 'RISK_NOM'],
                                'Mean canopy moisture $\\bar m$', '95th-percentile event delay (epochs)', legend=False,
                                extra='ymin=0'),
        'harv': sweep_plot(ha, d, 'h_mult', 0.5, 'cvar', P4, 'Harvest when on (EU per epoch)',
                           'CVaR$_{0.95}$ of missed-event loss', extra='ymin=0'),
        'load_ont': sweep_plot(lo, d, 'lam0', 0.12, 'ev_ontime', P4, 'Burst rate $\\lambda_0$ (per epoch)',
                               'On-time event delivery', extra='ymin=0.4,ymax=1.05,' + xt),
        'load_p95': sweep_plot(lo, d, 'lam0', 0.12, 'p95', P4, 'Burst rate $\\lambda_0$ (per epoch)',
                               '95th-percentile delay (epochs)', legend=False, extra='ymin=0,' + xt),
        'pareto': pareto_plot(load('pareto')),
    }
    for k, v in figs.items():
        open(f'generated/fig_{k}.tex', 'w').write(v)
    rows = []
    for p in ['TDMA', 'PAFS', 'ERA', 'EAMAC', 'ETEV', 'RISK_NOM', 'RISK']:
        R = d[key(p, {})]
        rows.append('%s & %s & %s & %s & %s & %s & %s\\\\' % (NAMES[p], fmt(R, 'ev_ontime', pct=True), fmt(R, 'p95', 2),
                    fmt(R, 'cvar'), fmt(R, 'e_comm'), fmt(R, 'depl', pct=True), fmt(R, 'routine', pct=True)))
    open('generated/tab_default.tex', 'w').write('\n'.join(rows))
    st = load('stress')
    lab = {'burst_rain': '(a) Rain burst', 'zero_harvest': '(b) No harvest', 'sink_loss': '(c) Sink loss'}
    rows = []
    for c in lab:
        for i, p in enumerate(['PAFS', 'ERA', 'EAMAC', 'ETEV', 'RISK']):
            R = st[key(p, {'stress': c})]
            rows.append('%s & %s & %s & %s & %s & %s\\\\' % (lab[c] if i == 0 else '', NAMES[p], fmt(R, 'ev_ontime', pct=True),
                        fmt(R, 'p95', 2), fmt(R, 'cvar'), fmt(R, 'e_comm')))
        rows.append('\\midrule')
    open('generated/tab_stress.tex', 'w').write('\n'.join(rows[:-1]))
    # statistics quoted in the text
    print('== Paired differences (mean, 95% half-width, p-value)')
    checks = [('default', {}, 'RISK', 'ERA'), ('default', {}, 'RISK', 'ETEV'), ('default', {}, 'RISK', 'RISK_NOM'),
              ('moist', {'mbar': 0.2}, 'RISK', 'RISK_NOM'), ('moist', {'mbar': 0.8}, 'RISK', 'ERA'),
              ('harv', {'h_mult': 0.25}, 'RISK', 'ERA'), ('load', {'lam0': 0.48}, 'RISK', 'ETEV'),
              ('stress', {'stress': 'burst_rain'}, 'RISK', 'ETEV'), ('stress', {'stress': 'zero_harvest'}, 'RISK', 'ETEV'),
              ('stress', {'stress': 'sink_loss'}, 'RISK', 'ETEV')]
    for name, cfg, a, b in checks:
        r = load(name)
        txt = ' | '.join('%s %+.4f +- %.4f (p=%.2g)' % ((k,) + paired(r, a, b, k, cfg))
                         for k in ['ev_ontime', 'p95', 'cvar', 'e_comm'])
        print(name, cfg, a, '-', b, ':', txt)
    print('== Means per configuration')
    for name in ['default', 'moist', 'harv', 'load', 'pareto', 'stress', 'sens']:
        for k, R in load(name).items():
            print(name, k, ' '.join('%s=%.3f' % (m, ci(vals(R, m))[0]) for m in
                                   ['ev_ontime', 'p95', 'cvar', 'e_comm', 'depl', 'routine', 'slack', 'rt_ms']))
    g = load('gap')
    a = np.array([x['gap'] for x in g])
    print('== Gap: n=%d optimal=%.3f mean=%.4f p95=%.4f max=%.4f' % (len(a), np.mean(a < 1e-6), a.mean(),
                                                                       np.percentile(a, 95), a.max()))
    print('== Runtime scaling', load('scal'))


if __name__ == '__main__':
    main()
