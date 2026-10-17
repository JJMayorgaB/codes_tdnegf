#!/usr/bin/env python3
"""
plot_kernel_elements.py -- grafica los armonicos k = 0, ±1 de las funciones de
Green del lead derecho (precesion en y) calculados por kernel_elements.jl, para
los pares de sitios (m, n) que se pidan.

Para cada par y cada componente mu = r, a, < se hace una figura 1x2 con
-(1/pi) Im de los coeficientes (como una LDOS):
    (a) k = 0 en la base propia de sigma_y:  alpha + beta (up_y),
        alpha - beta (down_y) y la traza 2 alpha (= suma de las dos)
    (b) k = +1 (gamma_+, coeficiente de tau_+) y k = -1 (gamma_-, de tau_-)

Los datos los genera kernel_elements.jl (se llama solo si el CSV no existe, si
la malla pedida no coincide o con --recompute). Todo lo que depende de los
sitios se decide aqui; la biblioteca de Julia no sabe de sitios concretos.

Uso:
    python plot_kernel_elements.py                         # par (1,1), datos de setup2
    python plot_kernel_elements.py --pairs "1,1;1,2;2,2"
    python plot_kernel_elements.py --wmin -2.5 --wmax 2.5 --nw 20001 --eta 5e-4
    python plot_kernel_elements.py --xlim -0.3 0.3 --recompute
    python plot_kernel_elements.py --jsd 0.3 --theta 20     # cambia parametros sobre el params.txt
    python plot_kernel_elements.py --params none --Omega 0.01   # parte de los valores por defecto
    (parametros: --gamma --gso --jsd --theta[grados] --Omega --mu --beta; si se cambia
     alguno, la carpeta es gam.._gso.._jsd.._th..deg_Om.._mu.._b..)

Salida (tag = nombre de la carpeta del params.txt):
    output/<tag>/data/kernel_elements_<pares>.csv
    output/<tag>/figures/kernel_elements_G{r,a,l}_m{m}n{n}.{jpg,svg,pdf}
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analytical_computations'))
import inbedding_leads_plot as P                     # noqa: E402  (rcParams comunes)
from inbedding_leads_plot import _fmt_axes            # noqa: E402

JL = os.path.join(HERE, 'kernel_elements.jl')
DEFAULT_PARAMS = os.path.join(HERE, '..', 'two_oscillators_setup', 'y_precession', 'output',
                              'setup2_gso0p1_jsd0p2_th10deg_Om0p005_buf4_adyn0p0', 'params.txt')

COMP = {'r': ('r', 'r'), 'a': ('a', 'a'), 'l': ('l', '<')}     # sufijo CSV, superindice
COLORS = {'up': 'blue', 'dn': 'red', 'tr': 'black', 'p': 'green', 'm': 'darkorange'}


def parse_pairs(s):
    out = []
    for tok in s.split(';'):
        tok = tok.strip()
        if not tok:
            continue
        m, n = (int(x) for x in tok.split(','))
        if m < 1 or n < 1:
            raise SystemExit(f'par {tok}: los sitios del lead son m, n >= 1')
        out.append((m, n))
    if not out:
        raise SystemExit('--pairs vacio')
    return out


def pairs_str(pairs):
    return ';'.join(f'{m},{n}' for m, n in pairs)


def csv_path(outdir, pairs):
    return os.path.join(outdir, 'kernel_elements_' + '_'.join(f'{m}-{n}' for m, n in pairs) + '.csv')


DEFAULTS = dict(gamma=np.sqrt(1 - 0.1**2), gso=0.1, jsd=0.2, theta=10.0, Omega=0.005, mu=0.0, beta=40.0)
PFILE_KEYS = dict(gamma='γ', gso='γso', jsd='j_sd', theta='θ_max', Omega='Ω', mu='E_F', beta='β')
CSV_KEYS = dict(gamma='gamma', gso='gammaSO', jsd='J_sd', theta='theta', Omega='Omega', mu='mu', beta='beta')


def read_params(path):
    """Mismos pares 'clave = valor' que read_params de kernel_elements.jl (θ en grados)."""
    if path == 'none':
        return dict(DEFAULTS)
    import re
    with open(path, encoding='utf-8') as fh:
        txt = fh.read()
    d = {k: float(v) for k, v in
         re.findall(r'(\S+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', txt)}
    return {k: d[v] for k, v in PFILE_KEYS.items()}


def tagnum(x):
    return ('%g' % x).replace('.', 'p').replace('-', 'm')


def param_tag(v):
    """Igual que param_tag de kernel_elements.jl."""
    return (f"gam{tagnum(v['gamma'])}_gso{tagnum(v['gso'])}_jsd{tagnum(v['jsd'])}_th{tagnum(v['theta'])}deg_"
            f"Om{tagnum(v['Omega'])}_mu{tagnum(v['mu'])}_b{tagnum(v['beta'])}")


def grid_matches(path, pairs, wmin, wmax, nw, eta, vals):
    """True si el CSV existe y tiene los pares, la malla y los parametros pedidos."""
    if not os.path.isfile(path):
        return False
    with open(path) as fh:
        head = fh.readline()
    meta = dict(kv.split('=') for kv in head.lstrip('#').split())
    if eta is not None and not np.isclose(float(meta['eta']), eta, rtol=1e-9, atol=0):
        return False
    for k, ck in CSV_KEYS.items():
        want = np.deg2rad(vals[k]) if k == 'theta' else vals[k]
        if not np.isclose(float(meta[ck]), want, rtol=1e-9, atol=1e-14):
            return False
    df = pd.read_csv(path, comment='#', usecols=['omega', 'm', 'n'])
    have = set(zip(df['m'], df['n']))
    if not set(pairs) <= have:
        return False
    w = np.sort(df.loc[(df['m'] == pairs[0][0]) & (df['n'] == pairs[0][1]), 'omega'].to_numpy())
    return len(w) == nw and np.isclose(w[0], wmin) and np.isclose(w[-1], wmax)


def run_julia(args, path, pairs):
    cmd = [args.julia, '-t', str(args.threads), JL, '--params', args.params,
           '--pairs', pairs_str(pairs), '--wmin', str(args.wmin), '--wmax', str(args.wmax),
           '--nw', str(args.nw), '--out', path]
    if args.eta is not None:
        cmd += ['--eta', str(args.eta)]
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            cmd += [f'--{k}', repr(v)]
    print('$ ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def coef(d, name):
    return d[f'Re_{name}'].to_numpy() + 1j * d[f'Im_{name}'].to_numpy()


def ldos(z):
    return -np.imag(z) / np.pi


def plot_pair(df, m, n, comp, outdir, xlim, full_y=False):
    suf, sup = COMP[comp]
    d = df[(df['m'] == m) & (df['n'] == n)].sort_values('omega')
    w = d['omega'].to_numpy()
    al, be = coef(d, f'alpha_{suf}'), coef(d, f'beta_{suf}')
    gp, gm = coef(d, f'gp_{suf}'), coef(d, f'gm_{suf}')

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax0.plot(w, ldos(2 * al), '-', color=COLORS['tr'], lw=2.4, alpha=0.35, zorder=2,
             label=r'$2\alpha$ (Tr)')
    ax0.plot(w, ldos(al + be), '-', color=COLORS['up'], lw=1.4, zorder=3,
             label=r'$\alpha+\beta$ ($\uparrow_y$)')
    ax0.plot(w, ldos(al - be), '-', color=COLORS['dn'], lw=1.4, zorder=3,
             label=r'$\alpha-\beta$ ($\downarrow_y$)')
    ax0.set_ylabel(rf'$-\frac{{1}}{{\pi}}\,\mathrm{{Im}}\,\mathcal{{G}}^{{{sup}}}_{{{m}{n},0}}(\omega)$')

    ax1.plot(w, ldos(gp), '-', color=COLORS['p'], lw=1.4, zorder=3, label=r'$k=+1$: $\gamma_{+}$')
    ax1.plot(w, ldos(gm), '-', color=COLORS['m'], lw=1.4, zorder=3, label=r'$k=-1$: $\gamma_{-}$')
    ax1.set_ylabel(rf'$-\frac{{1}}{{\pi}}\,\mathrm{{Im}}\,\mathcal{{G}}^{{{sup}}}_{{{m}{n},\pm1}}(\omega)$')

    lo, hi = xlim if xlim is not None else (w.min(), w.max())
    for ax, tag in ((ax0, '(a)'), (ax1, '(b)')):
        ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r'$\omega\ (\tilde{\gamma})$')
        ax.text(0.03, 0.95, tag, transform=ax.transAxes, ha='left', va='top', fontsize=18)
        ax.legend(loc='upper right', frameon=False, fontsize=13, handlelength=1.6)
        _fmt_axes(ax)
        sel = (w >= lo) & (w <= hi)
        ys = [l.get_ydata()[sel] for l in ax.get_lines() if len(l.get_ydata()) == len(w)]
        if ys:
            allv = np.concatenate(ys)
            if full_y:
                y0, y1 = allv.min(), allv.max()
            else:
                # los estados ligados (deltas de ancho η fuera de la banda) ocupan
                # muy pocos puntos: se recortan para ver la estructura de la banda
                y0, y1 = np.percentile(allv, 0.2), np.percentile(allv, 99.8)
                y0, y1 = min(y0, 0.0), max(y1, 0.0)
            pad = 0.08 * (y1 - y0 if y1 > y0 else 1.0)
            ax.set_ylim(y0 - pad, y1 + 3.5 * pad)          # hueco arriba para la leyenda

    P._save(fig, outdir, f'kernel_elements_G{suf}_m{m}n{n}')


def main():
    ap = argparse.ArgumentParser(description='Armonicos k=0,±1 de G^{r,a,<}_{mn} del lead (precesion en y).')
    ap.add_argument('--pairs', default='1,1', help='pares m,n separados por ";" (m, n >= 1)')
    ap.add_argument('--params', default=DEFAULT_PARAMS,
                    help='params.txt de la corrida de TDNEGF ("none": valores por defecto de KParams)')
    # cualquier parametro se puede cambiar sobre los del params.txt (θ en grados)
    for k, h in (('gamma', 'hopping γ'), ('gso', 'Rashba γ_SO'), ('jsd', 'J_sd'),
                 ('theta', 'angulo del cono (grados)'), ('Omega', 'Ω'), ('mu', 'potencial quimico'),
                 ('beta', 'inverso de la temperatura')):
        ap.add_argument(f'--{k}', type=float, default=None, help=f'{h} (sobre el params.txt)')
    ap.add_argument('--wmin', type=float, default=-2.5)
    ap.add_argument('--wmax', type=float, default=2.5)
    ap.add_argument('--nw', type=int, default=20001)
    ap.add_argument('--eta', type=float, default=None, help='ensanchamiento (defecto: 2 dω)')
    ap.add_argument('--xlim', type=float, nargs=2, default=None, help='ventana en ω de las figuras')
    ap.add_argument('--full-y', action='store_true',
                    help='eje y completo (por defecto se recortan los picos de los estados ligados)')
    ap.add_argument('--julia', default='julia')
    ap.add_argument('--threads', default='auto')
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    if args.params != 'none':
        args.params = os.path.abspath(args.params)
    vals = read_params(args.params)
    overridden = args.params == 'none' or any(getattr(args, k) is not None for k in DEFAULTS)
    for k in DEFAULTS:
        if getattr(args, k) is not None:
            vals[k] = getattr(args, k)
    # carpeta: la de la corrida si no se toca nada; si no, la etiqueta de los valores usados
    tag = param_tag(vals) if overridden else os.path.basename(os.path.dirname(args.params))
    datadir = os.path.join(HERE, 'output', tag, 'data')
    figdir = os.path.join(HERE, 'output', tag, 'figures')
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    path = csv_path(datadir, pairs)
    eta = args.eta if args.eta is not None else 2 * (args.wmax - args.wmin) / (args.nw - 1)
    if args.recompute or not grid_matches(path, pairs, args.wmin, args.wmax, args.nw, eta, vals):
        run_julia(args, path, pairs)
    else:
        print(f'usando datos existentes: {path}')

    df = pd.read_csv(path, comment='#')
    for m, n in pairs:
        for comp in ('r', 'a', 'l'):
            plot_pair(df, m, n, comp, figdir, args.xlim, args.full_y)


if __name__ == '__main__':
    main()
