#!/usr/bin/env python3
"""
plot_kernel_elements.py -- grafica los armonicos k = 0, ±1 de las funciones de
Green del lead derecho (precesion en y) y las 6 componentes independientes del
kernel χ̌_k(ω) con su derivada i∂_ωχ̌_k(ω), calculados por kernel_elements.jl,
para los pares de sitios (m, n) que se pidan (por defecto 1,1;1,2;2,1;2,2).

Kernel (--what kernel|all): por par, 36 figuras de una sola grafica (Re e Im
superpuestas): {χ̌, i∂_ωχ̌} x {xx0, yy0, xz0, xy1, yx1, xx2} x {_gamma: ±3.5 γ̃,
_gamma_full: ±6 γ̃, _Omega: ±2.5 Ω}:
    figures/kernel_{chi,idchi}_{comp}_m{m}n{n}_{gamma,gamma_full,Omega}.*
Los valores en ω = 0 (para la LLG) quedan en data/kernel_chi0_<pares>.csv.

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

Figuras ordenadas por par y componente:
    figures/m{m}n{n}/harmonics/kernel_elements_G{r,a,l}_m{m}n{n}.*
    figures/m{m}n{n}/kernel/{xx0,yy0,xz0,xy1,yx1,xx2}/kernel_{chi,idchi}_<comp>_m{m}n{n}_{gamma,gamma_full,Omega}.*

    figures/all_pairs/kernel_{chi,idchi}_<comp>_all_{gamma,gamma_full,Omega}.*
        (con varios pares: malla en sitios, i por columnas y j por filas, eje y propio por panel)

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
from matplotlib.lines import Line2D

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


def run_tag(params):
    """Carpeta de la corrida de TDNEGF sin los campos buf.. y adyn.. (igual que run_tag del .jl)."""
    toks = os.path.basename(os.path.dirname(os.path.abspath(params))).split('_')
    return '_'.join(t for t in toks if not t.startswith(('buf', 'adyn')))


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

    sub = os.path.join(outdir, f'm{m}n{n}', 'harmonics')
    os.makedirs(sub, exist_ok=True)
    P._save(fig, sub, f'kernel_elements_G{suf}_m{m}n{n}')


# ---------------------------------------------------------------- kernel
KCOMP = {'xx0': ('xx', 0), 'yy0': ('yy', 0), 'xz0': ('xz', 0),
         'xy1': ('xy', 1), 'yx1': ('yx', 1), 'xx2': ('xx', 2)}
KCOLORS = {'Re': 'blue', 'Im': 'red'}


def kernel_cfg(args, vals):
    """Configuracion del kernel tal como la usara kernel_elements.jl (η por defecto 5 dω)."""
    dw = vals['Omega'] / args.k_nomega
    return dict(eta=args.k_eta if args.k_eta is not None else 5 * dw, nOmega=args.k_nomega,
                wide=args.k_wide, nwide=args.k_nwide, narrow=args.k_narrow, margin=args.k_margin)


def chi0_path(datadir, pairs):
    return os.path.join(datadir, 'kernel_chi0_' + '_'.join(f'{m}-{n}' for m, n in pairs) + '.csv')


def kernel_matches(datadir, pairs, vals, cfg):
    """True si existen los CSV del kernel de todos los pares con los mismos parametros y malla."""
    files = [chi0_path(datadir, pairs)] + [os.path.join(datadir, f'kernel_chi_m{m}n{n}.csv') for m, n in pairs]
    for path in files:
        if not os.path.isfile(path):
            return False
        with open(path) as fh:
            meta = dict(kv.split('=') for kv in fh.readline().lstrip('#').split())
        for k, ck in CSV_KEYS.items():
            want = np.deg2rad(vals[k]) if k == 'theta' else vals[k]
            if not np.isclose(float(meta[ck]), want, rtol=1e-9, atol=1e-14):
                return False
        for k, v in cfg.items():
            if k not in meta or not np.isclose(float(meta[k]), v, rtol=1e-9, atol=0):
                return False
    return True


def run_julia_kernel(args, datadir, pairs):
    cmd = [args.julia, '-t', str(args.threads), JL, '--kernel', '--params', args.params,
           '--pairs', pairs_str(pairs), '--datadir', datadir,
           '--k-nomega', str(args.k_nomega), '--k-wide', repr(args.k_wide), '--k-nwide', str(args.k_nwide),
           '--k-narrow', repr(args.k_narrow), '--k-margin', repr(args.k_margin)]
    if args.k_eta is not None:
        cmd += ['--k-eta', repr(args.k_eta)]
    for k in DEFAULTS:
        v = getattr(args, k)
        if v is not None:
            cmd += [f'--{k}', repr(v)]
    print('$ ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def plot_kernel(datadir, figdir, m, n, vals, gxlim=3.5):
    """Figuras del par (m,n): 6 componentes x (χ̌, i∂_ωχ̌) x 3 ventanas:
    _gamma (±gxlim γ̃), _gamma_full (toda la ventana ancha calculada) y _Omega (±k-narrow Ω)."""
    df = pd.read_csv(os.path.join(datadir, f'kernel_chi_m{m}n{n}.csv'), comment='#')
    gt = np.hypot(vals['gamma'], vals['gso'])
    # (sufijo, ventana del CSV, unidad, etiqueta, limite en x o None = todo)
    views = (('gamma', 'gamma', gt, r'$\omega/\tilde{\gamma}$', gxlim),
             ('gamma_full', 'gamma', gt, r'$\omega/\tilde{\gamma}$', None),
             ('Omega', 'Omega', vals['Omega'], r'$\omega/\Omega$', None))
    for comp, (ab, k) in KCOMP.items():
        sub = os.path.join(figdir, f'm{m}n{n}', 'kernel', comp)
        os.makedirs(sub, exist_ok=True)
        for q, lab in (('chi', rf'$\check{{\chi}}^{{{ab}}}_{{{m}{n},{k}}}(\omega)$'),
                       ('idchi', rf'$i\,\partial_\omega\check{{\chi}}^{{{ab}}}_{{{m}{n},{k}}}(\omega)$')):
            for suf, win, u, xlab, lim in views:
                d = df[df['window'] == win].sort_values('omega')
                x = d['omega'].to_numpy() / u
                if lim is not None:
                    sel = np.abs(x) <= lim * (1 + 1e-9)
                    d, x = d[sel], x[sel]
                fig, ax = plt.subplots(figsize=(6.4, 4.6))
                ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
                ax.axvline(0.0, color='0.6', lw=0.8, ls=':', zorder=1)
                for part in ('Re', 'Im'):
                    ax.plot(x, d[f'{part}_{q}_{comp}'].to_numpy(), '-', color=KCOLORS[part], lw=1.5,
                            zorder=3, label=rf'$\mathrm{{{part}}}$')
                ax.set_xlim(-lim if lim is not None else x.min(), lim if lim is not None else x.max())
                ax.set_xlabel(xlab)
                ax.set_ylabel(lab)
                ax.legend(loc='best', frameon=False, fontsize=14, handlelength=1.6)
                _fmt_axes(ax)
                P._save(fig, sub, f'kernel_{q}_{comp}_m{m}n{n}_{suf}')


def plot_kernel_grid(datadir, figdir, pairs, vals, gxlim=3.5):
    """Mallas n x n en sitios (i por columnas, j por filas) con todos los pares pedidos:
    una figura por componente (6) y por cantidad (χ̌, i∂_ωχ̌), en las 3 ventanas -> 36.
    Re e Im superpuestas; cada panel con su propio eje y."""
    sites = sorted({m for m, _ in pairs} | {n for _, n in pairs})
    nS = len(sites)
    data = {(m, n): pd.read_csv(os.path.join(datadir, f'kernel_chi_m{m}n{n}.csv'), comment='#')
            for m, n in pairs}
    outdir = os.path.join(figdir, 'all_pairs')
    os.makedirs(outdir, exist_ok=True)
    gt = np.hypot(vals['gamma'], vals['gso'])
    views = (('gamma', 'gamma', gt, r'$\omega/\tilde{\gamma}$', gxlim),
             ('gamma_full', 'gamma', gt, r'$\omega/\tilde{\gamma}$', None),
             ('Omega', 'Omega', vals['Omega'], r'$\omega/\Omega$', None))
    handles = [Line2D([], [], color=KCOLORS[pt], lw=2.0) for pt in ('Re', 'Im')]
    for comp, (ab, k) in KCOMP.items():
        for q, lab in (('chi', rf'$\check{{\chi}}^{{{ab}}}_{{ij,{k}}}(\omega)$'),
                       ('idchi', rf'$i\,\partial_\omega\check{{\chi}}^{{{ab}}}_{{ij,{k}}}(\omega)$')):
            for suf, win, u, xlab, lim in views:
                fig, axes = plt.subplots(nS, nS, figsize=(5.0 * nS, 4.0 * nS), squeeze=False)   # (5,4) por panel
                for r, j in enumerate(sites):
                    for c, i in enumerate(sites):
                        ax = axes[r, c]
                        if (i, j) not in data:
                            ax.set_axis_off()
                            continue
                        d = data[(i, j)]
                        d = d[d['window'] == win].sort_values('omega')
                        x = d['omega'].to_numpy() / u
                        if lim is not None:
                            sel = np.abs(x) <= lim * (1 + 1e-9)
                            d, x = d[sel], x[sel]
                        ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
                        ax.axvline(0.0, color='0.6', lw=0.8, ls=':', zorder=1)
                        for part in ('Re', 'Im'):
                            ax.plot(x, d[f'{part}_{q}_{comp}'].to_numpy(), '-', color=KCOLORS[part],
                                    lw=1.4, zorder=3)
                        ax.set_xlim(-lim if lim is not None else x.min(), lim if lim is not None else x.max())
                        if r == 0:
                            ax.set_title(rf'$i={i}$', fontsize=22, pad=10)
                        if c == 0:
                            ax.text(-0.24, 0.5, rf'$j={j}$', transform=ax.transAxes, rotation=90,
                                    ha='center', va='center', fontsize=22)
                        if r == nS - 1:
                            ax.set_xlabel(xlab)
                        _fmt_axes(ax)
                fig.tight_layout()
                # la cantidad graficada va como titulo de la leyenda, en el hueco libre del primer panel
                ax0 = next(a for a in axes.ravel() if a.axison)
                P._legend_en_hueco(ax0, handles, [r'$\mathrm{Re}$', r'$\mathrm{Im}$'], ncol=1,
                                   title=lab, fontsize=16, title_fontsize=18, handlelength=1.5)
                for ext in ('jpg', 'svg', 'pdf'):
                    path = os.path.join(outdir, f'kernel_{q}_{comp}_all_{suf}.{ext}')
                    fig.savefig(path, bbox_inches='tight', dpi=300)
                    print(f'  -> {path}')
                plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Armonicos k=0,±1 de G^{r,a,<}_{mn} del lead (precesion en y).')
    ap.add_argument('--pairs', default='1,1;1,2;2,1;2,2', help='pares m,n separados por ";" (m, n >= 1)')
    ap.add_argument('--what', choices=['all', 'harmonics', 'kernel'], default='all',
                    help='que graficar: armonicos de G, kernel, o ambos')
    # kernel (ver kernel_elements.jl): dω = Ω/k-nomega, η (defecto 5dω), ventanas ±k-wide γ̃ y ±k-narrow Ω
    ap.add_argument('--k-nomega', type=int, default=40)
    ap.add_argument('--k-eta', type=float, default=None)
    ap.add_argument('--k-wide', type=float, default=6.0)
    ap.add_argument('--k-nwide', type=int, default=1601)
    ap.add_argument('--k-xlim', type=float, default=3.5,
                    help='limite |ω|/γ̃ de las figuras _gamma (las _gamma_full muestran todo ±k-wide)')
    ap.add_argument('--k-narrow', type=float, default=2.5)
    ap.add_argument('--k-margin', type=float, default=2.0)
    ap.add_argument('--params', default=DEFAULT_PARAMS,
                    help='params.txt de la corrida de TDNEGF ("none": valores por defecto de KParams)')
    # cualquier parametro se puede cambiar sobre los del params.txt (θ en grados)
    for k, h in (('gamma', 'hopping γ'), ('gso', 'Rashba γ_SO'), ('jsd', 'J_sd'),
                 ('theta', 'angulo del cono (grados)'), ('Omega', 'Ω'), ('mu', 'potencial quimico'),
                 ('beta', 'inverso de la temperatura')):
        ap.add_argument(f'--{k}', type=float, default=None, help=f'{h} (sobre el params.txt)')
    ap.add_argument('--wmin', type=float, default=-3.5)
    ap.add_argument('--wmax', type=float, default=3.5)
    ap.add_argument('--nw', type=int, default=28001)
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
    tag = param_tag(vals) if overridden else run_tag(args.params)
    datadir = os.path.join(HERE, 'output', tag, 'data')
    figdir = os.path.join(HERE, 'output', tag, 'figures')
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    if args.what in ('all', 'harmonics'):
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

    if args.what in ('all', 'kernel'):
        cfg = kernel_cfg(args, vals)
        if args.recompute or not kernel_matches(datadir, pairs, vals, cfg):
            run_julia_kernel(args, datadir, pairs)
        else:
            print(f'usando kernel existente en {datadir}')
        for m, n in pairs:
            plot_kernel(datadir, figdir, m, n, vals, args.k_xlim)
        if len(pairs) > 1:
            plot_kernel_grid(datadir, figdir, pairs, vals, args.k_xlim)


if __name__ == '__main__':
    main()
