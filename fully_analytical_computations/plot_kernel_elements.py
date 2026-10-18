#!/usr/bin/env python3
"""
plot_kernel_elements.py -- SOLO figuras (no llama a Julia) a partir de los CSV que
escribe kernel_elements.jl (en el cluster: run_cluster.sh) en output/<tag>/data/:

    kernel_elements_1-1.csv     armonicos k = 0, ±1 de G^{r,a,<} del sitio (1,1)
    kernel_chi_m{i}n{j}.csv     χ̌_k(ω) e i∂_ωχ̌_k(ω) (ventanas γ̃ y Ω), un archivo por par
    kernel_tau_m{i}n{j}.csv     χ̌_k(τ), un archivo por par

Si falta un CSV o su encabezado no coincide con los parametros/mallas pedidos, se
detiene y dice que archivo es y como generarlo.

Figuras (por defecto solo .jpg; --formats jpg,svg,pdf para el resto):
    figures/m{i}n{j}/harmonics/                 (solo m1n1) G^{r,a,<}: -(1/π) Im, k = 0 y ±1
    figures/m{i}n{j}/kernel/<comp>/             χ̌, i∂χ̌  x  _gamma, _gamma_full, _Omega
    figures/m{i}n{j}/kernel_tau/<comp>/         χ̌(τ)
    figures/m{i}n{j}/kernel_tmap/<spin>/        χ(t,t'): _tt200, _ttT, _ttau
    figures/configs/F{a}-{b}/kernel/<comp>/     mallas 2x2 (a,a)(a,b)(b,a)(b,b), nombre ..._F{a}-{b}_...
    figures/configs/F{a}-{b}/kernel_tau/<comp>/
    figures/configs/F{a}-{b}/kernel_tmap/<spin>/
  <comp> ∈ {xx0, yy0, xz0, xy1, yx1, xx2},  <spin> ∈ {xx, yy, xz, xy, yx}

Uso:
    python plot_kernel_elements.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005
    python plot_kernel_elements.py --tag <tag> --what kernel --configs "1,2;3,6;5,9"
    python plot_kernel_elements.py --tag <tag> --sites 10 --formats jpg,pdf
    python plot_kernel_elements.py                      # sin --tag: params.txt de setup2 (como antes)
"""

import argparse
import os
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
                              'DFF', 'F1-2', 'gso0p1_jsd0p2_th10deg_Om0p005_buf4_adyn0p0', 'params.txt')

FORMATS = ['jpg']                                             # --formats

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
    d = os.path.dirname(os.path.abspath(params))
    tag = '_'.join(t for t in os.path.basename(d).split('_') if not t.startswith(('buf', 'adyn')))
    # estructura nueva <eje>_precession/output/DFF/F../<tag>: mismo tag que el viejo setup2_<tag>
    lay = os.path.basename(os.path.dirname(os.path.dirname(d)))
    pre = {'DFF': 'setup2_', 'FDF': 'setup1_'}.get(lay, '')
    return tag if tag.startswith('setup') else pre + tag


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


def need(ok, what, hint):
    """Sin Julia en local: si falta un CSV (o no coincide), se detiene explicando como generarlo."""
    if not ok:
        raise SystemExit(f'falta o no coincide: {what}\n  generalo en el cluster con: {hint}')


def meta_vals(datadir):
    """Parametros (γ, γSO, J_sd, θ[grados], Ω, μ, β) del encabezado de un CSV de datadir."""
    for name in sorted(os.listdir(datadir)) if os.path.isdir(datadir) else []:
        if not name.endswith('.csv'):
            continue
        with open(os.path.join(datadir, name)) as fh:
            head = fh.readline()
        if not head.startswith('#'):
            continue
        meta = dict(kv.split('=') for kv in head.lstrip('#').split() if '=' in kv)
        if all(ck in meta for ck in CSV_KEYS.values()):
            v = {k: float(meta[ck]) for k, ck in CSV_KEYS.items()}
            v['theta'] = float(np.rad2deg(v['theta']))
            return v
    raise SystemExit(f'no hay CSV con encabezado de parametros en {datadir}')


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

    _save_fig(fig, os.path.join(outdir, f'm{m}n{n}', 'harmonics'), f'kernel_elements_G{suf}_m{m}n{n}', tight=True)


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
    files = [os.path.join(datadir, f'kernel_chi_m{m}n{n}.csv') for m, n in pairs]
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
                _save_fig(fig, sub, f'kernel_{q}_{comp}_m{m}n{n}_{suf}', tight=True)


def config_pairs(a, b):
    return [(a, a), (a, b), (b, a), (b, b)]


def plot_kernel_grid(datadir, figdir, pairs, vals, gxlim=3.5, label='all'):
    """Mallas en sitios (i por columnas, j por filas) de los pares dados (una configuracion
    D-F-F: (a,a)(a,b)(b,a)(b,b)): una figura por componente (6) y cantidad (χ̌, i∂_ωχ̌), en
    las 3 ventanas -> 36, en figures/configs/F<label>/kernel/<comp>/. Re e Im superpuestas;
    cada panel con su propio eje y."""
    sites = sorted({m for m, _ in pairs} | {n for _, n in pairs})
    nS = len(sites)
    data = {(m, n): pd.read_csv(os.path.join(datadir, f'kernel_chi_m{m}n{n}.csv'), comment='#')
            for m, n in pairs}
    outroot = os.path.join(figdir, 'configs', f'F{label}', 'kernel')
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
                _save_fig(fig, os.path.join(outroot, comp), f'kernel_{q}_{comp}_F{label}_{suf}')


# ---------------------------------------------------------------- kernel en tiempo
# armonicos que forman cada componente de espin en (t,t'):  (k, coeficiente, columna k>0 o k=0)
#   χ(t,t-τ) = Re χ̌_0(τ) + Σ_{k>0} 2 Re[c e^{-ikΩt} χ̌_k(τ)]
SPIN = {'xx': ((0, 1, 'xx0'), (2, 1, 'xx2')),
        'yy': ((0, 1, 'yy0'),),
        'xz': ((0, 1, 'xz0'), (2, -1j, 'xx2')),        # χ^{xz}_{±2} = ∓i χ^{xx}_{±2}
        'xy': ((1, 1, 'xy1'),),
        'yx': ((1, 1, 'yx1'),)}
TCMAP = plt.get_cmap('seismic')


def tau_cfg(args, vals):
    """Configuracion del kernel en τ tal como la usara kernel_elements.jl (τmax por defecto T)."""
    T = 2 * np.pi / vals['Omega']
    return dict(eta=args.t_eta, dtau=args.t_dtau, taumax=args.t_taumax if args.t_taumax else T,
                log2N=args.t_log2n)


def tau_matches(datadir, pairs, vals, cfg):
    for m, n in pairs:
        path = os.path.join(datadir, f'kernel_tau_m{m}n{n}.csv')
        if not os.path.isfile(path):
            return False
        with open(path) as fh:
            meta = dict(kv.split('=') for kv in fh.readline().lstrip('#').split())
        for k, ck in CSV_KEYS.items():
            want = np.deg2rad(vals[k]) if k == 'theta' else vals[k]
            if not np.isclose(float(meta[ck]), want, rtol=1e-9, atol=1e-14):
                return False
        for k, v in cfg.items():
            if k not in meta or not np.isclose(float(meta[k]), v, rtol=1e-6, atol=0):
                return False
    return True


def load_tau(datadir, m, n):
    """τ (malla simetrica) y χ̌_k(τ) complejo por componente de KCOMP."""
    d = pd.read_csv(os.path.join(datadir, f'kernel_tau_m{m}n{n}.csv'), comment='#')
    tau = d['tau'].to_numpy()
    return tau, {c: d[f'Re_chi_{c}'].to_numpy() + 1j * d[f'Im_chi_{c}'].to_numpy() for c in KCOMP}


def _save_fig(fig, outdir, name, tight=False):
    os.makedirs(outdir, exist_ok=True)
    if tight:
        fig.tight_layout()
    for ext in FORMATS:
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def _tau_curve(ax, tau, z, tplot):
    sel = (tau >= 0) & (tau <= tplot * (1 + 1e-12))
    ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
    for part, f in (('Re', np.real), ('Im', np.imag)):
        ax.plot(tau[sel], f(z[sel]), '-', color=KCOLORS[part], lw=1.2, zorder=3)
    ax.set_xlim(0.0, tplot)
    _fmt_axes(ax)


def plot_tau(datadir, figdir, pairs, tplot=200.0, configs=()):
    """χ̌_k(τ), τ ∈ [0, tplot]: una figura por par y componente, y la malla 2x2 por
    componente para cada configuracion (a,b) en figures/configs/F{a}-{b}/kernel_tau/."""
    data = {pr: load_tau(datadir, *pr) for pr in pairs}
    handles = [Line2D([], [], color=KCOLORS[pt], lw=2.0) for pt in ('Re', 'Im')]
    labs = [r'$\mathrm{Re}$', r'$\mathrm{Im}$']
    for comp, (ab, k) in KCOMP.items():
        for (m, n), (tau, chi) in data.items():
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            _tau_curve(ax, tau, chi[comp], tplot)
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel(rf'$\check{{\chi}}^{{{ab}}}_{{{m}{n},{k}}}(\tau)$')
            ax.legend(handles, labs, loc='best', frameon=False, fontsize=14, handlelength=1.6)
            _save_fig(fig, os.path.join(figdir, f'm{m}n{n}', 'kernel_tau', comp), f'kernel_tau_{comp}_m{m}n{n}')
        for (fa, fb) in configs:
            cp = config_pairs(fa, fb)
            sites = [fa, fb]; nS = 2
            fig, axes = plt.subplots(nS, nS, figsize=(5.0 * nS, 4.0 * nS), squeeze=False)
            for r, j in enumerate(sites):
                for c, i in enumerate(sites):
                    ax = axes[r, c]
                    tau, chi = data[(i, j)]
                    _tau_curve(ax, tau, chi[comp], tplot)
                    if r == 0:
                        ax.set_title(rf'$i={i}$', fontsize=22, pad=10)
                    if c == 0:
                        ax.text(-0.32, 0.5, rf'$j={j}$', transform=ax.transAxes, rotation=90,
                                ha='center', va='center', fontsize=22)
                    if r == nS - 1:
                        ax.set_xlabel(r'$\tau$')
            fig.tight_layout()
            ax0 = next(a for a in axes.ravel() if a.axison)
            P._legend_en_hueco(ax0, handles, labs, ncol=1,
                               title=rf'$\check{{\chi}}^{{{ab}}}_{{ij,{k}}}(\tau)$',
                               fontsize=16, title_fontsize=18, handlelength=1.5)
            _save_fig(fig, os.path.join(figdir, 'configs', f'F{fa}-{fb}', 'kernel_tau', comp),
                      f'kernel_tau_{comp}_F{fa}-{fb}')


def _block(Z, fr, fc):
    """Promedio por bloques fr x fc (sin submuestrear: evita aliasing al dibujar)."""
    nr, nc = (Z.shape[0] // fr) * fr, (Z.shape[1] // fc) * fc
    return Z[:nr, :nc].reshape(nr // fr, fr, nc // fc, fc).mean(axis=(1, 3))


def spin_map(tau, chi, spin, Omega, view, tplot, npix):
    """
    Mapa real de χ^{spin}_{ij}, promediado por bloques de f x f puntos de la malla fina
    (sin submuestrear: evita aliasing). view:
      'tt200': (t,t') ∈ [0,tplot]^2      'ttT': (t,t') ∈ [0,T]^2      'ttau': (t,τ) ∈ [0,T]x[0,tplot]
    El promedio es EXACTO y no construye la matriz fina: con a = Af+p, b = Bf+q,
        <e^{-ikΩ t_a} χ̌_k(t_a - t_b)>_bloque = e^{-ikΩ A f dt} S_k[A-B],
        S_k[D] = f^{-2} Σ_l w_k(l) χ̌_k((Df+l) dt),  w_k(l) = Σ_{p=max(0,l)}^{min(f-1,f-1+l)} e^{-ikΩ p dt},
    y en (t,τ) el bloque se separa en <e^{-ikΩt}>_t <χ̌_k(τ)>_τ.
    Devuelve (Z[x,y], extent, max|Im| relativo de los k = 0).
    """
    dt = tau[1] - tau[0]; N0 = int(np.argmin(np.abs(tau)))          # indice de τ = 0
    T = 2 * np.pi / Omega
    terms = [(k, cf, chi[col]) for k, cf, col in SPIN[spin]]
    imrel = max(np.abs(z.imag).max() / max(np.abs(z).max(), 1e-300) for k, _, z in terms if k == 0) \
        if any(k == 0 for k, _, _ in terms) else 0.0

    if view in ('tt200', 'ttT'):
        W = tplot if view == 'tt200' else T
        n = int(round(W / dt))
        if N0 - (n - 1) < 0 or N0 + (n - 1) >= len(tau):
            raise SystemExit(f'kernel_tau: τmax < {W:.1f}; corre de nuevo con --t-taumax mayor')
        f = max(1, int(np.ceil(n / npix)))
        nb = n // f
        ls = np.arange(-(f - 1), f)                                    # l = p - q
        Ds = np.arange(-(nb - 1), nb)                                  # D = A - B
        idx = N0 + Ds[:, None] * f + ls[None, :]
        A = np.arange(nb)
        Dm = A[:, None] - A[None, :] + (nb - 1)                        # indice de D en Ds
        Z = np.zeros((nb, nb))
        for k, cf, z in terms:
            p = np.arange(f)
            ph = np.exp(-1j * k * Omega * p * dt)
            w = np.array([ph[max(0, l):min(f - 1, f - 1 + l) + 1].sum() for l in ls])
            S = (z[idx] * w[None, :]).sum(axis=1) / f**2
            if k == 0:
                Z += np.real(S[Dm])
            else:
                Z += 2 * np.real(cf * np.exp(-1j * k * Omega * A * f * dt)[:, None] * S[Dm])
        ext = (0, nb * f * dt, 0, nb * f * dt)
        if view == 'ttT':
            ext = tuple(e / T for e in ext)
        return Z, ext, imrel
    # 'ttau': separable en t y τ
    nt = int(round(T / dt)); ntau = int(round(tplot / dt)) + 1
    ft = max(1, int(np.ceil(nt / npix))); fu = max(1, int(np.ceil(ntau / npix)))
    nbt, nbu = nt // ft, ntau // fu
    ta = np.arange(nbt * ft) * dt
    Z = np.zeros((nbt, nbu))
    for k, cf, z in terms:
        zu = z[N0:N0 + nbu * fu].reshape(nbu, fu).mean(axis=1)        # <χ̌_k(τ)>_bloque
        if k == 0:
            Z += np.real(zu)[None, :]
        else:
            et = np.exp(-1j * k * Omega * ta).reshape(nbt, ft).mean(axis=1)
            Z += 2 * np.real(cf * et[:, None] * zu[None, :])
    return Z, (0, nbt * ft * dt / T, 0, nbu * fu * dt), imrel


VIEWS = {'tt200': (r'$t$', r"$t'$"), 'ttT': (r'$t/T$', r"$t'/T$"), 'ttau': (r'$t/T$', r'$\tau$')}


def _draw_map(ax, fig, Z, ext, cbar_lab):
    v = np.abs(Z).max() or 1.0
    im = ax.imshow(Z.T, origin='lower', extent=ext, aspect='auto', interpolation='nearest',
                   cmap=TCMAP, vmin=-v, vmax=v)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_lab, fontsize=16)
    P._sci_cbar(cb)
    _fmt_axes(ax)


def plot_tmaps(datadir, figdir, pairs, vals, tplot=200.0, npix=800, configs=()):
    """Mapas de las 5 componentes de espin en las 3 vistas: por par y mallas 2x2 por
    configuracion (a,b) en figures/configs/F{a}-{b}/kernel_tmap/."""
    data = {pr: load_tau(datadir, *pr) for pr in pairs}
    keep = {pr for (fa, fb) in configs for pr in config_pairs(fa, fb)}     # mapas que usan las mallas
    for spin in SPIN:
        for view, (xl, yl) in VIEWS.items():
            maps = {}
            for (m, n), (tau, chi) in data.items():
                Z, ext, imrel = spin_map(tau, chi, spin, vals['Omega'], view, tplot, npix)
                if (m, n) in keep:
                    maps[(m, n)] = (Z, ext)
                if imrel > 1e-6:
                    print(f'  aviso: {spin} ({m},{n}) max|Im χ̌_0|/max|χ̌_0| = {imrel:.1e}')
                fig, ax = plt.subplots(figsize=(6.4, 4.6))
                _draw_map(ax, fig, Z, ext, rf"$\chi^{{{spin}}}_{{{m}{n}}}(t,t')$")
                ax.set_xlabel(xl); ax.set_ylabel(yl)
                _save_fig(fig, os.path.join(figdir, f'm{m}n{n}', 'kernel_tmap', spin),
                          f'kernel_tmap_{spin}_m{m}n{n}_{view}')
            for (fa, fb) in configs:
                sites = [fa, fb]; nS = 2
                fig, axes = plt.subplots(nS, nS, figsize=(5.0 * nS, 4.0 * nS), squeeze=False)
                for r, j in enumerate(sites):
                    for c, i in enumerate(sites):
                        ax = axes[r, c]
                        Z, ext = maps[(i, j)]
                        _draw_map(ax, fig, Z, ext, rf"$\chi^{{{spin}}}_{{{i}{j}}}$")
                        if r == 0:
                            ax.set_title(rf'$i={i}$', fontsize=22, pad=10)
                        if c == 0:
                            ax.text(-0.30, 0.5, rf'$j={j}$', transform=ax.transAxes, rotation=90,
                                    ha='center', va='center', fontsize=22)
                            ax.set_ylabel(yl)
                        if r == nS - 1:
                            ax.set_xlabel(xl)
                fig.tight_layout()
                _save_fig(fig, os.path.join(figdir, 'configs', f'F{fa}-{fb}', 'kernel_tmap', spin),
                          f'kernel_tmap_{spin}_F{fa}-{fb}_{view}')


def main():
    ap = argparse.ArgumentParser(description='Figuras de los armonicos de G y del kernel (sin Julia).')
    ap.add_argument('--tag', default=None, help='carpeta output/<tag> (parametros leidos de los CSV)')
    ap.add_argument('--sites', type=int, default=10, help='pares (i,j) ∈ [1,N]^2')
    ap.add_argument('--pairs', default=None, help='en vez de --sites: pares m,n separados por ";"')
    ap.add_argument('--configs', default='1,2;3,6;5,9', help='configuraciones D-F-F (a,b) de las mallas 2x2')
    ap.add_argument('--formats', default='jpg', help='formatos de las figuras, p. ej. jpg,svg,pdf')
    ap.add_argument('--what', choices=['all', 'harmonics', 'kernel', 'tau'], default='all')
    # kernel (ver kernel_elements.jl): dω = Ω/k-nomega, η (defecto 5dω), ventanas ±k-wide γ̃ y ±k-narrow Ω
    ap.add_argument('--k-nomega', type=int, default=40)
    ap.add_argument('--k-eta', type=float, default=None)
    ap.add_argument('--k-wide', type=float, default=6.0)
    ap.add_argument('--k-nwide', type=int, default=1601)
    ap.add_argument('--k-xlim', type=float, default=3.5,
                    help='limite |ω|/γ̃ de las figuras _gamma (las _gamma_full muestran todo ±k-wide)')
    ap.add_argument('--k-narrow', type=float, default=2.5)
    ap.add_argument('--k-margin', type=float, default=2.0)
    # kernel en tiempo (ver kernel_elements.jl --tau)
    ap.add_argument('--t-dtau', type=float, default=0.1)
    ap.add_argument('--t-taumax', type=float, default=None, help='defecto: T = 2π/Ω')
    ap.add_argument('--t-log2n', type=int, default=22)
    ap.add_argument('--t-eta', type=float, default=5e-5)
    ap.add_argument('--t-plot', type=float, default=200.0, help="τ maximo de las curvas y ventana (t,t') corta")
    ap.add_argument('--t-npix', type=int, default=800, help='pixeles por eje de los mapas (promedio por bloques)')
    # armonicos de G (solo el par (1,1))
    ap.add_argument('--wmin', type=float, default=-3.5)
    ap.add_argument('--wmax', type=float, default=3.5)
    ap.add_argument('--nw', type=int, default=28001)
    ap.add_argument('--eta', type=float, default=None, help='ensanchamiento de los armonicos (defecto: 2 dω)')
    ap.add_argument('--xlim', type=float, nargs=2, default=None, help='ventana en ω de los armonicos')
    ap.add_argument('--full-y', action='store_true',
                    help='eje y completo en los armonicos (por defecto se recortan los estados ligados)')
    # sin --tag: parametros desde un params.txt de TDNEGF (y overrides), como antes
    ap.add_argument('--params', default=DEFAULT_PARAMS)
    for k in DEFAULTS:
        ap.add_argument(f'--{k}', type=float, default=None)
    args = ap.parse_args()

    FORMATS[:] = [f.strip() for f in args.formats.split(',') if f.strip()]
    if args.tag:
        tag = args.tag
        datadir = os.path.join(HERE, 'output', tag, 'data')
        vals = meta_vals(datadir)
        hint_p = f'--tag {tag} (ver run_cluster.sh)'
    else:
        if args.params != 'none':
            args.params = os.path.abspath(args.params)
        vals = read_params(args.params)
        overridden = args.params == 'none' or any(getattr(args, k) is not None for k in DEFAULTS)
        for k in DEFAULTS:
            if getattr(args, k) is not None:
                vals[k] = getattr(args, k)
        tag = param_tag(vals) if overridden else run_tag(args.params)
        datadir = os.path.join(HERE, 'output', tag, 'data')
        hint_p = f'--params {args.params}'
    figdir = os.path.join(HERE, 'output', tag, 'figures')

    pairs = parse_pairs(args.pairs) if args.pairs else \
        [(i, j) for i in range(1, args.sites + 1) for j in range(1, args.sites + 1)]
    configs = []
    for tok in args.configs.split(';'):
        if tok.strip():
            a, b = (int(x) for x in tok.split(','))
            if all(pr in pairs for pr in config_pairs(a, b)):
                configs.append((a, b))
            else:
                print(f'  configuracion F{a}-{b}: faltan pares en la lista pedida, se omite')
    label = f'--sites {args.sites}' if not args.pairs else f'--pairs "{args.pairs}"'
    print(f'output/{tag}   |   {len(pairs)} pares   |   configuraciones {configs}   |   formatos {FORMATS}')

    if args.what in ('all', 'harmonics'):
        hp = [(1, 1)]
        path = csv_path(datadir, hp)
        eta = args.eta if args.eta is not None else 2 * (args.wmax - args.wmin) / (args.nw - 1)
        need(grid_matches(path, hp, args.wmin, args.wmax, args.nw, eta, vals), path,
             f'julia kernel_elements.jl {hint_p} --pairs "1,1"')
        df = pd.read_csv(path, comment='#')
        for comp in ('r', 'a', 'l'):
            plot_pair(df, 1, 1, comp, figdir, args.xlim, args.full_y)

    if args.what in ('all', 'kernel'):
        need(kernel_matches(datadir, pairs, vals, kernel_cfg(args, vals)), f'kernel_chi_m*n*.csv en {datadir}',
             f'julia kernel_elements.jl {hint_p} --kernel {label}')
        for m, n in pairs:
            plot_kernel(datadir, figdir, m, n, vals, args.k_xlim)
        for (a, b) in configs:
            plot_kernel_grid(datadir, figdir, config_pairs(a, b), vals, args.k_xlim, f'{a}-{b}')

    if args.what in ('all', 'tau'):
        need(tau_matches(datadir, pairs, vals, tau_cfg(args, vals)), f'kernel_tau_m*n*.csv en {datadir}',
             f'julia kernel_elements.jl {hint_p} --tau {label}')
        plot_tau(datadir, figdir, pairs, args.t_plot, configs)
        plot_tmaps(datadir, figdir, pairs, vals, args.t_plot, args.t_npix, configs)


if __name__ == '__main__':
    main()
