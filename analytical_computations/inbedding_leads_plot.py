#!/usr/bin/env python3

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HAS_LATEX = shutil.which('latex') is not None

plt.rcParams.update({
    'text.usetex': HAS_LATEX,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'] if HAS_LATEX else ['DejaVu Serif'],
    'font.size': 20,
    'axes.labelsize': 22,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.1,
    'axes.axisbelow': True,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LDOS_CSV = os.path.join(SCRIPT_DIR, 'output', 'inbedding_ldos.csv')
DEFAULT_RHO_CSV = os.path.join(SCRIPT_DIR, 'output', 'inbedding_rho_t.csv')


def _fmt_axes(ax):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=5.0, width=1.0)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if isinstance(fmt, mticker.ScalarFormatter):
            fmt.set_useMathText(False)


def _site_colors(sites):
    """Color por sitio: degradado oscuro->claro segun se entra al lead."""
    cmap = plt.get_cmap('viridis')
    return {n: cmap(0.12 + 0.72 * i / max(1, len(sites) - 1))
            for i, n in enumerate(sites)}


def _pick_sites(available, n=4):
    """
    Elige n sitios para graficar, con espaciado ~geometrico.

    El CSV puede traer muchos mas sitios de los que conviene superponer (p.ej.
    sweep_leads(sites=0:19) para la animacion). El espaciado geometrico da
    superficie + profundidades crecientes, que es donde esta el contraste.
    """
    av = sorted(available)
    if len(av) <= n:
        return av
    idx = sorted({min(len(av) - 1, int(round(f)) - 1)
                  for f in np.geomspace(1, len(av), n)})
    while len(idx) < n:                       # relleno si el redondeo colapso
        extra = next(i for i in range(len(av)) if i not in idx)
        idx = sorted(idx + [extra])
    return [av[i] for i in idx]


def _site_legend(ax, colors, loc='best'):
    handles = [Line2D([], [], color=c, lw=1.8, label=rf'$n={n}$')
               for n, c in colors.items()]
    ax.legend(handles=handles, frameon=False, loc=loc, fontsize=13,
              handlelength=1.6, labelspacing=0.3)


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def plot_ldos(df, outdir, lead, wmax=None):
    """Panel 1x2: LDOS de espin up y de espin down, lado a lado."""
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = [('LDOS_up', r'$A^{\uparrow}_n(\omega)\ (1/\gamma)$'),
            ('LDOS_dn', r'$A^{\downarrow}_n(\omega)\ (1/\gamma)$')]

    for ax, (col, ylab) in zip(axes, cols):
        for n in sites:
            s = df[df['site'] == n].sort_values('omega')
            w, y = s['omega'].to_numpy(), s[col].to_numpy()
            if wmax is not None:
                sel = np.abs(w) <= wmax
                w, y = w[sel], y[sel]
            ax.plot(w, y, '-', color=colors[n], lw=1.4, zorder=3)
        ax.set_xlabel(r'$\omega\ (\gamma)$')
        ax.set_ylabel(ylab)
        ax.set_xlim(w.min(), w.max())
        ax.set_ylim(bottom=0.0)
        _fmt_axes(ax)

    _site_legend(axes[0], colors, loc='upper right')
    _save(fig, outdir, f'inbedding_ldos_{lead}')


def plot_rho_t(df, outdir, lead, Omega):
    """Panel 2x2: las cuatro componentes de la matriz densidad."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    cols = [('n_up',        r'$\rho_{\uparrow\uparrow}(t)$'),
            ('n_dn',        r'$\rho_{\downarrow\downarrow}(t)$'),
            ('Re_rho_updn', r'$\text{Re}\,\rho_{\uparrow\downarrow}(t)$'),
            ('Im_rho_updn', r'$\text{Im}\,\rho_{\uparrow\downarrow}(t)$')]

    for ax, (col, ylab) in zip(axes.flat, cols):
        for n in sites:
            s = df[df['site'] == n].sort_values('t')
            x = s['t'].to_numpy() / T
            ax.plot(x, s[col].to_numpy(), '-', color=colors[n], lw=1.4, zorder=3)
        ax.set_ylabel(ylab)
        ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
    _site_legend(axes[0, 0], colors, loc='center right')
    _save(fig, outdir, f'inbedding_rho_t_{lead}')


def plot_spin_t(df, outdir, lead, Omega):
    """Panel 2x2: un sitio por panel, las tres componentes de espin en cada uno."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    comps = [('sx', 'red',   r'$\langle\sigma_x\rangle$'),
             ('sy', 'blue',  r'$\langle\sigma_y\rangle$'),
             ('sz', 'black', r'$\langle\sigma_z\rangle$')]

    for ax, n in zip(axes.flat, sites):
        s = df[df['site'] == n].sort_values('t')
        x = s['t'].to_numpy() / T
        for col, color, _ in comps:
            ax.plot(x, s[col].to_numpy(), '-', color=color, lw=1.5, zorder=3)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        ax.set_title(rf'$n={n}$', fontsize=18)
        ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax)

    for ax in axes.flat[len(sites):]:          # si hay menos de 4 sitios
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\langle\boldsymbol{\sigma}\rangle_n(t)$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')

    handles = [Line2D([], [], color=c, lw=1.8, label=lab) for _, c, lab in comps]
    axes[0, 0].legend(handles=handles, frameon=False, loc='best', fontsize=13,
                      handlelength=1.6, labelspacing=0.3)
    _save(fig, outdir, f'inbedding_spin_t_{lead}')


def plot_occupation_t(df, outdir, lead, Omega):
    """Panel 1x2: ocupacion de espin up y de espin down, lado a lado."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = [('n_up', r'$n^{\uparrow}(t)$'),
            ('n_dn', r'$n^{\downarrow}(t)$')]

    for ax, (col, ylab) in zip(axes, cols):
        for n in sites:
            s = df[df['site'] == n].sort_values('t')
            x = s['t'].to_numpy() / T
            ax.plot(x, s[col].to_numpy(), '-', color=colors[n], lw=1.4, zorder=3)
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
        ax.set_ylabel(ylab)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0.0, 1.0)
        _fmt_axes(ax)

    _site_legend(axes[0], colors, loc='center right')
    _save(fig, outdir, f'inbedding_occ_t_{lead}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ldos-csv', default=DEFAULT_LDOS_CSV)
    ap.add_argument('--rho-csv', default=DEFAULT_RHO_CSV)
    ap.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'output'))
    ap.add_argument('--wmax', type=float, default=None,
                    help='recorta el eje de omega (|omega|<=wmax) en el plot de LDOS')
    ap.add_argument('--Omega', type=float, default=0.005,
                    help='frecuencia de driving usada en el barrido (para el eje t/T)')
    ap.add_argument('--lead', default='R', help='lead a graficar (por defecto R)')
    ap.add_argument('--sites', default=None,
                    help='sitios a graficar, separados por coma (p.ej. 0,1,4,16). '
                         'Por defecto se eligen 4 con espaciado geometrico.')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    lead = args.lead.strip()
    df_ldos = pd.read_csv(args.ldos_csv)
    df_rho = pd.read_csv(args.rho_csv)
    dl = df_ldos[df_ldos['lead'] == lead]
    dr = df_rho[df_rho['lead'] == lead]
    if dl.empty and dr.empty:
        raise SystemExit(f'no hay datos para el lead {lead}')

    # El CSV puede traer muchos sitios; los paneles solo aguantan unos pocos.
    if args.sites is not None:
        sel = [int(s) for s in args.sites.split(',')]
    else:
        sel = _pick_sites(sorted(set(dr['site'].unique()) | set(dl['site'].unique())))
    print(f'  lead {lead}, sitios graficados: {sel}')
    dl = dl[dl['site'].isin(sel)]
    dr = dr[dr['site'].isin(sel)]

    plot_ldos(dl, args.outdir, lead, wmax=args.wmax)
    plot_rho_t(dr, args.outdir, lead, Omega=args.Omega)
    plot_spin_t(dr, args.outdir, lead, Omega=args.Omega)
    plot_occupation_t(dr, args.outdir, lead, Omega=args.Omega)


if __name__ == '__main__':
    main()
