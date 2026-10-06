#!/usr/bin/env python3
"""
plot_fully_analitical_floquet_gf.py -- figuras de los resultados puramente
analiticos de fully_analitical_floquet_gf.jl (mismas figuras que
floquet_gf_plot.py, mas los armonicos k=+-1 de G^r).

Lee  output/fa_floquet_ldos.csv  y  output/fa_floquet_rho_t.csv  y escribe
(jpg, svg, pdf):
    fa_floquet_ldos        A_up, A_dn, total vs omega
    fa_floquet_rho_t       rho^{uu}, rho^{dd}, Re/Im rho^{ud} vs t
    fa_floquet_occ_t       n_up, n_dn, n_tot vs t
    fa_floquet_spin_t      <sigma^alpha>(t)
    fa_floquet_Gr_pm1      Im G^{r,ud}_{+1}(omega), Im G^{r,du}_{-1}(omega)
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
DEFAULT_LDOS_CSV = os.path.join(SCRIPT_DIR, 'output', 'fa_floquet_ldos.csv')
DEFAULT_RHO_CSV = os.path.join(SCRIPT_DIR, 'output', 'fa_floquet_rho_t.csv')

# mismo criterio de color que floquet_gf_plot.py / inbedding_leads_plot.py
SPIN_COLORS = {'sx': 'blue', 'sy': 'green', 'sz': 'red'}


def _fmt_axes(ax):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=5.0, width=1.0)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if isinstance(fmt, mticker.ScalarFormatter):
            fmt.set_useMathText(False)


def _save(fig, outdir, name):
    plt.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def _omega_sel(df, wmax):
    w = df['omega'].to_numpy()
    return np.ones_like(w, bool) if wmax is None else np.abs(w) <= wmax


def plot_ldos(df, outdir, wmax=None):
    sel = _omega_sel(df, wmax)
    w = df['omega'].to_numpy()[sel]
    up, dn, tot = (df['LDOS_up'].to_numpy()[sel], df['LDOS_dn'].to_numpy()[sel],
                   df['LDOS_tot'].to_numpy()[sel])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(w, tot, '--', color='black', lw=1.0, zorder=2,
            label=r'$\text{Total}= \uparrow+ \downarrow$')
    ax.plot(w, up, '-', color='red', lw=1.5, zorder=3, label=r'$\uparrow$')
    ax.plot(w, dn, '-', color='blue', lw=1.5, zorder=3, label=r'$\downarrow$')
    ax.set_xlabel(r'$\omega\ (\gamma)$')
    ax.set_ylabel(r'$A(\omega)\ (1/\gamma)$')
    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(-0.1, 10.0)
    ax.legend(frameon=False, loc='best')
    _fmt_axes(ax)
    _save(fig, outdir, 'fa_floquet_ldos')


def plot_Gr_pm1(df, outdir, wmax=None):
    """Partes imaginarias de los armonicos k=+-1 de G^r (unicas componentes no nulas)."""
    sel = _omega_sel(df, wmax)
    w = df['omega'].to_numpy()[sel]
    p1 = df['ImGr_p1_updn'].to_numpy()[sel]
    m1 = df['ImGr_m1_dnup'].to_numpy()[sel]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
    ax.plot(w, p1, '-', color='black', lw=1.5, zorder=3,
            label=r'$\text{Im}\,\mathcal{G}^{r,\uparrow\downarrow}_{+1}$')
    ax.plot(w, m1, '--', color='purple', lw=1.5, zorder=3,
            label=r'$\text{Im}\,\mathcal{G}^{r,\downarrow\uparrow}_{-1}$')
    ax.set_xlabel(r'$\omega\ (\gamma)$')
    ax.set_ylabel(r'$\text{Im}\,\mathcal{G}^{r}_{\pm1}(\omega)\ (1/\gamma)$')
    ax.set_xlim(w.min(), w.max())
    ax.legend(frameon=False, loc='best')
    _fmt_axes(ax)
    _save(fig, outdir, 'fa_floquet_Gr_pm1')


def plot_rho_t(df, outdir, Omega):
    x = df['t'].to_numpy() / (2 * np.pi / Omega)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, df['n_up'], '-', color='red', lw=1.5, zorder=3,
            label=r'$\rho^{\uparrow\uparrow}$')
    ax.plot(x, df['n_dn'], '-', color='blue', lw=1.5, zorder=3,
            label=r'$\rho^{\downarrow\downarrow}$')
    ax.plot(x, df['Re_rho_updn'], '--', color='0.3', lw=1.3, zorder=2,
            label=r'$\text{Re}\,\rho^{\uparrow\downarrow}$')
    ax.plot(x, df['Im_rho_updn'], ':', color='0.3', lw=1.5, zorder=2,
            label=r'$\text{Im}\,\rho^{\uparrow\downarrow}$')
    ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
    ax.set_ylabel(r'$\rho(t)$')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(top=1.1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([r'$0$', r'$0.5$', r'$1$'])
    ax.legend(frameon=False, loc='best', fontsize=13)
    _fmt_axes(ax)
    _save(fig, outdir, 'fa_floquet_rho_t')


def plot_occupation_t(df, outdir, Omega):
    x = df['t'].to_numpy() / (2 * np.pi / Omega)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, df['n_tot'], '--', color='black', lw=1.0, zorder=2,
            label=r'$\text{Total}= \uparrow+ \downarrow$')
    ax.plot(x, df['n_up'], '-', color='red', lw=1.5, zorder=3, label=r'$\uparrow$')
    ax.plot(x, df['n_dn'], '-', color='blue', lw=1.5, zorder=3, label=r'$\downarrow$')
    ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
    ax.set_ylabel(r'$n(t)$')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0, 1.3)
    ax.legend(frameon=False, loc='best', fontsize=13)
    _fmt_axes(ax)
    _save(fig, outdir, 'fa_floquet_occ_t')


def plot_spin_t(df, outdir, Omega):
    x = df['t'].to_numpy() / (2 * np.pi / Omega)
    fig, ax = plt.subplots(figsize=(5, 4))
    for c in ('sx', 'sy', 'sz'):
        ax.plot(x, df[c], '-', color=SPIN_COLORS[c], lw=1.5, zorder=3)
    ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
    ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
    ax.set_ylabel(r'$\langle\hat{\sigma}^{\alpha}\rangle(t)$')
    ax.set_xlim(x.min(), x.max())
    lab = [r'$\alpha=$', r'x,', r'y,', r'z']
    col = ['black', SPIN_COLORS['sx'], SPIN_COLORS['sy'], SPIN_COLORS['sz']]
    ax.legend([Line2D([], [], ls='none') for _ in lab], lab,
              loc='best', frameon=False, ncol=len(lab), handlelength=0.0,
              handletextpad=0.0, columnspacing=0.45, labelcolor=col)
    _fmt_axes(ax)
    _save(fig, outdir, 'fa_floquet_spin_t')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ldos-csv', default=DEFAULT_LDOS_CSV)
    ap.add_argument('--rho-csv', default=DEFAULT_RHO_CSV)
    ap.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'output'))
    ap.add_argument('--wmax', type=float, default=None,
                    help='recorta el eje de omega (|omega|<=wmax) en los plots en frecuencia')
    ap.add_argument('--Omega', type=float, default=0.005,
                    help='frecuencia de driving usada en el barrido (para el eje t/T)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_ldos = pd.read_csv(args.ldos_csv)
    df_rho = pd.read_csv(args.rho_csv)

    plot_ldos(df_ldos, args.outdir, wmax=args.wmax)
    plot_Gr_pm1(df_ldos, args.outdir, wmax=args.wmax)
    plot_rho_t(df_rho, args.outdir, Omega=args.Omega)
    plot_occupation_t(df_rho, args.outdir, Omega=args.Omega)
    plot_spin_t(df_rho, args.outdir, Omega=args.Omega)


if __name__ == '__main__':
    main()
