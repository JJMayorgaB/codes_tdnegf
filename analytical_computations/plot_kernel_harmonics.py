#!/usr/bin/env python3
"""
plot_kernel_harmonics.py -- armonicos de Floquet del kernel, chi^{mu nu}_{ij,k},
calculados por kernel_harmonics.jl.

Para cada (mu,nu) salen DOS figuras n x n en el espacio (i,j), con la misma
estructura que las de plot_time_kernel.py (i por columnas, j por filas):
    kh_tau_{mu}{nu}_{sitios}_Re     Re chi_k(tau)
    kh_omega_{mu}{nu}_{sitios}_Re   Re chi_k(omega),  chi_k(omega) = int_0 dtau e^{i omega tau} chi_k(tau)
En cada panel van los 5 armonicos k = 0, +-1, +-2: mismo color para k y -k,
k en linea solida y -k en linea punteada.

Uso:
    python plot_kernel_harmonics.py --all --sites 1,2,3,4 --indir output\\kernel_harmonics\\data_dt0.5
    python plot_kernel_harmonics.py --mu x --nu x --sites 1,2,3,4
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# importarlo tambien fija los rcParams comunes a todas las figuras
from inbedding_leads_plot import _fmt_axes, _sci_yaxis    # noqa: E402

DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'output', 'kernel_harmonics', 'data')
DEFAULT_FIG = os.path.join(SCRIPT_DIR, 'output', 'kernel_harmonics', 'figures')
COMPS = ('x', 'y', 'z')
PANEL = (5.0, 4.0)
KCOL = {0: 'black', 1: 'tab:blue', 2: 'tab:red'}     # color por |k|
MARKERS_BAJO = 150      # con menos puntos que esto se marcan las muestras


def _save(fig, outdir, name):
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def _leyenda(fig):
    h, l = [], []
    for k in (0, 1, -1, 2, -2):
        h.append(Line2D([], [], color=KCOL[abs(k)], lw=2.0, ls='-' if k >= 0 else '--'))
        l.append(r'$k=' + f'{k}' + r'$')
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=20,
               bbox_to_anchor=(0.5, 0.965), handlelength=2.4, columnspacing=1.6)


def _malla(x, y, labels, xlabel, ylabel, titulo, fname, outdir, muestras):
    """y[k_idx, x, i_idx, j_idx] real (k_idx = k + 2)."""
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(PANEL[0] * n, PANEL[1] * n),
                             squeeze=False, sharex=True, sharey=True)
    mk = dict(marker='o', ms=3.0) if muestras else {}
    for r, j in enumerate(labels):
        for c, i in enumerate(labels):
            ax = axes[r, c]
            for kab in (2, 1, 0):                       # k = 0 encima
                col = KCOL[kab]
                ax.plot(x, y[kab + 2, :, c, r], '-', color=col, lw=1.3, zorder=3 + (2 - kab), **mk)
                if kab:
                    ax.plot(x, y[-kab + 2, :, c, r], '--', color=col, lw=1.3,
                            zorder=3 + (2 - kab), **mk)
            ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
            if r == 0:
                ax.set_title(r'$\text{i}=' + f'{i}' + r'$', fontsize=22, pad=12)
            if c == 0:
                ax.text(-0.3, 0.5, r'$\text{j}=' + f'{j}' + r'$', transform=ax.transAxes,
                        rotation=90, ha='center', va='center', fontsize=22)
                ax.set_ylabel(ylabel)
            if r == n - 1:
                ax.set_xlabel(xlabel)
            ax.set_xlim(x[0], x[-1])
            _fmt_axes(ax)
            ax.label_outer()
    _sci_yaxis(axes[0, 0])
    fig.suptitle(titulo, fontsize=26, y=0.995)
    _leyenda(fig)
    _save(fig, outdir, fname)


def _transformada(tau, X, nw=1201):
    """chi_k(omega) = int_0^taumax dtau e^{i omega tau} chi_k(tau), trapecio,
    en omega in [-pi/dtau, pi/dtau] (frecuencia maxima sin aliasing)."""
    dtau = tau[1] - tau[0]
    w = np.full(len(tau), dtau); w[0] = w[-1] = dtau / 2
    om = np.linspace(-np.pi / dtau, np.pi / dtau, nw)
    P = np.exp(1j * np.outer(om, tau)) * w[None, :]          # (nw, ntau)
    Y = np.einsum('wt,tkij->kwij', P, X)                     # (5, nw, n, n)
    return om, Y


def plot_mn(mu, nu, labels, indir, outdir, tag, meta, tau):
    f = os.path.join(indir, f'kh_{mu}{nu}_{tag}.npy')
    if not os.path.isfile(f):
        print(f'  (no hay {os.path.basename(f)}, lo salto)')
        return False
    A = np.load(f)                                  # (ntau, 5, n, n)
    muestras = len(tau) < MARKERS_BAJO
    comp = r'\text{' + mu + nu + r'}'
    Yt = np.moveaxis(A, 1, 0).real                  # (5, ntau, n, n)
    _malla(tau, Yt, labels, r'$\text{Time}\ \tau$',
           r'$\text{Re}\,\chi_{k}(\tau)$',
           r'$\text{Re}\,\chi^{' + comp + r'}_{\text{ij},k}(\tau)$' +
           rf'$\quad (d\tau={meta["dtau"]:.3g})$',
           f'kh_tau_{mu}{nu}_{tag}_Re', outdir, muestras)
    om, Yw = _transformada(tau, A)
    _malla(om, Yw.real, labels, r'$\text{Frequency}\ \omega$',
           r'$\text{Re}\,\chi_{k}(\omega)$',
           r'$\text{Re}\,\chi^{' + comp + r'}_{\text{ij},k}(\omega)$' +
           rf'$\quad (d\tau={meta["dtau"]:.3g})$',
           f'kh_omega_{mu}{nu}_{tag}_Re', outdir, False)
    return True


def main():
    ap = argparse.ArgumentParser(description='Armonicos chi_k del kernel en tau y en omega.')
    ap.add_argument('--mu', choices=COMPS)
    ap.add_argument('--nu', choices=COMPS)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--sites', default='1,2,3,4')
    ap.add_argument('--indir', default=DEFAULT_DIR)
    ap.add_argument('--outdir', default=DEFAULT_FIG)
    args = ap.parse_args()

    labels = [int(s) for s in args.sites.split(',')]
    tag = '-'.join(str(s) for s in labels)
    indir = args.indir
    if os.path.isdir(os.path.join(indir, tag)):
        indir = os.path.join(indir, tag)
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    print(f'  datos:   {indir}\n  figuras: {outdir}')

    with open(os.path.join(indir, f'kh_{tag}_meta.json')) as fh:
        meta = json.load(fh)
    tau = np.load(os.path.join(indir, f'kh_{tag}_tau.npy'))

    if args.all:
        hechos = sum(plot_mn(mu, nu, labels, indir, outdir, tag, meta, tau)
                     for mu in COMPS for nu in COMPS)
        print(f'{hechos} combinaciones (mu,nu) graficadas')
    else:
        if args.mu is None or args.nu is None:
            raise SystemExit('hace falta --mu y --nu, o --all')
        plot_mn(args.mu, args.nu, labels, indir, outdir, tag, meta, tau)


if __name__ == '__main__':
    main()
