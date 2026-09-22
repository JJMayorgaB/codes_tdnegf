#!/usr/bin/env python3
"""
plot_time_kernel.py -- paneles de chi^{mu nu}_{ij}(t,t') calculados por
time_kernel.jl.

Para cada (mu,nu) salen DOS figuras de n x n paneles (n = numero de sitios):
    time_kernel_{mu}{nu}_{sitios}_Re.{jpg,svg,pdf}   Re chi
    time_kernel_{mu}{nu}_{sitios}_Im.{jpg,svg,pdf}   Im chi
Panel (fila j, columna i), como en
    (i=1,j=1) | (i=2,j=1) | ...
    (i=1,j=2) | (i=2,j=2) | ...
Eje x = t, eje y = t' (en periodos del drive). Cada panel lleva su propia barra
de color, seismic centrada en cero.

Uso:
    python plot_time_kernel.py --mu x --nu y --sites 1,2,3,4
    python plot_time_kernel.py --all --sites 1,7,13,20
    python plot_time_kernel.py --mu x --nu y --sites 1,2,3,4 f11.npy f21.npy ... (16)
Los 16 archivos explicitos van en el orden de lectura de la figura:
(i=1,j=1), (i=2,j=1), ..., (i=4,j=1), (i=1,j=2), ...
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# importarlo tambien fija los rcParams comunes a todas las figuras
from inbedding_leads_plot import _fmt_axes, _sci_cbar    # noqa: E402

DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'output', 'time_kernel')
COMPS = ('x', 'y', 'z')
PANEL = (5.0, 4.0)                  # tamano de cada panel individual


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def _load_meta(indir, tag):
    path = os.path.join(indir, f'time_kernel_{tag}_meta.json')
    if not os.path.isfile(path):
        raise SystemExit(f'no encuentro {path} (lo escribe time_kernel.jl)')
    with open(path) as f:
        return json.load(f)


def _paths(indir, mu, nu, labels, tag):
    """Orden de lectura de la figura: j por filas, i por columnas."""
    return [os.path.join(indir, f'time_kernel_{mu}{nu}_{tag}_i{i}_j{j}.npy')
            for j in labels for i in labels]


def _titulo(parte, mu, nu):
    return (r'$\text{' + parte + r'}\,\chi^{\text{' + mu + nu +
            r'}}_{\text{ij}}(t,t^{\prime})$')


def plot_parte(data, t, labels, mu, nu, parte, outdir, tag):
    """Una figura n x n para Re o Im."""
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(PANEL[0] * n, PANEL[1] * n),
                             squeeze=False)
    f = np.real if parte == 'Re' else np.imag

    for r, j in enumerate(labels):
        for c, i in enumerate(labels):
            ax = axes[r, c]
            Z = f(data[(i, j)])                  # Z[a,b] = chi(t_a, t'_b)
            vmax = np.abs(Z).max()
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            # pcolormesh espera C[y,x]: y = t' (indice b), x = t (indice a)
            im = ax.pcolormesh(t, t, Z.T, shading='nearest', cmap='seismic',  vmin=-vmax, vmax=vmax, rasterized=True)
            ax.set_title(r'$(\text{i},\text{j})=(' + f'{i},{j}' + r')$',  fontsize=18, pad=6, loc='left')
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(t[0], t[-1])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
            _fmt_axes(ax)
            ax.set_axisbelow(False)              # ticks por encima del mapa
            _sci_cbar(fig.colorbar(im, ax=ax, pad=0.025, fraction=0.046))
            if r == n - 1:
                ax.set_xlabel(r'$\text{Time}\ t\ (2\pi/\Omega)$')
            if c == 0:
                ax.set_ylabel(r'$\text{Time}\ t^{\prime}\ (2\pi/\Omega)$')

    fig.suptitle(_titulo(parte, mu, nu), fontsize=26, y=1.0)
    _save(fig, outdir, f'time_kernel_{mu}{nu}_{tag}_{parte}')


def plot_mn(mu, nu, labels, indir, outdir, files=None):
    tag = '-'.join(str(s) for s in labels)
    meta = _load_meta(indir, tag)
    T = 2 * np.pi / meta['Omega']
    t = np.arange(meta['Nt']) * meta['dt'] / T         # en periodos

    paths = files if files else _paths(indir, mu, nu, labels, tag)
    if len(paths) != len(labels) ** 2:
        raise SystemExit(f'hacen falta {len(labels) ** 2} componentes, llegaron {len(paths)}')
    faltan = [q for q in paths if not os.path.isfile(q)]
    if faltan:
        raise SystemExit('faltan archivos:\n  ' + '\n  '.join(faltan))

    keys = [(i, j) for j in labels for i in labels]
    data = {k: np.load(q) for k, q in zip(keys, paths)}

    print(f'  chi^{mu}{nu}   max|Re|   max|Im|')
    for (i, j), Z in data.items():
        print(f'    ({i},{j})   {np.abs(Z.real).max():.3e}  {np.abs(Z.imag).max():.3e}')

    for parte in ('Re', 'Im'):
        plot_parte(data, t, labels, mu, nu, parte, outdir, tag)


def main():
    ap = argparse.ArgumentParser(description='Paneles de chi^{mu nu}_{ij}(t,t\').')
    ap.add_argument('files', nargs='*',
                    help='las n^2 componentes en orden de lectura de la figura '
                         '(opcional; por defecto se arman desde --indir)')
    ap.add_argument('--mu', choices=COMPS)
    ap.add_argument('--nu', choices=COMPS)
    ap.add_argument('--all', action='store_true',
                    help='las 9 combinaciones (mu,nu) que existan en --indir')
    ap.add_argument('--sites', default='1,2,3,4')
    ap.add_argument('--indir', default=DEFAULT_DIR)
    ap.add_argument('--outdir', default=None, help='por defecto = --indir')
    args = ap.parse_args()

    labels = [int(s) for s in args.sites.split(',')]
    outdir = args.outdir or args.indir
    os.makedirs(outdir, exist_ok=True)

    if args.all:
        if args.files:
            raise SystemExit('--all no admite archivos explicitos')
        tag = '-'.join(str(s) for s in labels)
        hechos = 0
        for mu in COMPS:
            for nu in COMPS:
                if all(os.path.isfile(q) for q in _paths(args.indir, mu, nu, labels, tag)):
                    plot_mn(mu, nu, labels, args.indir, outdir)
                    hechos += 1
                else:
                    print(f'  (sin datos completos para {mu}{nu}, lo salto)')
        print(f'{hechos} combinaciones (mu,nu) graficadas')
    else:
        if args.mu is None or args.nu is None:
            raise SystemExit('hace falta --mu y --nu, o --all')
        plot_mn(args.mu, args.nu, labels, args.indir, outdir, args.files or None)


if __name__ == '__main__':
    main()
