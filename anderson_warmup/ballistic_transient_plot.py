#!/usr/bin/env python3
"""Figura I_R(t) del alambre balistico, con el plateau de Landauer anotado.

Lee ballistic_transient.csv (generado por ballistic_transient.jl) y produce la
figura con los ejes en unidades fisicas: tiempo en fs, corriente en 2e*gamma/h.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
DEFAULT_CSV = os.path.join(SCRIPT_DIR, 'output', 'ballistic_transient.csv')


def _fmt_axes(ax):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=5.0, width=1.0)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if isinstance(fmt, mticker.ScalarFormatter):
            fmt.set_useMathText(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'output'))
    ap.add_argument('--tmax-fs', type=float, default=None,
                    help='recorta el eje temporal (fs) para hacer zoom al transitorio')
    ap.add_argument('--channels', type=int, default=4,
                    help='numero de canales abiertos T (para la etiqueta de la flecha)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    t = df['t_fs'].to_numpy()
    I = df['I_R_2egh'].to_numpy()

    if args.tmax_fs is not None:
        sel = t <= args.tmax_fs
        t, I = t[sel], I[sel]

    # plateau: promedio del ultimo 20% de la serie completa
    i0 = max(1, int(0.8 * len(df)))
    plateau = float(np.mean(df['I_R_2egh'].to_numpy()[i0:]))

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    ax.plot(t, I, '-', color='red', lw=1.5, zorder=3)
    ax.axhline(plateau, color='0.5', ls='--', lw=1.0, zorder=2)

    # Flecha apuntando al plateau
    x_arrow = t[-1] * 0.62
    y_text = plateau * 0.5
    label = (r'$I_R(t\!\to\!\infty)/V_b = ' + str(args.channels) + r'e^2/h$')
    ax.annotate(label,
                xy=(x_arrow, plateau),
                xytext=(x_arrow, y_text),
                ha='center', va='top', fontsize=20,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black',
                                shrinkA=0, shrinkB=2))

    ax.set_xlabel(r'Time (fs)')
    ax.set_ylabel(r'$I_R\ (2e\gamma/h)$')
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0.0, plateau * 1.25)
    ax.minorticks_on()
    _fmt_axes(ax)

    plt.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(args.outdir, f'ballistic_transient.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)

    print(f'\nplateau = {plateau:.6f}  [2e*gamma/h]')


if __name__ == '__main__':
    main()
