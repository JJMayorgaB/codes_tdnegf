#!/usr/bin/env python3

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

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.plot(t, I, '-', color='red', lw=1.5, zorder=3)
    ax.axhline(plateau, color='0.5', ls='--', lw=1.0, zorder=2)

    # Flecha apuntando al plateau
    x_arrow = 80 * 0.9
    y_text = plateau * 1.4
    label = (r'$\text{I}_{\text{R}}(\text{t}\!\to\!\infty)/\text{V}_b = ' + str(args.channels) + r'\text{e}^2/\text{h}$')
    ax.annotate(label,
                xy=(x_arrow, plateau*1.025),
                xytext=(x_arrow*0.65, y_text*0.95),
                ha='center', va='top', fontsize=20,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black',
                                shrinkA=0, shrinkB=2))
    ax.set_xlabel(r'Time (fs)')
    ax.set_ylabel(r'$\text{I}_{\text{R}}\ (2\text{e}\gamma/\text{h})$')
    ax.set_xlim(t.min(), 80)
    ax.set_ylim(0.0, 0.03)

    # ticks en 0,1,2,3 con factor de escala x10^-2 aparte
    ax.set_yticks(np.arange(0, 0.031, 0.01))
    ax.set_yticklabels([f'{y:.0f}' for y in range(4)])
    ax.text(-0.16, 1.03, r'$\times 10^{-2}$', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=17)

    ax.set_xticks(np.arange(0, 81, 20))
    ax.set_xticklabels([f'{x:.0f}' for x in np.arange(0, 81, 20)])
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
