#!/usr/bin/env python3
"""
plot_additional_figures.py -- figura adicional con los elementos de la LLG efectiva
(marco rotante) para i = 1 frente a |i - j|:

    (1) α^S_{i,j} (negro)   (2) α^A_{i,j} (rojo)
    (3) t^{y,A}_{i,j} (negro)

con X^S_{i,j} = (X_{i,j} + X_{j,i})/2,  X^A_{i,j} = (X_{i,j} - X_{j,i})/2.

SOLO figuras (no llama a Julia). Lee output/<tag>/data/llg_elements_sites1-N.csv y
escribe en output/<tag>/figures/additional/llg_alpha_ty_i1.{jpg,...}

Uso:
    python plot_additional_figures.py
    python plot_additional_figures.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005 --sites 10 --formats jpg,svg,pdf
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_kernel_elements as K                       # noqa: E402  (rcParams, FORMATS)
from plot_kernel_elements import P, _fmt_axes          # noqa: E402
from plot_llg_elements import load, _save              # noqa: E402

TAG = 'setup2_gso0p1_jsd0p2_th10deg_Om0p005'


def parts(M, i):
    """X^S_{i,j}, X^A_{i,j} para j = 1..N."""
    row, col = M[i - 1, :], M[:, i - 1]
    return (row + col) / 2, (row - col) / 2


def main():
    ap = argparse.ArgumentParser(description='α^{S,A}_{i,j} y t^{y,A}_{i,j} frente a |i-j|, i = 1.')
    ap.add_argument('--tag', default=TAG, help='carpeta output/<tag>')
    ap.add_argument('--sites', type=int, default=10)
    ap.add_argument('--formats', default='jpg', help='formatos de las figuras, p. ej. jpg,svg,pdf')
    args = ap.parse_args()

    K.FORMATS[:] = [f.strip() for f in args.formats.split(',') if f.strip()]
    N, i = args.sites, 1
    fl = os.path.join(HERE, 'output', args.tag, 'data', f'llg_elements_sites1-{N}.csv')
    K.need(os.path.isfile(fl), fl, 'julia llg_elements.jl (ver run_cluster.sh)')
    X = load(fl, N)

    j = np.arange(1, N + 1)
    x = np.abs(i - j)
    aS, aA = parts(X['α'], i)
    _, tA = parts(X['ty'], i)

    fig, axes = plt.subplots(1, 3, figsize=(5.0 * 3, 4.0))
    for ax, y, c, lab in zip(axes, (aS, aA, tA), ('black', 'red', 'black'),
                             (r'$\alpha^{S}_{i,j}$', r'$\alpha^{A}_{i,j}$', r'$t^{y,A}_{i,j}$')):
        ax.plot(x, y, 'o-', color=c, ms=5, lw=1.3)
        ax.set_ylabel(lab)
    for ax in axes:
        ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
        ax.set_xlabel(r'$|i - j|$')
        ax.set_xticks(x)
        ax.text(1.0, 1.015, rf'$i={i}$', transform=ax.transAxes, ha='right', va='bottom', fontsize=15)
        _fmt_axes(ax)
        P._sci_yaxis(ax)
    _save(fig, os.path.join(HERE, 'output', args.tag, 'figures', 'additional'), f'llg_alpha_ty_i{i}')


if __name__ == '__main__':
    main()
