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
DEFAULT_LDOS_CSV = os.path.join(SCRIPT_DIR, 'output', 'floquet_ldos.csv')
DEFAULT_RHO_CSV = os.path.join(SCRIPT_DIR, 'output', 'floquet_rho_t.csv')


def _fmt_axes(ax):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=5.0, width=1.0)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if isinstance(fmt, mticker.ScalarFormatter):
            fmt.set_useMathText(False)


def plot_ldos(df, outdir, wmax=None):
    w = df['omega'].to_numpy()
    up, dn, tot = (df['LDOS_up'].to_numpy(), df['LDOS_dn'].to_numpy(),
                   df['LDOS_tot'].to_numpy())

    if wmax is not None:
        sel = np.abs(w) <= wmax
        w, up, dn, tot = w[sel], up[sel], dn[sel], tot[sel]

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(w, tot, '--', color='black', lw=1.0, zorder=2,
            label=r'$\text{Total}= \uparrow+ \downarrow$')
    ax.plot(w, up, '-', color='red', lw=1.5, zorder=3,
            label=r'$\uparrow$')
    ax.plot(w, dn, '-', color='blue', lw=1.5, zorder=3,
            label=r'$\downarrow$')

    ax.set_xlabel(r'$\omega\ (\gamma)$')
    ax.set_ylabel(r'$A(\omega)\ (1/\gamma)$')
    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(-0.1, 10.0)
    ax.legend(frameon=False, loc='best')
    _fmt_axes(ax)

    plt.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'floquet_ldos.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def plot_rho_t(df, outdir, Omega):
    T = 2 * np.pi / Omega
    x = df['t'].to_numpy() / T   # fraccion de periodo, 0 a 1

    n_up = df['n_up'].to_numpy()
    n_dn = df['n_dn'].to_numpy()
    re_c = df['Re_rho_updn'].to_numpy()
    im_c = df['Im_rho_updn'].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(x, n_up, '-', color='red', lw=1.5, zorder=3,
            label=r'$\rho_{\uparrow\uparrow}$')
    ax.plot(x, n_dn, '-', color='blue', lw=1.5, zorder=3,
            label=r'$\rho_{\downarrow\downarrow}$')
    ax.plot(x, re_c, '--', color='0.3', lw=1.3, zorder=2,
            label=r'$\text{Re}\,\rho_{\uparrow\downarrow}$')
    ax.plot(x, im_c, ':', color='0.3', lw=1.5, zorder=2,
            label=r'$\text{Im}\,\rho_{\uparrow\downarrow}$')

    ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
    ax.set_ylabel(r'$\rho(t)$')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(top=1.1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([r'$0$', r'$0.5$', r'$1$'])
    ax.legend(frameon=False, loc='best', fontsize=13)
    _fmt_axes(ax)

    plt.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'floquet_rho_t.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def plot_occupation_t(df, outdir, Omega):
    T = 2 * np.pi / Omega
    x = df['t'].to_numpy() / T   # fraccion de periodo

    n_up = df['n_up'].to_numpy()
    n_dn = df['n_dn'].to_numpy()
    n_tot = df['n_tot'].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(x, n_tot, '--', color='black', lw=1.0, zorder=2,
            label=r'$\text{Total}= \uparrow+ \downarrow$')
    ax.plot(x, n_up, '-', color='red', lw=1.5, zorder=3,
            label=r'$\uparrow$')
    ax.plot(x, n_dn, '-', color='blue', lw=1.5, zorder=3,
            label=r'$\downarrow$')

    ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
    ax.set_ylabel(r'$n(t)$')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0, 1.3)
    ax.legend(frameon=False, loc='best', fontsize=13)
    _fmt_axes(ax)

    plt.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'floquet_occ_t.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def plot_spin_t(df, outdir, Omega):
    T = 2 * np.pi / Omega
    x = df['t'].to_numpy() / T   # fraccion de periodo, 0 a 1

    sx, sy, sz = (df['sx'].to_numpy(), df['sy'].to_numpy(), df['sz'].to_numpy())

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(x, sx, '-', color='red', lw=1.5, zorder=3,  label=r'$\langle\sigma_x\rangle$')
    ax.plot(x, sy, '-', color='blue', lw=1.5, zorder=3,  label=r'$\langle\sigma_y\rangle$')
    ax.plot(x, sz, '-', color='black', lw=1.5, zorder=3, label=r'$\langle\sigma_z\rangle$')
    ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)

    ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
    ax.set_ylabel(r'$\langle\boldsymbol{\sigma}\rangle(t)$')
    ax.set_xlim(x.min(), x.max())
    ax.legend(frameon=False, loc='best')
    _fmt_axes(ax)

    plt.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'floquet_spin_t.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ldos-csv', default=DEFAULT_LDOS_CSV)
    ap.add_argument('--rho-csv', default=DEFAULT_RHO_CSV)
    ap.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'output'))
    ap.add_argument('--wmax', type=float, default=None,
                    help='recorta el eje de omega (|omega|<=wmax) en el plot de LDOS')
    ap.add_argument('--Omega', type=float, default=0.005,
                    help='frecuencia de driving usada en el barrido (para el eje t/T)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df_ldos = pd.read_csv(args.ldos_csv)
    df_rho = pd.read_csv(args.rho_csv)

    plot_ldos(df_ldos, args.outdir, wmax=args.wmax)
    plot_rho_t(df_rho, args.outdir, Omega=args.Omega)
    plot_occupation_t(df_rho, args.outdir, Omega=args.Omega)
    plot_spin_t(df_rho, args.outdir, Omega=args.Omega)


if __name__ == '__main__':
    main()
