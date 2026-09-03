#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'axes.axisbelow': True,
})

C_R = plt.cm.seismic(0.85)   # lead derecho: tono cálido
C_L = plt.cm.seismic(0.15)   # lead izquierdo: tono frío
C_KPOS = plt.cm.seismic(0.85)  # k>0
C_KNEG = plt.cm.seismic(0.15)  # k<0

#(ver oscillators.jl)
T_ON_G3 = 250.0   # arranca la precesión uniforme del grupo3
T_ON_G1 = 1000.0  # arranca la onda viajera del grupo1
OMEGA = 0.5       # frecuencia de driving (ver oscillators.jl)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'output')

ROWS = [
    ('I',   'I_L',   'I_R',   r'$I$'),
    ('Isx', 'Isx_L', 'Isx_R', r'$I^S_x$'),
    ('Isy', 'Isy_L', 'Isy_R', r'$I^S_y$'),
    ('Isz', 'Isz_L', 'Isz_R', r'$I^S_z$'),
]


def _fmt_axes(ax, xlabel=None, ylabel=None):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                    left=True, right=True, size=6.0, width=1.0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _milestones(ax):
    ax.axvline(T_ON_G3, color='k', lw=1.0, ls=':', alpha=0.8)
    ax.axvline(T_ON_G1, color='k', lw=1.0, ls='--', alpha=0.8)


def load_run(data_dir, name):
    path = os.path.join(data_dir, f'oscillators_trace_{name}.csv')
    df = pd.read_csv(path)
    print(f'  [{name}] leído {path}  ({len(df)} filas)')
    return df


def _delta_label(ylab):
    # r'$I^S_x$' -> r'$\Delta I^S_x$'
    return r'$\Delta ' + ylab.strip('$') + '$'


def _window(df, tmin):
    return df if tmin is None else df[df['t'] >= tmin]


def plot_currents(dpos, dneg, outdir, tmin=None, stem='currents_kpos_vs_kneg'):
    dpos, dneg = _window(dpos, tmin), _window(dneg, tmin)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        ax.plot(dpos['t'], dpos[colL], '-',  color=C_L, lw=1.5, label=r'$L$, $k{>}0$')
        ax.plot(dpos['t'], dpos[colR], '-',  color=C_R, lw=1.5, label=r'$R$, $k{>}0$')
        ax.plot(dneg['t'], dneg[colL], '--', color=C_L, lw=1.5, label=r'$L$, $k{<}0$')
        ax.plot(dneg['t'], dneg[colR], '--', color=C_R, lw=1.5, label=r'$R$, $k{<}0$')
        if tmin is None:
            _milestones(ax)
        ax.set_xlim(dpos['t'].min(), dpos['t'].max())
        _fmt_axes(ax,
                   xlabel=(r'$t\ (\hbar/\gamma)$' if row_i == len(ROWS) - 1 else None),
                   ylabel=ylab)
        if row_i == 0:
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                       loc='best', borderpad=0.5, handlelength=2.0, labelspacing=0.3,
                       ncol=2)

    plt.tight_layout()
    for ext in ('jpg', 'svg'):
        fig.savefig(os.path.join(outdir, f'{stem}.{ext}'), bbox_inches='tight', dpi=300)
    plt.show()
    print(f'  → {stem}.jpg / .svg')


def plot_deltas(dpos, dneg, outdir, tmin=None, stem='delta_I_kpos_vs_kneg'):
    """ΔI = I_R - I_L: bombeo neto hacia el lead derecho, kpos vs kneg."""
    dpos, dneg = _window(dpos, tmin), _window(dneg, tmin)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        ax.axhline(0.0, color='0.6', lw=0.9, ls='-')
        ax.plot(dpos['t'], dpos[colR] - dpos[colL], '-',
                color=C_KPOS, lw=1.5, label=r'$k{>}0$')
        ax.plot(dneg['t'], dneg[colR] - dneg[colL], '--',
                color=C_KNEG, lw=1.5, label=r'$k{<}0$')
        if tmin is None:
            _milestones(ax)
        ax.set_xlim(dpos['t'].min(), dpos['t'].max())
        _fmt_axes(ax,
                   xlabel=(r'$t\ (\hbar/\gamma)$' if row_i == len(ROWS) - 1 else None),
                   ylabel=_delta_label(ylab))
        if row_i == 0:
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                       loc='best', borderpad=0.5, handlelength=2.0, labelspacing=0.3,
                       ncol=2)

    plt.tight_layout()
    for ext in ('jpg', 'svg'):
        fig.savefig(os.path.join(outdir, f'{stem}.{ext}'), bbox_inches='tight', dpi=300)
    plt.show()
    print(f'  → {stem}.jpg / .svg')


def _fft_amp(t, x, t_min):
    """Espectro de amplitud (ventana de Hann) de x(t) para t>=t_min."""
    sel = t >= t_min
    tt = np.asarray(t[sel]); xx = np.asarray(x[sel])
    n = len(xx)
    dt = np.mean(np.diff(tt))
    win = np.hanning(n)
    xw = (xx - xx.mean()) * win
    spec = np.fft.rfft(xw)
    omega = 2.0 * np.pi * np.fft.rfftfreq(n, d=dt)
    amp = np.abs(spec) / max(win.sum(), 1e-12)
    return omega, amp


FFT_LABELS = {
    'I':   r'$|I(\omega)|^2$',
    'Isx': r'$|I^S_{x}(\omega)|^2$',
    'Isy': r'$|I^S_{y}(\omega)|^2$',
    'Isz': r'$|I^S_{z}(\omega)|^2$',
}


def plot_fourier(dpos, dneg, outdir):
    """FFT de las corrientes (L y R) para t>=T_ON_G1, kpos vs kneg."""
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        #ax.set_yscale('log')
        for d, ls, ktag in [(dpos, '-', r'$k{>}0$'), (dneg, '--', r'$k{<}0$')]:
            wL, aL = _fft_amp(d['t'], d[colL], T_ON_G1)
            wR, aR = _fft_amp(d['t'], d[colR], T_ON_G1)
            ax.plot(wL / OMEGA, aL**2, ls, color=C_L, lw=1.3, label=fr'$L$, {ktag}')
            ax.plot(wR / OMEGA, aR**2, ls, color=C_R, lw=1.3, label=fr'$R$, {ktag}')
        for h in range(1, 5):
            ax.axvline(h, color='k', lw=0.8, ls=':', alpha=0.5)
        ax.set_xlim(0.0, 4.5)
        _fmt_axes(ax,
                   xlabel=(r'$\omega/\Omega$' if row_i == len(ROWS) - 1 else None),
                   ylabel=FFT_LABELS[tag])
        if row_i == 0:
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                       loc='best', borderpad=0.5, handlelength=2.0, labelspacing=0.3,
                       ncol=2, fontsize=11)

    plt.tight_layout()
    for ext in ('jpg', 'svg'):
        fig.savefig(os.path.join(outdir, f'fourier_currents_kpos_vs_kneg.{ext}'),
                    bbox_inches='tight', dpi=300)
    plt.show()
    print('  → fourier_currents_kpos_vs_kneg.jpg / .svg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=DEFAULT_DIR)
    ap.add_argument('--outdir', default=DEFAULT_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dpos = load_run(args.data_dir, 'kpos')
    dneg = load_run(args.data_dir, 'kneg')

    plot_currents(dpos, dneg, args.outdir)
    plot_deltas(dpos, dneg, args.outdir)
    plot_currents(dpos, dneg, args.outdir, tmin=T_ON_G1,
                  stem='currents_kpos_vs_kneg_zoom')
    plot_deltas(dpos, dneg, args.outdir, tmin=T_ON_G1,
                stem='delta_I_kpos_vs_kneg_zoom')
    plot_fourier(dpos, dneg, args.outdir)


if __name__ == '__main__':
    main()
