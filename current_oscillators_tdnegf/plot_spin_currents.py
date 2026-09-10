#!/usr/bin/env python3

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import run_config as rc

HAS_LATEX = shutil.which('latex') is not None

plt.rcParams.update({
    'text.usetex': HAS_LATEX,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'] if HAS_LATEX else ['DejaVu Serif'],
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
    'axes.formatter.use_mathtext': False,
})

C_R = plt.cm.seismic(0.85)   # lead derecho: tono calido
C_L = plt.cm.seismic(0.15)   # lead izquierdo: tono frio

# Se sobreescriben en main() con los valores reales de la corrida
T_ON_G3 = 5000.0
OMEGA = 0.01

ROWS = [
    ('I',   'I_L',   'I_R',   r'$I$'),
    ('Isx', 'Isx_L', 'Isx_R', r'$I^S_x$'),
    ('Isy', 'Isy_L', 'Isy_R', r'$I^S_y$'),
    ('Isz', 'Isz_L', 'Isz_R', r'$I^S_z$'),
]

FFT_LABELS = {
    'I':   r'$|I(\omega)|^2$',
    'Isx': r'$|I^S_{x}(\omega)|^2$',
    'Isy': r'$|I^S_{y}(\omega)|^2$',
    'Isz': r'$|I^S_{z}(\omega)|^2$',
}


def load_trace(path):
    cur = [c for _, cL, cR, _ in ROWS for c in (cL, cR)]
    dtypes = {'t': 'float64', **{c: 'float32' for c in cur}}
    df = pd.read_csv(path, usecols=['t'] + cur, dtype=dtypes)
    print(f'  leido {path}  ({len(df)} filas)')
    return df


def _save(fig, outdir, stem):
    for ext in ('jpg', 'svg'):
        fig.savefig(os.path.join(outdir, f'{stem}.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  -> {stem}.jpg / .svg')


def _fmt_axes(ax, xlabel=None, ylabel=None):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=6.0, width=1.0)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if isinstance(fmt, mticker.ScalarFormatter):
            fmt.set_useMathText(False)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _window(df, tmin):
    return df if tmin is None else df[df['t'] >= tmin]


def _delta_label(ylab):
    return r'$\Delta ' + ylab.strip('$') + '$'


def plot_currents(df, outdir, stem, title=None, tmin=None):
    d = _window(df, tmin)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        ax.plot(d['t'], d[colL], '-', color=C_L, lw=1.5, label=r'$L$')
        ax.plot(d['t'], d[colR], '-', color=C_R, lw=1.5, label=r'$R$')
        if tmin is None:
            ax.axvline(T_ON_G3, color='k', lw=1.0, ls=':', alpha=0.8)
        ax.set_xlim(d['t'].min(), d['t'].max())
        _fmt_axes(ax,
                  xlabel=(r'$t\ (\hbar/\gamma)$' if row_i == len(ROWS) - 1 else None),
                  ylabel=ylab)
        if row_i == 0:
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      loc='best', borderpad=0.5, handlelength=2.0, labelspacing=0.3,
                      ncol=2)
    plt.tight_layout()
    _save(fig, outdir, stem)


def plot_deltas(df, outdir, stem, title=None, tmin=None):
    """DeltaI = I_R - I_L: corriente neta que atraviesa el dispositivo."""
    d = _window(df, tmin)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        ax.axhline(0.0, color='0.6', lw=0.9, ls='-')
        ax.plot(d['t'], d[colR] - d[colL], '-', color=C_R, lw=1.5)
        if tmin is None:
            ax.axvline(T_ON_G3, color='k', lw=1.0, ls=':', alpha=0.8)
        ax.set_xlim(d['t'].min(), d['t'].max())
        _fmt_axes(ax,
                  xlabel=(r'$t\ (\hbar/\gamma)$' if row_i == len(ROWS) - 1 else None),
                  ylabel=_delta_label(ylab))
    plt.tight_layout()
    _save(fig, outdir, stem)


def _fft_amp(t, x, t_min):
    """Espectro de amplitud (ventana de Hann) de x(t) para t>=t_min."""
    sel = t >= t_min
    tt = np.asarray(t[sel]); xx = np.asarray(x[sel])
    n = len(xx)
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    dt = np.mean(np.diff(tt))
    win = np.hanning(n)
    xw = (xx - xx.mean()) * win
    spec = np.fft.rfft(xw)
    omega = 2.0 * np.pi * np.fft.rfftfreq(n, d=dt)
    amp = np.abs(spec) / max(win.sum(), 1e-12)
    return omega, amp


def plot_fourier(df, outdir, stem, t_from, title=None):
    """FFT de las corrientes para t >= t_on_g3."""
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title)
    t = df['t'].to_numpy()
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        wL, aL = _fft_amp(t, df[colL].to_numpy(), t_from)
        wR, aR = _fft_amp(t, df[colR].to_numpy(), t_from)
        ax.plot(wL / OMEGA, aL**2, '-', color=C_L, lw=1.3, label=r'$L$')
        ax.plot(wR / OMEGA, aR**2, '-', color=C_R, lw=1.3, label=r'$R$')
        for h in range(1, 5):
            ax.axvline(h, color='k', lw=0.8, ls=':', alpha=0.35, zorder=0)
        ax.set_xlim(0.0, 4.5)
        _fmt_axes(ax,
                  xlabel=(r'$\omega/\Omega$' if row_i == len(ROWS) - 1 else None),
                  ylabel=FFT_LABELS[tag])
        if row_i == 0:
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      loc='best', borderpad=0.5, handlelength=2.0, labelspacing=0.3,
                      ncol=2)
    plt.tight_layout()
    _save(fig, outdir, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-tag', default=None,
                    help='subcarpeta de output/. Por defecto la que corresponde a '
                         'los parametros actuales de oscillators.jl.')
    ap.add_argument('--trace', default='prep_trace.csv',
                    help='nombre del CSV dentro de la carpeta, o ruta completa')
    ap.add_argument('--outdir', default=None, help='override directo')
    args = ap.parse_args()

    run_dir = rc.resolve_run_dir(args.run_tag)
    csv = args.trace if os.path.isabs(args.trace) else os.path.join(run_dir, args.trace)
    outdir = args.outdir or os.path.join(run_dir, 'figs')
    os.makedirs(outdir, exist_ok=True)

    global T_ON_G3, OMEGA
    prot = rc.protocol(run_dir)
    T_ON_G3, OMEGA = prot['t_on_g3'], prot['Omega']
    t_from = T_ON_G3 + (prot['t_rise'] or 0.0)
    print(f'run_tag = {os.path.basename(run_dir)}\n'
          f't_on_g3 = {T_ON_G3}   Omega = {OMEGA}\n'
          f'ventana de zoom/FFT desde t = {t_from}\noutdir  = {outdir}')

    df = load_trace(csv)
    stem = os.path.splitext(os.path.basename(csv))[0]

    titulo = r'Steady state preparation ($g_3$ driver only)'
    plot_currents(df, outdir, f'{stem}_currents', titulo)
    plot_deltas(df, outdir, f'{stem}_delta_I', titulo)
    plot_currents(df, outdir, f'{stem}_currents_zoom', titulo, tmin=t_from)
    plot_deltas(df, outdir, f'{stem}_delta_I_zoom', titulo, tmin=t_from)
    plot_fourier(df, outdir, f'{stem}_fourier', t_from, titulo)


if __name__ == '__main__':
    main()
