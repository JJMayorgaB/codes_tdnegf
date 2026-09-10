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
    'axes.formatter.use_mathtext': False,  # evita \mathdefault{} en ticks (rompe con usetex)
})

C_R = plt.cm.seismic(0.85)   # lead derecho: tono cálido
C_L = plt.cm.seismic(0.15)   # lead izquierdo: tono frío
C_KPOS = plt.cm.seismic(0.85)  # k>0
C_KNEG = plt.cm.seismic(0.15)  # k<0

# Se sobreescriben en main() con los valores reales de la corrida
# (params.txt de la carpeta, o oscillators.jl como respaldo).
T_ON_G3 = 500.0   # arranca la precesión uniforme del grupo3
T_ON_G1 = 2000.0  # arranca la onda viajera del grupo1
OMEGA = 0.05      # frecuencia de driving

ROWS = [
    ('I',   'I_L',   'I_R',   r'$I$'),
    ('Isx', 'Isx_L', 'Isx_R', r'$I^S_x$'),
    ('Isy', 'Isy_L', 'Isy_R', r'$I^S_y$'),
    ('Isz', 'Isz_L', 'Isz_R', r'$I^S_z$'),
]


def _fmt_axes(ax, xlabel=None, ylabel=None):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                    left=True, right=True, size=6.0, width=1.0)
    # con usetex=True, ScalarFormatter fuerza \mathdefault{} en los ticks
    # (macro de mathtext, no existe en LaTeX real) -> lo desactivamos.
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if isinstance(fmt, mticker.ScalarFormatter):
            fmt.set_useMathText(False)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _milestones(ax):
    # T_ON_G1 es None en el modo prep (g1 nunca se enciende): se omite la linea.
    if T_ON_G3 is not None:
        ax.axvline(T_ON_G3, color='k', lw=1.0, ls=':', alpha=0.8)
    if T_ON_G1 is not None:
        ax.axvline(T_ON_G1, color='k', lw=1.0, ls='--', alpha=0.8)


R_VALUES = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)


def r_tag(r):
    return 'r' + str(r).replace('.', 'p')


def load_csv(path, label=''):
    cur = [c for _, cL, cR, _ in ROWS for c in (cL, cR)]
    dtypes = {'t': 'float64', **{c: 'float32' for c in cur}}
    df = pd.read_csv(path, usecols=['t'] + cur, dtype=dtypes)
    print(f'  [{label}] leído {path}  ({len(df)} filas)')
    return df


def load_run(data_dir, name):
    return load_csv(os.path.join(data_dir, f'oscillators_trace_{name}.csv'), name)


def plot_single(df, outdir, stem, title=None, tmin=None, delta=False):
    """Una sola trayectoria (sin comparacion kpos/kneg): usado para el prep."""
    d = _window(df, tmin)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        if delta:
            ax.axhline(0.0, color='0.6', lw=0.9, ls='-')
            ax.plot(d['t'], d[colR] - d[colL], '-', color=C_R, lw=1.5)
        else:
            ax.plot(d['t'], d[colL], '-', color=C_L, lw=1.5, label=r'$L$')
            ax.plot(d['t'], d[colR], '-', color=C_R, lw=1.5, label=r'$R$')
        if tmin is None:
            _milestones(ax)
        ax.set_xlim(d['t'].min(), d['t'].max())
        _fmt_axes(ax,
                  xlabel=(r'$t\ (\hbar/\gamma)$' if row_i == len(ROWS) - 1 else None),
                  ylabel=_delta_label(ylab) if delta else ylab)
        if row_i == 0 and not delta:
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                      loc='best', borderpad=0.5, handlelength=2.0, labelspacing=0.3,
                      ncol=2)
    plt.tight_layout()
    _save(fig, outdir, stem)


def plot_single_fourier(df, outdir, stem, t_from, title=None):
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


def _save(fig, outdir, stem):
    for ext in ('jpg', 'svg'):
        fig.savefig(os.path.join(outdir, f'{stem}.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  → {stem}.jpg / .svg')


def _delta_label(ylab):
    # r'$I^S_x$' -> r'$\Delta I^S_x$'
    return r'$\Delta ' + ylab.strip('$') + '$'


def _window(df, tmin):
    return df if tmin is None else df[df['t'] >= tmin]


def plot_currents(dpos, dneg, outdir, stem, title=None, tmin=None):
    dpos, dneg = _window(dpos, tmin), _window(dneg, tmin)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        ax.plot(dpos['t'], dpos[colL], '-',  color=C_L, lw=1.0, label=r'$L$, $k{>}0$')
        ax.plot(dpos['t'], dpos[colR], '-',  color=C_R, lw=1.0, label=r'$R$, $k{>}0$')
        ax.plot(dneg['t'], dneg[colL], '--', color=C_L, lw=1.0, label=r'$L$, $k{<}0$')
        ax.plot(dneg['t'], dneg[colR], '--', color=C_R, lw=1.0, label=r'$R$, $k{<}0$')
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
    _save(fig, outdir, stem)


def plot_deltas(dpos, dneg, outdir, stem, title=None, tmin=None):
    """ΔI = I_R - I_L: bombeo neto hacia el lead derecho, kpos vs kneg."""
    dpos, dneg = _window(dpos, tmin), _window(dneg, tmin)
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        ax.axhline(0.0, color='0.6', lw=0.9, ls='-')
        ax.plot(dpos['t'], dpos[colR] - dpos[colL], '-',
                color=C_KPOS, lw=1.0, label=r'$k{>}0$')
        ax.plot(dneg['t'], dneg[colR] - dneg[colL], '--',
                color=C_KNEG, lw=1.0, label=r'$k{<}0$')
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
    _save(fig, outdir, stem)


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


def plot_fourier(dpos, dneg, outdir, stem, title=None, t_from=0.0):
    """FFT de las corrientes (L y R) para t >= t_from, kpos vs kneg."""
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title)
    for row_i, (tag, colL, colR, ylab) in enumerate(ROWS):
        ax = axes[row_i]
        for d, ls, ktag in [(dpos, '-', r'$k{>}0$'), (dneg, '--', r'$k{<}0$')]:
            wL, aL = _fft_amp(d['t'], d[colL], t_from)
            wR, aR = _fft_amp(d['t'], d[colR], t_from)
            ax.plot(wL / OMEGA, aL**2, ls, color=C_L, lw=1.0, label=fr'$L$, {ktag}')
            ax.plot(wR / OMEGA, aR**2, ls, color=C_R, lw=1.0, label=fr'$R$, {ktag}')
        for h in range(1, 5):
            ax.axvline(h, color='k', lw=0.8, ls=':', alpha=0.35, zorder=0)
        ax.set_xlim(0.0, 4.5)
        _fmt_axes(ax,
                   xlabel=(r'$\omega/\Omega$' if row_i == len(ROWS) - 1 else None),
                   ylabel=FFT_LABELS[tag])
        if row_i == 0:
            ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                       loc='best', borderpad=0.5, handlelength=2.0, labelspacing=0.3,
                       ncol=2, fontsize=14)

    plt.tight_layout()
    _save(fig, outdir, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-tag', default=None,
                     help='subcarpeta de output/. Por defecto se usa la que '
                          'corresponde a los parámetros actuales de oscillators.jl.')
    ap.add_argument('--data-dir', default=None, help='override directo (ignora --run-tag)')
    ap.add_argument('--outdir', default=None, help='override directo (ignora --run-tag)')
    ap.add_argument('--r', type=float, nargs='*', default=list(R_VALUES))
    ap.add_argument('--trace', default=None,
                     help='CSV unico a graficar (p.ej. prep_trace.csv). Con esta '
                          'opcion no hay comparacion kpos/kneg ni barrido en r.')
    args = ap.parse_args()

    prefix = 'steady_state_' if args.trace else 'pumping_'
    run_dir = rc.resolve_run_dir(args.run_tag, prefix=prefix)
    data_dir = args.data_dir or run_dir
    outdir = args.outdir or run_dir
    os.makedirs(outdir, exist_ok=True)

    global T_ON_G3, T_ON_G1, OMEGA
    prot = rc.protocol(data_dir)
    T_ON_G3, T_ON_G1, OMEGA = prot['t_on_g3'], prot['t_on_g1'], prot['Omega']
    t_rise = prot['t_rise'] or 0.0

    # La ventana de la FFT arranca despues de la rampa adiabatica de encendido:
    # incluirla mete un transiente que no es parte de la respuesta periodica.
    t_on = T_ON_G1 if T_ON_G1 is not None else T_ON_G3
    t_from = (t_on or 0.0) + t_rise

    print(f'run_tag = {os.path.basename(run_dir)}\n'
          f't_on_g3 = {T_ON_G3}   t_on_g1 = {T_ON_G1}   Omega = {OMEGA}\n'
          f'ventana de zoom/FFT desde t = {t_from}')

    if args.trace:
        path = args.trace if os.path.isabs(args.trace) else os.path.join(data_dir, args.trace)
        stem = os.path.splitext(os.path.basename(path))[0]
        odir = os.path.join(outdir, 'figs')
        os.makedirs(odir, exist_ok=True)
        df = load_csv(path, stem)
        titulo = r'Steady state preparation ($g_3$ driver only, $g_1$ off)'
        plot_single(df, odir, f'{stem}_currents', titulo)
        plot_single(df, odir, f'{stem}_delta_I', titulo, delta=True)
        plot_single(df, odir, f'{stem}_currents_zoom', titulo, tmin=t_from)
        plot_single(df, odir, f'{stem}_delta_I_zoom', titulo, tmin=t_from, delta=True)
        plot_single_fourier(df, odir, f'{stem}_fourier', t_from, titulo)
        return

    for r in args.r:
        tag = r_tag(r)
        dpos = load_run(data_dir, f'{tag}_kpos')
        dneg = load_run(data_dir, f'{tag}_kneg')
        title = rf'$r=k/\Omega={r}$'
        odir = os.path.join(outdir, tag)
        os.makedirs(odir, exist_ok=True)

        plot_currents(dpos, dneg, odir, 'currents', title)
        plot_deltas(dpos, dneg, odir, 'delta_I', title)
        plot_currents(dpos, dneg, odir, 'currents_zoom', title, tmin=t_from)
        plot_deltas(dpos, dneg, odir, 'delta_I_zoom', title, tmin=t_from)
        plot_fourier(dpos, dneg, odir, 'fourier', title, t_from=t_from)


if __name__ == '__main__':
    main()
