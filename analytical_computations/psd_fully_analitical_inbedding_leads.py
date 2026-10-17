#!/usr/bin/env python3
"""
psd_fully_analitical_inbedding_leads.py -- espectros de potencia (PSD = |FT|^2) de
los resultados completamente analiticos (fully_analitical_inbedding_leads.jl),
lead derecho, solo componente z, en dos figuras: (i) la densidad de espin en los
sitios 1 y 2 (mismo panel) y (ii) la corriente de espin del enlace 1 -> 2.

Las series analiticas son EXACTAMENTE T-periodicas (estado estacionario de
Floquet, solo armonicos k = 0, +-1). Para que los espectros sean comparables con
los de TDNEGF (psd_setup.py, ventana de 10 periodos) se toma un periodo del CSV y
se repite --periods veces (por defecto 10): misma resolucion Omega/10 y misma
ventana de Hann. Misma normalizacion:
    P(omega) = | sum_n w_n x_n e^{i omega t_n} |^2 / (sum_n w_n)^2
Lineas punteadas verticales en omega = Omega y 2 Omega.

Lee output/fa_inbedding_rho_t.csv, fa_inbedding_bond_rho_t.csv y
fa_inbedding_bond_H.csv y escribe en output/ (jpg, svg, pdf):
    fa_inbedding_psd_sz_R_{log,lin}        PSD de <sigma^z>_1 y <sigma^z>_2
    fa_inbedding_psd_bond_Isz_R_{log,lin}  PSD de I^{S_z}_{1->2}

Uso:
    python psd_fully_analitical_inbedding_leads.py
    python psd_fully_analitical_inbedding_leads.py --periods 10 --wmax 3 --window hann
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inbedding_leads_plot as P                                  # noqa: E402  (rcParams)
from inbedding_leads_plot import (_fmt_axes, _sci_yaxis, _legend_en_hueco,  # noqa: E402
                                  SPIN_COLORS, bond_currents)
from matplotlib.lines import Line2D                               # noqa: E402

COL_Z = SPIN_COLORS['sz']


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def psd(t, x, Omega, window, pad, wmax):
    """PSD normalizada de x(t) (malla uniforme) en omega/Omega <= wmax."""
    dt = t[1] - t[0]
    w = np.hanning(len(x)) if window == 'hann' else np.ones(len(x))
    nfft = pad * len(x)
    F = np.fft.rfft(w * x, n=nfft)
    Pw = np.abs(F) ** 2 / w.sum() ** 2
    om = 2 * np.pi * np.fft.rfftfreq(nfft, dt) / Omega
    sel = om <= wmax
    return om[sel], Pw[sel]


def periodica(t, x, T, nper):
    """Un periodo exacto [t0, t0+T) de la malla, repetido nper veces."""
    dt = t[1] - t[0]
    n1 = int(round(T / dt))
    if abs(n1 * dt - T) > 1e-6 * T or len(t) < n1:
        raise SystemExit('la malla temporal del CSV no contiene un periodo entero')
    xx = np.tile(x[:n1], nper)
    return t[0] + dt * np.arange(len(xx)), xx


def main():
    ap = argparse.ArgumentParser(description='PSD de sigma^z (sitios 1, 2) e I^{S_z} (enlace 1 -> 2), analitico.')
    ap.add_argument('--csv-dir', default=os.path.join(HERE, 'output'))
    ap.add_argument('--outdir', default=os.path.join(HERE, 'output'))
    ap.add_argument('--Omega', type=float, default=0.005)
    ap.add_argument('--periods', type=int, default=10, help='periodos de la serie (se repite uno)')
    ap.add_argument('--wmax', type=float, default=3.0)
    ap.add_argument('--window', choices=['hann', 'rect'], default='hann')
    ap.add_argument('--pad', type=int, default=8, help='factor de relleno con ceros (solo dibujo)')
    args = ap.parse_args()

    T = 2 * np.pi / args.Omega
    kw = dict(Omega=args.Omega, window=args.window, pad=args.pad, wmax=args.wmax)

    # densidad de espin sigma^z en los sitios 1 y 2 (sitios 0 y 1 del CSV)
    dr = pd.read_csv(os.path.join(args.csv_dir, 'fa_inbedding_rho_t.csv'))
    dr = dr[dr['lead'] == 'R']
    curvas = []
    for s, ls in ((0, '-'), (1, '--')):
        d = dr[dr['site'] == s].sort_values('t')
        om, Pw = psd(*periodica(d['t'].to_numpy(), d['sz'].to_numpy(), T, args.periods), **kw)
        curvas.append((om, Pw, ls, rf'$\langle\hat{{\sigma}}^{{z}}_{{{s + 1}}}\rangle$'))

    # corriente de espin I^{S_z} del enlace 1 -> 2
    dfb = pd.read_csv(os.path.join(args.csv_dir, 'fa_inbedding_bond_rho_t.csv'))
    dfh = pd.read_csv(os.path.join(args.csv_dir, 'fa_inbedding_bond_H.csv'))
    t, _, IS, im = bond_currents(dfb, dfh, 'R', 0, 1)
    print(f'  enlace 1 -> 2: max|Im|/max|Re| = {im:.1e}')
    om, Pw = psd(*periodica(t, IS[:, 2], T, args.periods), **kw)
    corriente = [(om, Pw, '-', None)]

    # dos figuras: densidades de espin (sitios 1 y 2) y corriente de espin del enlace
    figs = ((curvas, r'PSD $\langle\hat{\sigma}^{z}_{\text{i}}\rangle$', None,
             'fa_inbedding_psd_sz_R'),
            (corriente, r'PSD $\text{I}^{\text{S}_{z}}_{\text{i}\rightarrow\text{j}}$',
             r'$\text{i}=1\rightarrow \text{j}=2$', 'fa_inbedding_psd_bond_Isz_R'))
    for cur, ylab, tit, nombre in figs:
        pmax = max(c[1].max() for c in cur)
        for esc in ('log', 'lin'):
            fig, ax = plt.subplots(figsize=(5, 4))
            for om, Pw, ls, _ in cur:
                ax.plot(om, Pw, ls, color=COL_Z, lw=1.6, zorder=3)
            for n in (1, 2):
                ax.axvline(n, color='0.5', ls=':', lw=1.2, zorder=1)
            if esc == 'log':
                ax.set_yscale('log')
                ax.set_ylim(pmax * 1e-14, pmax * 10)
            else:
                ax.set_ylim(0.0, 1.08 * pmax)
            ax.set_xlim(0.0, args.wmax)
            ax.set_xlabel(r'$\omega/\Omega$')
            ax.set_ylabel(ylab)
            if tit is not None:
                ax.set_title(tit, fontsize=16)
            _fmt_axes(ax)
            if esc == 'lin':
                _sci_yaxis(ax)
            if len(cur) > 1:
                _legend_en_hueco(ax, [Line2D([], [], color=COL_Z, lw=1.8, ls=ls) for _, _, ls, _ in cur],
                                 [lab for *_, lab in cur], ncol=1, fontsize=14, handlelength=1.8)
            _save(fig, args.outdir, f'{nombre}_{esc}')


if __name__ == '__main__':
    main()
