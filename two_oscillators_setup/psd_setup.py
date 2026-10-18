#!/usr/bin/env python3
"""
psd_setup.py -- espectros de potencia (PSD = |FT|^2) de una corrida de los setups
de dos osciladores (TDNEGF con LMMs), en una ventana de tiempo ya estacionaria.

Figuras (jpg, svg, pdf) en <corrida>/figures/ (carpeta que ya existe), cada una
en dos versiones, eje y logaritmico (_log) y lineal (_lin):
    psd_spin_density_<a>-<b>T_<esc>   1x3: PSD de <sigma^alpha>_i en los sitios de D, F, F
    psd_lmm_<a>-<b>T_<esc>            1x3: PSD de M^alpha de los tres LMMs (D, F, F)
    psd_lead_currents_<a>-<b>T_<esc>  1x2: PSD de las corrientes de espin en el lead L y en el R
    psd_bond_<i>-<j>_<a>-<b>T_<esc>   1x1: PSD de las corrientes de espin del enlace i -> j (F -> F),
                                      solo si los F son primeros vecinos (DFF/F1-2)
Cada panel mide (5, 4). Lineas punteadas verticales en omega = Omega y 2 Omega.
En la escala lineal de las dos figuras 1x3, la componente y (con su pico DC
grande) va en un eje derecho propio, en verde, para no aplastar a x y z.

PSD:  P(omega) = | sum_n w_n x_n e^{i omega t_n} |^2 / (sum_n w_n)^2
con ventana de Hann w_n (o rectangular con --window rect). Con esta normalizacion
una senal A cos(omega0 t) da un pico de altura (A/2)^2 en omega0 y una constante
c da c^2 en omega = 0. La ventana de tiempo [a, b) en periodos del driver
(por defecto [10, 20)) tiene un numero entero de periodos, asi que los armonicos
n Omega caen sobre la malla de frecuencias; resolucion Omega / (b - a). El
relleno con ceros (--pad) solo suaviza el dibujo de los picos.

Uso:
    python psd_setup.py y_precession/output/DFF/F1-2/gso0p1_jsd0p2_th10deg_Om0p005_buf4_adyn0p0
    python psd_setup.py <corrida> --t-start 10 --t-end 20 --wmax 3 --window hann
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analytical_computations'))
# importarlo fija los rcParams comunes a todas las figuras
import inbedding_leads_plot as P                                          # noqa: E402
from inbedding_leads_plot import (_fmt_axes, _sci_yaxis, _alpha_legend,  # noqa: E402
                                  SPIN_COLORS, bond_currents)

COMP = (('x', SPIN_COLORS['sx']), ('y', SPIN_COLORS['sy']), ('z', SPIN_COLORS['sz']))
WLAB = r'$\omega/\Omega$'
ESCALAS = ('log', 'lin')


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def leer_corrida(run):
    geo = pd.read_csv(os.path.join(run, 'geometry.csv'))
    txt = open(os.path.join(run, 'params.txt'), encoding='utf-8').read()
    Omega = float(re.search(r'Ω = ([0-9.eE+-]+)', txt).group(1))
    return geo, Omega


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


def ventana(df, T, a, b):
    """Filas con a T <= t < b T (intervalo semiabierto: numero entero de periodos)."""
    d = df[(df['t'] >= a * T) & (df['t'] < b * T)].sort_values('t')
    if d.empty:
        raise SystemExit(f'no hay datos en [{a}T, {b}T)')
    return d


def _ylim(ax, pmax, escala):
    if escala == 'log':
        ax.set_yscale('log')
        if pmax > 0:
            ax.set_ylim(pmax * 1e-14, pmax * 10)
    else:
        ax.set_ylim(0.0, 1.08 * pmax if pmax > 0 else 1.0)


def panel_psd(ax, curvas, escala, wmax, twin=False):
    """
    curvas: lista de (omega, P, color). Punteadas en Omega y 2 Omega.
    Con twin=True y escala lineal, la componente y (verde) va en un segundo eje
    a la derecha, en verde; x y z quedan en el eje izquierdo. Devuelve ese eje
    derecho (o None).
    """
    cy = SPIN_COLORS['sy']
    usar_twin = twin and escala == 'lin'
    izq = [c for c in curvas if not (usar_twin and c[2] == cy)]
    der = [c for c in curvas if usar_twin and c[2] == cy]

    for om, Pw, col in izq:
        ax.plot(om, Pw, '-', color=col, lw=1.5, zorder=3)
    for n in (1, 2):
        ax.axvline(n, color='0.5', ls=':', lw=1.2, zorder=1)
    _ylim(ax, max(Pw.max() for _, Pw, _ in izq), escala)
    ax.set_xlim(0.0, wmax)
    _fmt_axes(ax)
    if escala == 'lin':
        _sci_yaxis(ax)
    if not der:
        return None

    ax.tick_params(axis='y', right=False)          # el lado derecho es del eje verde
    ax2 = ax.twinx()
    for om, Pw, col in der:
        ax2.plot(om, Pw, '-', color=col, lw=1.5, zorder=3)
    _ylim(ax2, max(Pw.max() for _, Pw, _ in der), escala)
    ax2.tick_params(axis='y', direction='in', right=True, left=False, size=5.0, width=1.0,
                    colors=cy)
    ax2.spines['right'].set_color(cy)
    _sci_yaxis(ax2)
    ax2.yaxis.get_offset_text().set_color(cy)
    return ax2


def _unificar_y(axs, mostrar, lado):
    """Mismos limites en y para todos los ejes; numeros y multiplicador solo en axs[mostrar]."""
    lims = [a.get_ylim() for a in axs]
    lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
    for k, a in enumerate(axs):
        a.set_ylim(lo, hi)
        if k != mostrar:
            a.tick_params(axis='y', **{f'label{lado}': False})
            a.yaxis.get_offset_text().set_visible(False)


def figura_paneles(paneles, titulos, ylabels, escala, wmax, figdir, nombre,
                   fila=False, twin=False, ylabel_der=None):
    """
    Paneles de (5, 4) cada uno: en fila (1 x n) o en columna (n x 1), con la
    leyenda alpha en el primero. Con twin=True (solo en la escala lineal) la
    componente y usa el eje derecho, en verde, con etiqueta ylabel_der en el
    ultimo panel.
    """
    n = len(paneles)
    if fila:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    else:
        fig, axes = plt.subplots(n, 1, figsize=(5, 4 * n), sharex=True, squeeze=False)
    axes = axes.ravel()
    ders = []
    for ax, curvas, tit, yl in zip(axes, paneles, titulos, ylabels):
        ders.append(panel_psd(ax, curvas, escala, wmax, twin=twin))
        ax.set_title(tit, fontsize=18 if n > 1 else 16)
        if not fila or ax is axes[0]:
            ax.set_ylabel(yl)
    for ax in (axes if fila else axes[-1:]):
        ax.set_xlabel(WLAB)
    if fila and n > 1:
        # misma escala en toda la fila: numeros del eje izquierdo solo en el primer
        # panel y los del eje derecho (verde) solo en el ultimo
        _unificar_y(list(axes), mostrar=0, lado='left')
        tw = [d for d in ders if d is not None]
        if len(tw) > 1:
            _unificar_y(tw, mostrar=len(tw) - 1, lado='right')
    if ders[-1] is not None and ylabel_der is not None:
        ders[-1].set_ylabel(ylabel_der, color=SPIN_COLORS['sy'])
    _alpha_legend(axes[0])
    _save(fig, figdir, f'{nombre}_{escala}')


def main():
    ap = argparse.ArgumentParser(description='PSD (|FT|^2) de una corrida de los setups.')
    ap.add_argument('run')
    ap.add_argument('--t-start', type=float, default=10.0, help='inicio de la ventana (periodos)')
    ap.add_argument('--t-end', type=float, default=20.0, help='fin de la ventana (periodos)')
    ap.add_argument('--wmax', type=float, default=3.0, help='omega/Omega maximo')
    ap.add_argument('--window', choices=['hann', 'rect'], default='hann')
    ap.add_argument('--pad', type=int, default=8, help='factor de relleno con ceros (solo dibujo)')
    args = ap.parse_args()

    run = os.path.abspath(args.run)
    geo, Omega = leer_corrida(run)
    T = 2 * np.pi / Omega
    a, b = args.t_start, args.t_end
    figdir = os.path.join(run, 'figures')
    os.makedirs(figdir, exist_ok=True)
    tag = f'{a:g}-{b:g}T'
    kw = dict(Omega=Omega, window=args.window, pad=args.pad, wmax=args.wmax)
    roles, sites = list(geo['role']), [int(s) for s in geo['site']]
    tit_sitio = [rf'$\text{{i}}={s}\ (\text{{{r}}})$' for s, r in zip(sites, roles)]
    print(f'{run}\n  ventana [{a:g}T, {b:g}T), {args.window}, resolucion {1 / (b - a):.3f} Omega')

    # 1. densidades de espin en los sitios de los LMMs
    ds = pd.read_csv(os.path.join(run, 'sites_rho_t.csv'))
    pan1 = []
    for s in sites:
        d = ventana(ds[ds['site'] == s], T, a, b)
        t = d['t'].to_numpy()
        pan1.append([(*psd(t, d[f's{c}'].to_numpy(), **kw), col) for c, col in COMP])

    # 2. trayectorias de los LMMs
    dm = ventana(pd.read_csv(os.path.join(run, 'spins_t.csv')), T, a, b)
    t = dm['t'].to_numpy()
    pan2 = [[(*psd(t, dm[f'M{m}_{c}'].to_numpy(), **kw), col) for c, col in COMP]
            for m in range(1, len(sites) + 1)]

    # 3. corrientes de espin en los leads
    dl = ventana(pd.read_csv(os.path.join(run, 'lead_currents_t.csv')), T, a, b)
    t = dl['t'].to_numpy()
    pan3 = [[(*psd(t, dl[f'Is{c}_{lead}'].to_numpy(), **kw), col) for c, col in COMP]
            for lead in ('L', 'R')]

    # 4. corriente de espin del enlace F -> F (solo si los F son primeros vecinos)
    fr = [s for s, r in zip(sites, roles) if r == 'F']
    i, j = fr[0], fr[1]
    pan4 = None
    if j - i == 1:
        dfb = pd.read_csv(os.path.join(run, 'bond_rho_t.csv'))
        dfb = dfb[(dfb['n'] == i) & (dfb['m'] == j)]
        tb, _, IS, im = bond_currents(dfb, pd.read_csv(os.path.join(run, 'bond_H.csv')), 'C', i, j)
        sel = (tb >= a * T) & (tb < b * T)
        print(f'  enlace {i} -> {j}: max|Im|/max|Re| = {im:.1e}')
        pan4 = [[(*psd(tb[sel], IS[sel, c], **kw), col) for c, (_, col) in enumerate(COMP)]]
    else:
        print(f'  AVISO: los F (sitios {i}, {j}) no son primeros vecinos: se omite psd_bond')

    for esc in ESCALAS:
        figura_paneles(pan1, tit_sitio, [r'PSD $\langle\hat{\sigma}^{\alpha}_{\text{i}}\rangle$'] * 3,
                       esc, args.wmax, figdir, f'psd_spin_density_{tag}', fila=True, twin=True,
                       ylabel_der=r'PSD $\langle\hat{\sigma}^{y}_{\text{i}}\rangle$')
        figura_paneles(pan2, tit_sitio, [r'PSD $M^{\alpha}_{\text{i}}$'] * 3,
                       esc, args.wmax, figdir, f'psd_lmm_{tag}', fila=True, twin=True,
                       ylabel_der=r'PSD $M^{y}_{\text{i}}$')
        figura_paneles(pan3, [r'$\text{lead}\ L$', r'$\text{lead}\ R$'],
                       [r'PSD $\text{I}^{\text{S}_{\alpha}}$'] * 2,
                       esc, args.wmax, figdir, f'psd_lead_currents_{tag}', fila=True)
        if pan4 is not None:
            figura_paneles(pan4, [rf'$\text{{i}}={i}\ (\text{{F}})\rightarrow \text{{j}}={j}\ (\text{{F}})$'],
                           [r'PSD $\text{I}^{\text{S}_{\alpha}}_{\text{i}\rightarrow\text{j}}$'],
                           esc, args.wmax, figdir, f'psd_bond_{i}-{j}_{tag}')


if __name__ == '__main__':
    main()
