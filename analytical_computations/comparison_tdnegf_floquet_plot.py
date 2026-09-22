#!/usr/bin/env python3
"""
Superpone el analitico de Floquet con TDNEGF en los ultimos N periodos.

Analitico -> linea SOLIDA, TDNEGF -> linea DASHED, mismo color y mismos ejes,
asi que donde coinciden se ve una sola curva con el trazo discontinuo encima.

ALINEACION DE FASE: los dos barridos arrancan en tiempos absolutos distintos
(el analitico cubre [0, 3T], TDNEGF llega hasta 20T), pero el estado
estacionario de Floquet es periodico, asi que basta restar a cada uno un numero
ENTERO de periodos para que las fases del driving coincidan. Si se restara un
numero fraccionario las curvas saldrian desfasadas aunque la fisica fuera la
misma. El corte se toma en t_max - N*T y el origen en floor(t_corte/T).

El estilo (rcParams, ejes, colores, notacion cientifica, leyenda) se reutiliza
de inbedding_leads_plot para que las figuras salgan identicas a las demas.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# El import trae los rcParams ya configurados (es efecto de nivel de modulo).
from inbedding_leads_plot import (          # noqa: E402
    _fmt_axes, _sci_yaxis, _site_colors, _pick_sites, _top_legend, SPIN_COLORS,
)

DEFAULT_FLOQUET_CSV = os.path.join(SCRIPT_DIR, 'output', 'inbedding_rho_t.csv')

# Gris para las entradas de leyenda que solo indican el estilo de linea: el
# color ya esta gastado en el sitio (o en la componente de espin).
STYLE_COLOR = '0.25'


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def _window(df, lead, Omega, nper, etiqueta):
    """
    Recorta al lead pedido y a los ultimos nper periodos, y define la columna
    'x' = tiempo en periodos medido desde un origen que es multiplo entero de T.
    """
    T = 2 * np.pi / Omega
    d = df[df['lead'] == lead]
    if d.empty:
        raise SystemExit(f'no hay datos para el lead {lead} en {etiqueta}')

    t_cut = d['t'].max() - nper * T
    d = d[d['t'] >= t_cut].copy()
    if d.empty:
        raise SystemExit(f'no quedan datos con t >= {t_cut} en {etiqueta}')

    # El origen se ancla al PRIMER tiempo que queda, no a t_cut: t_cut sale de
    # t_max - nper*T y t_max no cae exacto sobre un multiplo de T (TDNEGF para
    # en 25132.1, no en 20T=25132.74), asi que su floor se iba un periodo.
    # Cualquier entero preserva la fase del driving; este ademas deja x>=0.
    n0 = np.floor(d['t'].min() / T)
    d['x'] = d['t'] / T - n0

    print(f'  {etiqueta:9s}  t = [{d["t"].min():.1f}, {d["t"].max():.1f}]  '
          f'-> origen {n0:.0f}T, x = [{d["x"].min():.3f}, {d["x"].max():.3f}]  '
          f'({d["t"].nunique()} puntos)')
    return d


def _curva(d, site, col):
    s = d[d['site'] == site].sort_values('x')
    return s['x'].to_numpy(), s[col].to_numpy()


def _every(n, nper, por_periodo):
    """
    Paso de markevery para dejar ~por_periodo marcadores en cada periodo.

    TDNEGF escribe ~630 puntos por periodo: dibujados todos, los circulos se
    solapan y tapan por completo la curva analitica de abajo.
    """
    return max(1, int(round(n / max(nper, 1e-9) / por_periodo)))


def plot_par(ax, da, dt, site, col, color, nper, por_periodo):
    """Analitico como linea solida, TDNEGF como circulos encima."""
    x, y = _curva(da, site, col)
    ax.plot(x, y, '-', color=color, lw=1.8, zorder=3)
    x, y = _curva(dt, site, col)
    ax.plot(x, y, 'o', color=color, ms=4.0, mec='white', mew=0.5,
            ls='none', markevery=_every(len(x), nper, por_periodo), zorder=6)


def _legend_estilos():
    return [Line2D([], [], color=STYLE_COLOR, lw=2.0, ls='-', label='Floquet'),
            Line2D([], [], color=STYLE_COLOR, ls='none', marker='o', ms=7.0,
                   mec='white', mew=0.5, label='TDNEGF')]


def plot_rho_cmp(da, dt, outdir, lead, sites, nper, por_periodo):
    """Panel 2x2: las cuatro componentes de rho, un color por sitio."""
    colors = _site_colors(sites)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    cols = [('n_up',        r'$\rho^{\uparrow\uparrow}_{\text{ii}}(t)$'),
            ('n_dn',        r'$\rho^{\downarrow\downarrow}_{\text{ii}}(t)$'),
            ('Re_rho_updn', r'$\text{Re}\,\rho^{\uparrow\downarrow}_{\text{ii}}(t)$'),
            ('Im_rho_updn', r'$\text{Im}\,\rho^{\uparrow\downarrow}_{\text{ii}}(t)$')]

    for ax, (col, ylab) in zip(axes.flat, cols):
        for n in sites:
            plot_par(ax, da, dt, n, col, colors[n], nper, por_periodo)
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, nper)
        _fmt_axes(ax)
        _sci_yaxis(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')

    _top_legend(fig, [Line2D([], [], color=c, lw=2.0,
                             label=rf'$\text{{i}}={n + 1}$')
                      for n, c in colors.items()] + _legend_estilos())
    _save(fig, outdir, f'cmp_rho_t_{lead}')


def plot_spin_cmp(da, dt, outdir, lead, sites, nper, por_periodo):
    """Panel 2x2: un sitio por panel, las tres componentes de espin."""
    sites = sites[:4]
    comps = [('sx', SPIN_COLORS['sx'], r'$\langle\hat{\sigma}^{\text{x}}_{\text{i}}\rangle$'),
             ('sy', SPIN_COLORS['sy'], r'$\langle\hat{\sigma}^{\text{y}}_{\text{i}}\rangle$'),
             ('sz', SPIN_COLORS['sz'], r'$\langle\hat{\sigma}^{\text{z}}_{\text{i}}\rangle$')]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    for ax, n in zip(axes.flat, sites):
        for col, color, _ in comps:
            plot_par(ax, da, dt, n, col, color, nper, por_periodo)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        ax.set_title(rf'$\text{{i}}={n + 1}$', fontsize=18)
        ax.set_xlim(0.0, nper)
        _fmt_axes(ax)
        _sci_yaxis(ax)

    for ax in axes.flat[len(sites):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\langle\hat{\sigma}^{\alpha}_{\text{i}}\rangle(t)$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')

    _top_legend(fig, [Line2D([], [], color=c, lw=2.0, label=lab)
                      for _, c, lab in comps] + _legend_estilos())
    _save(fig, outdir, f'cmp_spin_t_{lead}')


def reporte_discrepancia(da, dt, sites):
    """
    Discrepancia maxima entre las dos curvas, interpolando TDNEGF a la malla
    temporal del analitico (los dos barridos no muestrean los mismos tiempos).
    """
    obs = ['n_up', 'n_dn', 'Re_rho_updn', 'Im_rho_updn', 'sx', 'sy', 'sz', 'n_tot']
    print(f'\n  discrepancia maxima |Floquet - TDNEGF| (y relativa a la '
          f'amplitud del analitico)')
    print(f'  {"sitio":>6}  ' + '  '.join(f'{o:>13}' for o in obs))
    for n in sites:
        fila = []
        for col in obs:
            xa, ya = _curva(da, n, col)
            xt, yt = _curva(dt, n, col)
            yi = np.interp(xa, xt, yt)
            dmax = np.abs(ya - yi).max()
            esc = np.abs(ya).max()
            fila.append(f'{dmax:.2e}' + (f'/{100*dmax/esc:4.1f}%' if esc > 0 else ''))
        print(f'  {n + 1:>6}  ' + '  '.join(f'{c:>13}' for c in fila))


def main():
    ap = argparse.ArgumentParser(
        description='Superpone el analitico de Floquet y TDNEGF en los '
                    'ultimos N periodos (solido vs dashed).')
    ap.add_argument('--floquet-csv', default=DEFAULT_FLOQUET_CSV,
                    help='CSV del analitico (inbedding_rho_t.csv)')
    ap.add_argument('--tdnegf-csv', required=True,
                    help='CSV de TDNEGF (tdnegf_rho_t.csv)')
    ap.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'output'))
    ap.add_argument('--lead', default='R')
    ap.add_argument('--Omega', type=float, default=0.005)
    ap.add_argument('--periods', type=float, default=3.0,
                    help='cuantos periodos finales superponer (por defecto 3)')
    ap.add_argument('--markers-per-period', type=int, default=12, metavar='N',
                    help='cuantos circulos de TDNEGF dibujar por periodo. Con '
                         'todos los puntos (~630/periodo) los marcadores tapan '
                         'la curva analitica.')
    ap.add_argument('--sites', default=None,
                    help='sitios a graficar, separados por coma (0-based). Por '
                         'defecto 4 con espaciado geometrico, de entre los que '
                         'existen en AMBOS CSV.')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    da = _window(pd.read_csv(args.floquet_csv), args.lead, args.Omega,
                 args.periods, 'Floquet')
    dt = _window(pd.read_csv(args.tdnegf_csv), args.lead, args.Omega,
                 args.periods, 'TDNEGF')

    comunes = sorted(set(da['site'].unique()) & set(dt['site'].unique()))
    if not comunes:
        raise SystemExit('los dos CSV no comparten ningun sitio')
    sites = ([int(s) for s in args.sites.split(',')] if args.sites is not None
             else _pick_sites(comunes))
    faltan = [s for s in sites if s not in comunes]
    if faltan:
        raise SystemExit(f'sitios no presentes en ambos CSV: {faltan}')
    print(f'  sitios: {sites}  ->  en las figuras: {[s + 1 for s in sites]}')

    plot_rho_cmp(da, dt, args.outdir, args.lead, sites, args.periods,
                 args.markers_per_period)
    plot_spin_cmp(da, dt, args.outdir, args.lead, sites, args.periods,
                  args.markers_per_period)
    reporte_discrepancia(da, dt, sites)


if __name__ == '__main__':
    main()
