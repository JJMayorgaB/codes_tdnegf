#!/usr/bin/env python3
"""
comparison_tdnegf_fully_analitical_inbedding.py -- superpone la dinamica
completamente analitica (fully_analitical_inbedding_leads.jl, linea solida) con
TDNEGF (tdnegf_single_spin.jl con PREC_AXIS=y, circulos) en el lead derecho,
sitios 1 y 2: componentes de rho, densidades de espin y corrientes del enlace 1 -> 2.

Las dos dinamicas se alinean en fase igual que en comparison_tdnegf_floquet_plot.py:
se toman los ultimos N periodos de cada una y se mide el tiempo desde un multiplo
entero de T (TDNEGF arranca el driver en t_on = 5T, asi que la fase del driver es
la misma que la del analitico, phi = Omega t).

Figuras (jpg, svg, pdf) en --outdir:
    cmp_rho_t_R, cmp_spin_t_R, cmp_current_t_R, cmp_spin_current_t_R
y en pantalla la discrepancia maxima (absoluta y relativa a la amplitud analitica).

Uso:
    python comparison_tdnegf_fully_analitical_inbedding.py
    python comparison_tdnegf_fully_analitical_inbedding.py --tdnegf-csv <corrida>/tdnegf_rho_t.csv --periods 3
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import comparison_tdnegf_floquet_plot as C                      # noqa: E402
from inbedding_leads_plot import SPIN_COLORS                    # noqa: E402

TAG_Y = 'single_gso0p1_jsd0p2_th10deg_Om0p005_buf20_precy'
DEFAULT_FA = os.path.join(SCRIPT_DIR, 'output', 'fa_inbedding_rho_t.csv')
DEFAULT_TD = os.path.join(SCRIPT_DIR, 'test_tdnegf', 'output', TAG_Y, 'tdnegf_rho_t.csv')
DEFAULT_OUT = os.path.join(SCRIPT_DIR, 'output', 'cmp_fa_tdnegf_y')


# estilo: el de comparison_tdnegf_floquet_plot.py (analitico solido, TDNEGF circulos)
def plot_spin_sites(da, dt, outdir, lead, sites, nper, por_periodo):
    """Panel 1 x len(sites): un sitio por panel, las tres componentes de espin."""
    fig, axes = plt.subplots(1, len(sites), figsize=(5 * len(sites), 4))
    for ax, n in zip(axes, sites):
        for col in ('sx', 'sy', 'sz'):
            C.plot_par(ax, da, dt, n, col, SPIN_COLORS[col], nper, por_periodo)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        ax.set_title(rf'$\text{{i}}={n + 1}$', fontsize=18)
        ax.set_xlim(0.0, nper)
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
        C._fmt_axes(ax)
        C._sci_yaxis(ax)
    axes[0].set_ylabel(r'$\langle\hat{\sigma}^{\alpha}_{\text{i}}\rangle(t)$')
    C._legend_doble_en_hueco(
        axes[0],
        ([Line2D([], [], ls='none') for _ in C._LAB_A], C._LAB_A, C._KW_A),
        (C._handles_estilo(), ['Floquet', 'TDNEGF'], C._KW_E))
    C._save(fig, outdir, f'cmp_spin_t_{lead}')


def main():
    ap = argparse.ArgumentParser(description='Analitico (solido) vs TDNEGF (punteado), precesion en y.')
    ap.add_argument('--fa-csv', default=DEFAULT_FA, help='fa_inbedding_rho_t.csv del analitico')
    ap.add_argument('--tdnegf-csv', default=DEFAULT_TD, help='tdnegf_rho_t.csv de la corrida PREC_AXIS=y')
    ap.add_argument('--outdir', default=DEFAULT_OUT)
    ap.add_argument('--lead', default='R')
    ap.add_argument('--Omega', type=float, default=0.005)
    ap.add_argument('--periods', type=float, default=3.0, help='periodos finales a superponer')
    ap.add_argument('--markers-per-period', type=int, default=12, metavar='N',
                    help='circulos de TDNEGF por periodo (con todos los puntos tapan la curva analitica)')
    args = ap.parse_args()
    mpp = args.markers_per_period
    os.makedirs(args.outdir, exist_ok=True)
    for f in (args.fa_csv, args.tdnegf_csv):
        if not os.path.exists(f):
            raise SystemExit(f'no existe {f}')

    da = C._window(pd.read_csv(args.fa_csv), args.lead, args.Omega, args.periods, 'analitico')
    dt = C._window(pd.read_csv(args.tdnegf_csv), args.lead, args.Omega, args.periods, 'TDNEGF')
    sites = [s for s in (0, 1) if s in set(da['site']) and s in set(dt['site'])]
    if not sites:
        raise SystemExit('los dos CSV no comparten los sitios 0 y 1 (sitios 1 y 2 del lead)')
    print(f'  sitios: {sites}  ->  en las figuras: {[s + 1 for s in sites]}')

    C.plot_rho_cmp(da, dt, args.outdir, args.lead, sites, args.periods, mpp)
    plot_spin_sites(da, dt, args.outdir, args.lead, sites, args.periods, mpp)
    C.reporte_discrepancia(da, dt, sites)

    ca = C._df_corrientes(args.fa_csv, 'fa_inbedding', args.lead, 0, 1, 'analitico')
    ct = C._df_corrientes(args.tdnegf_csv, 'tdnegf', args.lead, 0, 1, 'TDNEGF')
    if ca is not None and ct is not None:
        ca = C._window(ca, args.lead, args.Omega, args.periods, 'analitico')
        ct = C._window(ct, args.lead, args.Omega, args.periods, 'TDNEGF')
        C.plot_current_cmp(ca, ct, args.outdir, args.lead, args.periods, mpp)
        C.plot_spin_current_cmp(da, dt, ca, ct, args.outdir, args.lead, 0, args.periods, mpp)


if __name__ == '__main__':
    main()
