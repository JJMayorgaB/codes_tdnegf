#!/usr/bin/env python3
"""
plot_fully_analitical_inbedding_leads.py -- figuras de los resultados puramente
analiticos de fully_analitical_inbedding_leads.jl (lead derecho, sitios 1 y 2,
precesion en y + Rashba en y).

Usa las mismas funciones de figura que inbedding_leads_plot.py (mismo estilo),
leyendo los CSV con prefijo fa_inbedding:
    output/fa_inbedding_ldos.csv, fa_inbedding_rho_t.csv,
    fa_inbedding_bond_rho_t.csv, fa_inbedding_bond_H.csv
y escribe (jpg, svg, pdf) en --outdir:
    fa_inbedding_ldos_R, fa_inbedding_rho_t_R, fa_inbedding_spin_t_R,
    fa_inbedding_occ_t_R, fa_inbedding_current_t_R
(los nombres exactos los pone inbedding_leads_plot.py). Los heatmaps no se
generan: con dos sitios no aportan nada.

Los sitios en los CSV van con la numeracion del codigo (0 = sitio 1, 1 = sitio 2)
y las figuras los etiquetan como i = n+1, igual que antes.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inbedding_leads_plot as P                      # noqa: E402

PREFIX = 'fa_inbedding'
LEAD = 'R'


def main():
    ap = argparse.ArgumentParser(description='Figuras del lead derecho (formulas cerradas).')
    ap.add_argument('--csv-dir', default=os.path.join(HERE, 'output'),
                    help='carpeta con los CSV fa_inbedding_*')
    ap.add_argument('--outdir', default=os.path.join(HERE, 'output'))
    ap.add_argument('--Omega', type=float, default=0.005,
                    help='frecuencia de driving del barrido (para el eje t/T)')
    ap.add_argument('--wmax', type=float, default=None,
                    help='recorta el eje de omega (|omega|<=wmax) en la LDOS')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    P.PREFIX = PREFIX                                   # nombres de archivo de las figuras
    csv = lambda name: os.path.join(args.csv_dir, f'{PREFIX}_{name}.csv')

    dl = pd.read_csv(csv('ldos'));  dl = dl[dl['lead'] == LEAD]
    dr = pd.read_csv(csv('rho_t')); dr = dr[dr['lead'] == LEAD]
    sites = sorted(int(s) for s in dr['site'].unique())
    print(f'  sitios del CSV: {sites}  ->  en las figuras: {[s + 1 for s in sites]}')

    P.plot_ldos(dl, args.outdir, LEAD, wmax=args.wmax)
    P.plot_rho_t(dr, args.outdir, LEAD, Omega=args.Omega)
    P.plot_spin_t(dr, args.outdir, LEAD, Omega=args.Omega)
    P.plot_occupation_t(dr, args.outdir, LEAD, Omega=args.Omega)

    t, I, IS, im = P.bond_currents(pd.read_csv(csv('bond_rho_t')), pd.read_csv(csv('bond_H')),
                                   LEAD, 0, 1)
    print(f'  corrientes del enlace 1 -> 2: max|Im|/max|Re| = {im:.1e}')
    P.plot_current_t(t, I, IS, args.outdir, LEAD, Omega=args.Omega)


if __name__ == '__main__':
    main()
