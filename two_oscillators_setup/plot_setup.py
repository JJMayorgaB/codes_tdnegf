#!/usr/bin/env python3
"""
plot_setup.py -- figuras de una o varias corridas de los setups de dos osciladores
(common.jl). Lee de cada carpeta de corrida:
    geometry.csv  params.txt  spins_t.csv  sites_rho_t.csv  bond_rho_t.csv
    bond_H.csv    lead_currents_t.csv
y escribe en <corrida>/figures/:
    spins_t          M^alpha_m(t) de los 3 LMMs                        (1x3)
    spin_density_t   <sigma^alpha_i>(t) en los 3 sitios con espin      (1x3)
    rho_t            rho_ii: uu, dd, Re/Im ud en los 3 sitios          (2x2)
    bond_rho_t       rho_ij de los enlaces entre espines: Re (-), Im (--) (2x2)
    current_bond_n-m corriente de carga | de espin del enlace n -> m   (1x2)
    lead_currents_t  corrientes de los leads L (-) y R (--)            (1x2)
Mismo estilo que analytical_computations/inbedding_leads_plot.py (de donde se
importan rcParams, formato de ejes, colores de espin y bond_currents).

Uso:
    python plot_setup.py z_precession/output/setup1_<tag>  [otra_corrida ...]
    python plot_setup.py <corrida> --last-periods 3
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
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analytical_computations'))

# importarlo fija los rcParams comunes a todas las figuras
import inbedding_leads_plot as P                     # noqa: E402
from inbedding_leads_plot import (_fmt_axes, _sci_yaxis, _alpha_legend,  # noqa: E402
                                  _legend_en_hueco, SPIN_COLORS, bond_currents)

COMP = (('x', SPIN_COLORS['sx']), ('y', SPIN_COLORS['sy']), ('z', SPIN_COLORS['sz']))
ROLE_TXT = {'D': 'D', 'F': 'F'}
SITE_COLORS = ('#1f4e9c', '#c1272d', '#2a9d5c')           # m = 1, 2, 3
TLAB = r'$\text{Time}\, (2\pi/\Omega)$'


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
    t_on = float(re.search(r't_on = t_relax = ([0-9.eE+-]+)', txt).group(1))
    setup = re.search(r'setup = (\S+)', txt).group(1)
    axis = re.search(r'eje de precesion = (\S+)', txt).group(1)
    return geo, Omega, t_on, setup, axis


def _recorte(df, t_min):
    return df if t_min is None else df[df['t'] >= t_min]


def _tit(ax, texto):
    ax.set_title(texto, fontsize=18)


def plot_spins(run, geo, T, t_min, figdir):
    d = _recorte(pd.read_csv(os.path.join(run, 'spins_t.csv')), t_min)
    x = d['t'].to_numpy() / T
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, (_, g) in zip(axes, geo.iterrows()):
        m = int(g['m'])
        for a, col in COMP:
            ax.plot(x, d[f'M{m}_{a}'].to_numpy(), '-', color=col, lw=1.5, zorder=3)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        _tit(ax, rf'$\text{{m}}={m}\ (\text{{{g["role"]}}},\ \text{{i}}={int(g["site"])})$')
        ax.set_xlabel(TLAB); ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax); _sci_yaxis(ax)
    axes[0].set_ylabel(r'$M^{\alpha}_{\text{m}}(t)$')
    _alpha_legend(axes[0])
    _save(fig, figdir, 'spins_t')


def plot_spin_density(run, geo, T, t_min, figdir):
    d = _recorte(pd.read_csv(os.path.join(run, 'sites_rho_t.csv')), t_min)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, (_, g) in zip(axes, geo.iterrows()):
        s = d[d['site'] == int(g['site'])].sort_values('t')
        x = s['t'].to_numpy() / T
        for a, col in COMP:
            ax.plot(x, s[f's{a}'].to_numpy(), '-', color=col, lw=1.5, zorder=3)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        _tit(ax, rf'$\text{{i}}={int(g["site"])}\ (\text{{{g["role"]}}})$')
        ax.set_xlabel(TLAB); ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax); _sci_yaxis(ax)
    axes[0].set_ylabel(r'$\langle\hat{\sigma}^{\alpha}_{\text{i}}\rangle(t)$')
    _alpha_legend(axes[0])
    _save(fig, figdir, 'spin_density_t')


def plot_rho_sites(run, geo, T, t_min, figdir):
    d = _recorte(pd.read_csv(os.path.join(run, 'sites_rho_t.csv')), t_min)
    cols = [('n_up', r'$\rho^{\uparrow\uparrow}_{\text{ii}}(t)$'),
            ('n_dn', r'$\rho^{\downarrow\downarrow}_{\text{ii}}(t)$'),
            ('Re_rho_updn', r'$\text{Re}\,\rho^{\uparrow\downarrow}_{\text{ii}}(t)$'),
            ('Im_rho_updn', r'$\text{Im}\,\rho^{\uparrow\downarrow}_{\text{ii}}(t)$')]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, (col, ylab) in zip(axes.flat, cols):
        for k, (_, g) in enumerate(geo.iterrows()):
            s = d[d['site'] == int(g['site'])].sort_values('t')
            x = s['t'].to_numpy() / T
            ax.plot(x, s[col].to_numpy(), '-', color=SITE_COLORS[k], lw=1.4, zorder=3)
        ax.set_ylabel(ylab); ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax); _sci_yaxis(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(TLAB)
    _legend_en_hueco(axes[0, 0], [Line2D([], [], color=c, lw=2.0) for c in SITE_COLORS],
                     [rf'$\text{{i}}={int(g["site"])}\ (\text{{{g["role"]}}})$' for _, g in geo.iterrows()],
                     ncol=1, fontsize=15, handlelength=1.6, labelspacing=0.35, handletextpad=0.6)
    _save(fig, figdir, 'rho_t')


def _enlaces_espin(geo):
    s = sorted(int(v) for v in geo['site'])
    return [(s[0], s[1]), (s[1], s[2])]


def plot_bond_rho(run, geo, T, t_min, figdir):
    d = _recorte(pd.read_csv(os.path.join(run, 'bond_rho_t.csv')), t_min)
    role = {int(g['site']): g['role'] for _, g in geo.iterrows()}
    ent = (('uu', r'\uparrow\uparrow'), ('ud', r'\uparrow\downarrow'),
           ('du', r'\downarrow\uparrow'), ('dd', r'\downarrow\downarrow'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    enl = _enlaces_espin(geo)
    for ax, (s, lab) in zip(axes.flat, ent):
        for k, (n, m) in enumerate(enl):
            b = d[(d['n'] == n) & (d['m'] == m)].sort_values('t')
            x = b['t'].to_numpy() / T
            col = SITE_COLORS[k]
            # rho^{ij} = <c_i^dag c_j> = rho_mn del CSV (convencion de bond_currents)
            ax.plot(x, b[f'Re_rho_mn_{s}'].to_numpy(), '-', color=col, lw=1.4, zorder=3)
            ax.plot(x, b[f'Im_rho_mn_{s}'].to_numpy(), '--', color=col, lw=1.4, zorder=3)
        ax.set_ylabel(rf'$\rho^{{{lab}}}_{{\text{{ij}}}}(t)$'); ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax); _sci_yaxis(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(TLAB)
    h = [Line2D([], [], color=SITE_COLORS[k], lw=2.0) for k in range(len(enl))] + \
        [Line2D([], [], color='0.25', lw=2.0, ls='-'), Line2D([], [], color='0.25', lw=2.0, ls='--')]
    l = [rf'$\text{{i}}\rightarrow\text{{j}}={n}\,(\text{{{role[n]}}})\rightarrow{m}\,(\text{{{role[m]}}})$'
         for n, m in enl] + [r'$\text{Re}$', r'$\text{Im}$']
    _legend_en_hueco(axes[0, 0], h, l, ncol=1, fontsize=13, handlelength=1.6,
                     labelspacing=0.3, handletextpad=0.6)
    _save(fig, figdir, 'bond_rho_t')


def _fig_corriente(x, I, IS, figdir, name, titulo):
    """Panel 1x2 como inbedding_current_t: carga | espin, alpha = x, y, z arriba."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(x, I, '-', color='black', lw=1.5, zorder=3)
    ax.set_ylabel(r'$\text{I}_{\text{i}\rightarrow \text{j}}(\text{t})$')
    lo, hi = I.min(), I.max()
    pad = 0.5 * (hi - lo) if hi > lo else max(abs(hi), 1e-300)
    ax.set_ylim(lo - pad, hi + pad)
    if lo - pad < 0.0 < hi + pad:
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
    ax = axes[1]
    for c, (_, col) in enumerate(COMP):
        ax.plot(x, IS[:, c], '-', color=col, lw=1.5, zorder=3)
    ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
    ax.set_ylabel(r'$\text{I}_{\text{i}\rightarrow \text{j}}^{\text{S}_{\alpha}}(\text{t})$')
    for ax in axes:
        ax.set_xlabel(TLAB); ax.set_xlim(x.min(), x.max()); _fmt_axes(ax)
    axes[0].set_title(titulo, fontsize=16, loc='left')
    lab = [r'$\alpha=$', r'x,', r'y,', r'z']
    colr = ['black', SPIN_COLORS['sx'], SPIN_COLORS['sy'], SPIN_COLORS['sz']]
    leg = axes[1].legend([Line2D([], [], ls='none') for _ in lab], lab, loc='upper center',
                         ncol=len(lab), frameon=False, fontsize=20, handlelength=0.0,
                         handletextpad=0.0, columnspacing=0.45, labelcolor=colr, borderaxespad=0.2)
    A = np.abs(IS).max()
    if A > 0:
        fig.tight_layout(); fig.canvas.draw()
        h = leg.get_window_extent().transformed(axes[1].transAxes.inverted()).height
        axes[1].set_ylim(-A / (1.0 - 2.0 * (h + 0.04)), A / (1.0 - 2.0 * (h + 0.04)))
    for ax in axes:
        _sci_yaxis(ax)
    _save(fig, figdir, name)


def plot_bond_currents(run, geo, T, t_min, figdir):
    dfb = _recorte(pd.read_csv(os.path.join(run, 'bond_rho_t.csv')), t_min)
    dfh = pd.read_csv(os.path.join(run, 'bond_H.csv'))
    role = {int(g['site']): g['role'] for _, g in geo.iterrows()}
    for n, m in _enlaces_espin(geo):
        t, I, IS, im = bond_currents(dfb, dfh, 'C', n, m)
        print(f'  enlace {n} -> {m}: max|Im|/max|Re| = {im:.1e}')
        _fig_corriente(t / T, I, IS, figdir, f'current_bond_{n}-{m}',
                       rf'$\text{{i}}={n}\,(\text{{{role[n]}}})\rightarrow\text{{j}}={m}\,(\text{{{role[m]}}})$')


def plot_lead_currents(run, T, t_min, figdir):
    d = _recorte(pd.read_csv(os.path.join(run, 'lead_currents_t.csv')), t_min)
    x = d['t'].to_numpy() / T
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(x, d['I_L'], '-', color='black', lw=1.5, zorder=3)
    ax.plot(x, d['I_R'], '--', color='black', lw=1.5, zorder=3)
    ax.set_ylabel(r'$\text{I}_{\text{L,R}}(\text{t})$')
    ax = axes[1]
    for a, col in COMP:
        ax.plot(x, d[f'Is{a}_L'], '-', color=col, lw=1.5, zorder=3)
        ax.plot(x, d[f'Is{a}_R'], '--', color=col, lw=1.5, zorder=3)
    ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
    ax.set_ylabel(r'$\text{I}^{\text{S}_{\alpha}}_{\text{L,R}}(\text{t})$')
    for ax in axes:
        ax.set_xlabel(TLAB); ax.set_xlim(x.min(), x.max()); _fmt_axes(ax); _sci_yaxis(ax)
    _legend_en_hueco(axes[0], [Line2D([], [], color='0.25', lw=2.0, ls='-'),
                               Line2D([], [], color='0.25', lw=2.0, ls='--')],
                     ['L', 'R'], ncol=1, fontsize=15, handlelength=1.6)
    _alpha_legend(axes[1])
    _save(fig, figdir, 'lead_currents_t')


def main():
    ap = argparse.ArgumentParser(description='Figuras de los setups de dos osciladores.')
    ap.add_argument('runs', nargs='+', help='carpetas de corrida (<precesion>/output/<setup>_<tag>)')
    corte = ap.add_mutually_exclusive_group()
    corte.add_argument('--t-min', type=float, default=None, help='graficar solo t >= t_min')
    corte.add_argument('--last-periods', type=float, default=None, metavar='N',
                       help='graficar solo los ultimos N periodos del driver')
    args = ap.parse_args()

    for run in args.runs:
        run = os.path.abspath(run)
        geo, Omega, t_on, setup, axis = leer_corrida(run)
        T = 2 * np.pi / Omega
        t_min = args.t_min
        figdir = os.path.join(run, 'figures')
        if args.last_periods is not None:
            t_max = pd.read_csv(os.path.join(run, 'spins_t.csv'), usecols=['t'])['t'].max()
            t_min = t_max - args.last_periods * T
            figdir = os.path.join(run, f'figures_last{args.last_periods:g}T')
        elif t_min is not None:
            figdir = os.path.join(run, f'figures_tmin{t_min:g}')
        os.makedirs(figdir, exist_ok=True)
        print(f'== {setup}, precesion en {axis}: {run}\n   roles {list(geo["role"])} en los sitios '
              f'{list(geo["site"])}   T = {T:.1f}   t_on = {t_on:.1f}')
        plot_spins(run, geo, T, t_min, figdir)
        plot_spin_density(run, geo, T, t_min, figdir)
        plot_rho_sites(run, geo, T, t_min, figdir)
        plot_bond_rho(run, geo, T, t_min, figdir)
        plot_bond_currents(run, geo, T, t_min, figdir)
        plot_lead_currents(run, T, t_min, figdir)


if __name__ == '__main__':
    main()
