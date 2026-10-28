#!/usr/bin/env python3
"""
plot_llg_two_moments.py -- figuras de la LLG efectiva de dos momentos (llg_two_moments.jl).

Figuras en output/<tag>/F<a>-<b>/figures/ (o en --llgdir/figures):
    spins_t_cmp      1x3 (D, F1, F2): M^alpha_m(t) en el laboratorio, ultimos --last periodos.
                     TDNEGF en circulos abiertos y LLG-Floquet en linea continua (mismo color
                     por componente). El panel del D es el control de la fase (deben coincidir).
    spins_rot_llg    1x3 (D, F1, F2): Mbar^alpha_m(t) en el marco rotante, toda la evolucion
                     desde t0, solo LLG (no se compara con TDNEGF).

Uso (desde LLG_equation):
    python plot_llg_two_moments.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005 --sites 3,6
    python plot_llg_two_moments.py --tag <tag> --sites 3,6 --last 3 --tdnegf none
    python plot_llg_two_moments.py --llgdir output/<tag>/F3-6_icpy
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
sys.path.insert(0, os.path.join(HERE, '..', '..', 'analytical_computations'))
# importarlo fija los rcParams comunes a todas las figuras
import inbedding_leads_plot as P                                     # noqa: E402,F401
from inbedding_leads_plot import _fmt_axes, _sci_yaxis, _alpha_legend, SPIN_COLORS   # noqa: E402

COMP = (('x', SPIN_COLORS['sx']), ('y', SPIN_COLORS['sy']), ('z', SPIN_COLORS['sz']))
TLAB = r'$\text{Time}\, (2\pi/\Omega)$'
N_MARK = 70                     # circulos de TDNEGF por curva en la ventana
TDROOT = os.path.join(HERE, '..', '..', 'two_oscillators_setup', 'y_precession', 'output', 'DFF')


def _save(fig, outdir, name, formats):
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    for ext in formats:
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def leer_params(llgdir):
    txt = open(os.path.join(llgdir, 'params_llg.txt'), encoding='utf-8').read()
    num = lambda k: float(re.search(k + r'\s*=\s*([-+0-9.eE]+)', txt).group(1))
    td = re.search(r'tdnegf = (.*)', txt).group(1).strip()
    sit = re.search(r'F1 = (\d+), F2 = (\d+)', txt)
    tag = re.search(r'tag = (\S+)', txt).group(1)
    a, b = int(sit.group(1)), int(sit.group(2))
    if td != 'none' and not os.path.isdir(td):
        # la ruta guardada es la de la maquina donde se corrio (p. ej. el cluster): misma regla local
        rest = re.sub(r'^setup\d+_', '', tag)
        td = os.path.join(TDROOT, f'F{a}-{b}', rest + '_buf4_adyn0p0')
    return dict(Omega=num('Omega'), t_on=num('t_on'), t0=num('t0'), tdnegf=td, a=a, b=b)


def titulos(p, td):
    """Titulos de los paneles; con TDNEGF, los mismos de plot_setup.py (m, rol, sitio)."""
    if td != 'none' and os.path.isfile(os.path.join(td, 'geometry.csv')):
        geo = pd.read_csv(os.path.join(td, 'geometry.csv'))
        sD = int(geo.loc[geo.role == 'D', 'site'].iloc[0])
        mD = int(geo.loc[geo.role == 'D', 'm'].iloc[0])
        mF = [int(geo.loc[geo.site == sD + d, 'm'].iloc[0]) for d in (p['a'], p['b'])]
        ms = [mD] + mF
        sites = [sD, sD + p['a'], sD + p['b']]
        roles = ['D', 'F', 'F']
        return ms, [rf'$\text{{m}}={m}\ (\text{{{r}}},\ \text{{i}}={s})$' for m, r, s in zip(ms, roles, sites)]
    return [1, 2, 3], [r'$\text{D}$', rf'$\text{{F}}_1\ (\text{{d}}={p["a"]})$',
                       rf'$\text{{F}}_2\ (\text{{d}}={p["b"]})$']


def fig_cmp(llgdir, p, td, last, formats):
    T = 2 * np.pi / p['Omega']
    L = pd.read_csv(os.path.join(llgdir, 'spins_t_llg.csv'))
    t_min = L['t'].max() - last * T
    L = L[L['t'] >= t_min - 1e-9]
    ms, tit = titulos(p, td)
    D = None
    if td != 'none':
        D = pd.read_csv(os.path.join(td, 'spins_t.csv'))
        D = D[(D['t'] >= t_min - 1e-9) & (D['t'] <= L['t'].max() + 1e-9)]
        D = D.iloc[::max(1, len(D) // N_MARK)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for k, ax in enumerate(axes):
        for a, col in COMP:
            if D is not None:
                ax.plot(D['t'] / T, D[f'M{ms[k]}_{a}'], 'o', mfc='none', mec=col, ms=5, mew=1.0, zorder=3)
            ax.plot(L['t'] / T, L[f'M{k + 1}_{a}'], '-', color=col, lw=1.5, zorder=4)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        ax.set_title(tit[k], fontsize=18)
        ax.set_xlabel(TLAB); ax.set_xlim(L['t'].min() / T, L['t'].max() / T)
        _fmt_axes(ax); _sci_yaxis(ax)
    axes[0].set_ylabel(r'$M^{\alpha}_{\text{m}}(t)$')
    _alpha_legend(axes[0])
    # leyenda de metodos en el panel del D (sus curvas dejan libre la franja central)
    h = [Line2D([], [], color='k', lw=1.5, label='LLG-Floquet')]
    if D is not None:
        h.insert(0, Line2D([], [], ls='none', marker='o', mfc='none', mec='k', ms=6, label='TDNEGF'))
    axes[0].add_artist(axes[0].get_legend())
    axes[0].legend(handles=h, loc='center right', fontsize=14, frameon=False)
    _save(fig, os.path.join(llgdir, 'figures'), 'spins_t_cmp', formats)


def fig_rot(llgdir, p, td, formats):
    T = 2 * np.pi / p['Omega']
    R = pd.read_csv(os.path.join(llgdir, 'spins_rot_llg.csv'))
    _, tit = titulos(p, td)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for k, ax in enumerate(axes):
        for a, col in COMP:
            ax.plot(R['t'] / T, R[f'M{k + 1}_{a}'], '-', color=col, lw=1.5, zorder=3)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        ax.set_title(tit[k], fontsize=18)
        ax.set_xlabel(TLAB); ax.set_xlim(R['t'].min() / T, R['t'].max() / T)
        _fmt_axes(ax); _sci_yaxis(ax)
    axes[0].set_ylabel(r'$\bar{M}^{\alpha}_{\text{m}}(t)$')
    _alpha_legend(axes[0])
    _save(fig, os.path.join(llgdir, 'figures'), 'spins_rot_llg', formats)


def main():
    ap = argparse.ArgumentParser(description='Figuras de llg_two_moments.jl (comparacion con TDNEGF y marco rotante).')
    ap.add_argument('--tag', default='')
    ap.add_argument('--sites', default='', help='a,b (sitios del analitico de F1 y F2)')
    ap.add_argument('--llgdir', default='', help='carpeta de la corrida (defecto output/<tag>/F<a>-<b>)')
    ap.add_argument('--tdnegf', default='', help='carpeta de TDNEGF o none (defecto: la de params_llg.txt)')
    ap.add_argument('--last', type=float, default=5.0, help='periodos de la ventana de comparacion')
    ap.add_argument('--formats', default='jpg,svg,pdf')
    args = ap.parse_args()
    if args.llgdir:
        llgdir = args.llgdir
    else:
        if not (args.tag and args.sites):
            ap.error('usa --tag y --sites, o --llgdir')
        a, b = args.sites.split(',')
        llgdir = os.path.join(HERE, 'output', args.tag, f'F{a}-{b}')
    p = leer_params(llgdir)
    td = args.tdnegf or p['tdnegf']
    if td != 'none' and not os.path.isdir(td):
        print(f'  aviso: no encuentro la corrida de TDNEGF {td}; solo se dibuja la LLG')
        td = 'none'
    formats = [f.strip() for f in args.formats.split(',') if f.strip()]
    print(llgdir)
    fig_cmp(llgdir, p, td, args.last, formats)
    fig_rot(llgdir, p, td, formats)


if __name__ == '__main__':
    main()
