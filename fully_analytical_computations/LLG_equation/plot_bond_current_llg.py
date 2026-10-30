#!/usr/bin/env python3
"""
plot_bond_current_llg.py -- figura de la corriente de enlace F1 -> F2: LLG-Floquet
(bond_rho_llg.csv de bond_current_llg.jl) frente a TDNEGF (bond_rho_t.csv), en el estado
estacionario (ultimos --last periodos de cada simulacion, la LLG corrida un numero entero de
periodos como en spins_t_cmp).

Las dos corrientes se calculan con la MISMA funcion (bond_currents de inbedding_leads_plot) y el
mismo bond_H.csv de TDNEGF:
    I_{i->j}(t)       = -i Tr_s[ H_ij rho^ij - rho^ji H_ji ]
    I^{S_a}_{i->j}(t) = -i Tr_s[ sigma^a (H_ij rho^ij - rho^ji H_ji) ]
Ademas imprime los controles: <sigma> en los sitios de los F y las componentes de rho^ij
(Tr[rho], Tr[sigma^a rho]) contra TDNEGF.

Figura en output/<tag>/F<a>-<b>/figures/:
    current_bond_<a>-<b>_cmp   1x2 (5,4) por panel: carga | espin (TDNEGF circulos, LLG continua)

Uso (desde LLG_equation):
    python plot_bond_current_llg.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005 --sites 1,2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_llg_two_moments import (leer_params, td_plot, td_handle, _etiqueta, _save,   # noqa: E402
                                  COMP, TLAB, LS_LLG, _fmt_axes, _sci_yaxis, SPIN_COLORS)
from inbedding_leads_plot import bond_currents, _bloques, _PAULI                     # noqa: E402

N_MARK = 70


def figura(xD, ID, ISD, xL, IL, ISL, a, b, outdir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    td_plot(ax, xD, ID, 'black', N_MARK)
    ax.plot(xL, IL, color='black', zorder=4, **LS_LLG)
    ax.set_ylabel(r'$\text{I}_{\text{i}\rightarrow \text{j}}(\text{t})$')
    allI = np.concatenate([IL, ID])
    lo, hi = allI.min(), allI.max()
    pad = 0.5 * (hi - lo) if hi > lo else max(abs(hi), 1e-300)
    ax.set_ylim(lo - pad, hi + pad)
    ax = axes[1]
    for c, (_, col) in enumerate(COMP):
        td_plot(ax, xD, ISD[:, c], col, N_MARK)
        ax.plot(xL, ISL[:, c], color=col, zorder=4, **LS_LLG)
    ax.set_ylabel(r'$\text{I}_{\text{i}\rightarrow \text{j}}^{\text{S}_{\alpha}}(\text{t})$')
    for ax in axes:
        ax.axhline(0.0, color='0.5', ls=':', lw=1.0, zorder=1)
        ax.set_xlabel(TLAB); ax.set_xlim(0.0, xL.max()); _fmt_axes(ax)
    _etiqueta(axes[0], rf'$\text{{i}}={a}\,(\text{{F}})\rightarrow\text{{j}}={b}\,(\text{{F}})$')
    lab = [r'$\alpha=$', r'x,', r'y,', r'z']
    colr = ['black', SPIN_COLORS['sx'], SPIN_COLORS['sy'], SPIN_COLORS['sz']]
    axes[1].legend([Line2D([], [], ls='none') for _ in lab], lab, loc='upper center', ncol=len(lab),
                   frameon=False, fontsize=20, handlelength=0.0, handletextpad=0.0, columnspacing=0.45,
                   labelcolor=colr, borderaxespad=0.2)
    A = np.abs(np.concatenate([ISL, ISD])).max()
    axes[1].set_ylim(-1.6 * A, 1.6 * A)
    h = [td_handle(), Line2D([], [], color='k', **LS_LLG, label='LLG-Floquet')]
    axes[1].add_artist(axes[1].get_legend())
    axes[1].legend(handles=h, loc='lower center', ncol=2, fontsize=14, frameon=False)
    for ax in axes:
        _sci_yaxis(ax)
    _save(fig, outdir, f'current_bond_{a}-{b}_cmp', formats)


def main():
    ap = argparse.ArgumentParser(description='Corriente de enlace F1 -> F2: LLG-Floquet frente a TDNEGF.')
    ap.add_argument('--tag', default='setup2_gso0p1_jsd0p2_th10deg_Om0p005')
    ap.add_argument('--sites', default='1,2')
    ap.add_argument('--last', type=float, default=5.0)
    ap.add_argument('--formats', default='jpg,svg,pdf')
    args = ap.parse_args()
    formats = [f.strip() for f in args.formats.split(',') if f.strip()]
    a, b = (int(s) for s in args.sites.split(','))
    llgdir = os.path.join(HERE, 'output', args.tag, f'F{a}-{b}')
    p = leer_params(llgdir)
    td = p['tdnegf']
    T = 2 * np.pi / p['Omega']

    geo = pd.read_csv(os.path.join(td, 'geometry.csv'))
    sD = int(geo.loc[geo.role == 'D', 'site'].iloc[0])
    si, sj = sD + a, sD + b
    dfh = pd.read_csv(os.path.join(td, 'bond_H.csv'))
    dfL = pd.read_csv(os.path.join(llgdir, 'bond_rho_llg.csv'))
    dfD = pd.read_csv(os.path.join(td, 'bond_rho_t.csv'))

    tL, IL, ISL, imL = bond_currents(dfL, dfh, 'C', si, sj)
    tb, ID, ISD, _ = bond_currents(dfD, dfh, 'C', si, sj)
    # ventana: ultimos --last periodos de TDNEGF; la de la LLG es la misma corrida un numero
    # entero de periodos (misma fase del driver), dentro de lo que escribio bond_current_llg.jl
    tD = tb.max()
    shift = np.floor((tL.max() - tD) / T + 1e-9) * T
    t_ini = tD - args.last * T
    if t_ini + shift < tL.min() - 1e-6:
        raise SystemExit('bond_rho_llg.csv no cubre la ventana: corre bond_current_llg.jl con el mismo --last')
    wL = (tL >= t_ini + shift - 1e-6) & (tL <= tD + shift + 1e-6)
    tL, IL, ISL = tL[wL], IL[wL], ISL[wL]
    w = tb >= t_ini
    xD, ID, ISD = (tb[w] - t_ini) / T, ID[w], ISD[w]
    xL = (tL - shift - t_ini) / T
    dfL = dfL[(dfL['t'] >= t_ini + shift - 1e-6) & (dfL['t'] <= tD + shift + 1e-6)]
    print(f'output/{args.tag}/F{a}-{b}: enlace {si} -> {sj} de TDNEGF; LLG corrida {round(shift / T)}T; '
          f'max |Im I| / max |I| (LLG) = {imL:.1e}')

    # ---- controles contra TDNEGF en los tiempos de la LLG
    f = lambda y: np.interp(xL, xD, y)
    for c, nm in enumerate('xyz'):
        print(f'  I^S{nm}: max|LLG - TDNEGF| = {np.abs(ISL[:, c] - f(ISD[:, c])).max():.2e}   '
              f'max|TDNEGF| = {np.abs(ISD[:, c]).max():.2e}   max|LLG| = {np.abs(ISL[:, c]).max():.2e}')

    def rho_blk(df, n, m):
        d = df[(df['lead'] == 'C') & (df['n'] == n) & (df['m'] == m)].sort_values('t')
        return d['t'].to_numpy(), _bloques(d, 'rho', 'mn')            # rho^{ij} = <c_i^dag c_j>
    tr = lambda s, X: np.einsum('ab,tba->t', s, X)
    _, rL = rho_blk(dfL, si, sj); tr_, rD = rho_blk(dfD, si, sj)
    xr = (tr_ - t_ini) / T
    for nm, s in zip(('1', 'x', 'y', 'z'), (np.eye(2),) + _PAULI):
        yL = tr(s, rL); yD = np.interp(xL, xr, tr(s, rD).real) + 1j * np.interp(xL, xr, tr(s, rD).imag)
        print(f'  Tr[{nm} rho^ij]: max|LLG - TDNEGF| = {np.abs(yL - yD).max():.2e}   max|TDNEGF| = {np.abs(yD).max():.2e}')
    sr = pd.read_csv(os.path.join(td, 'sites_rho_t.csv'))
    for m, sm in ((a, si), (b, sj)):
        _, r = rho_blk(dfL, sm, sm)
        d = sr[sr['site'] == sm].sort_values('t')
        xs = (d['t'].to_numpy() - t_ini) / T
        err = max(np.abs(tr(s, r).real - np.interp(xL, xs, d[f's{c}'].to_numpy())).max()
                  for s, c in zip(_PAULI, 'xyz'))
        print(f'  <sigma_{m}>: max|LLG - TDNEGF| = {err:.2e}   '
              f'max|TDNEGF| = {np.abs(d.loc[d["t"] >= t_ini, ["sx", "sy", "sz"]].to_numpy()).max():.2e}')

    figura(xD, ID, ISD, xL, IL, ISL, a, b, os.path.join(llgdir, 'figures'), formats)


if __name__ == '__main__':
    main()
