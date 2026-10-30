#!/usr/bin/env python3
"""
plot_llg_two_moments.py -- figuras de la LLG efectiva de dos momentos (llg_two_moments.jl)
comparada con TDNEGF. Se compara el ESTADO FINAL de cada simulacion (cada una al final de su
propia evolucion), no ventanas de tiempo iguales.

Figuras en output/<tag>/F<a>-<b>/figures/ (o en --llgdir/figures):
    spins_t_cmp         1x3 (D, F1, F2): M^alpha_m(t) en el laboratorio, ultimos --last periodos
                        de cada simulacion. TDNEGF en circulos abiertos (--td-style dashed: linea
                        discontinua) y LLG-Floquet en linea continua (mismo color por componente). La ventana de la LLG se corre un
                        numero entero de periodos para que caiga sobre la de TDNEGF: la fase del
                        driver coincide y el panel del D sirve de control.
    psd_cmp_{log,lin}   1x3 (D, F1, F2): PSD de M^alpha_m en los ultimos --psd-periods periodos de
                        cada simulacion (TDNEGF en circulos, LLG continua). Misma definicion que
                        psd_setup.py: P(w) = |Σ w_n x_n e^{iwt_n}|^2/(Σ w_n)^2 con ventana de Hann,
                        A cos(w0 t) -> pico (A/2)^2; en la escala lineal la componente y va en un
                        eje derecho propio (verde).
    spins_rot_llg       1x3 (D, F1, F2): Mbar^alpha_m(t) en el marco rotante, toda la evolucion
                        de la LLG (convergencia al atractor). No se compara con TDNEGF.

Uso (desde LLG_equation):
    python plot_llg_two_moments.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005 --sites 3,6
    python plot_llg_two_moments.py --tag <tag> --sites 3,6 --last 3 --psd-periods 20 --tdnegf none
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
sys.path.insert(0, os.path.join(HERE, '..', '..', 'two_oscillators_setup'))
# importarlo fija los rcParams comunes a todas las figuras
import inbedding_leads_plot as P                                     # noqa: E402,F401
from inbedding_leads_plot import _fmt_axes, _sci_yaxis, _alpha_legend, SPIN_COLORS   # noqa: E402
from psd_setup import psd, _ylim, _unificar_y                         # noqa: E402

COMP = (('x', SPIN_COLORS['sx']), ('y', SPIN_COLORS['sy']), ('z', SPIN_COLORS['sz']))
TLAB = r'$\text{Time}\, (2\pi/\Omega)$'
WLAB = r'$\omega/\Omega$'
TDROOT = os.path.join(HERE, '..', '..', 'two_oscillators_setup', 'y_precession', 'output', 'DFF')
LS_TD = dict(ls='--', lw=1.3)          # TDNEGF con --td-style dashed
TD_STYLE = 'circles'                   # TDNEGF: circulos abiertos (defecto) o linea discontinua
N_MARK_T = 70                          # circulos por curva en spins_t_cmp
N_MARK_W = 90                          # circulos por curva en la PSD (en [0, wmax])
LS_LLG = dict(ls='-', lw=1.6)          # LLG-Floquet


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
    return dict(Omega=num('Omega'), t_on=num('t_on'), tdnegf=td, a=a, b=b)


def titulos(p, td):
    """Titulos y columna m de TDNEGF de cada panel (D, F1, F2), como en plot_setup.py."""
    if td != 'none' and os.path.isfile(os.path.join(td, 'geometry.csv')):
        geo = pd.read_csv(os.path.join(td, 'geometry.csv'))
        sD = int(geo.loc[geo.role == 'D', 'site'].iloc[0])
        mD = int(geo.loc[geo.role == 'D', 'm'].iloc[0])
        mF = [int(geo.loc[geo.site == sD + d, 'm'].iloc[0]) for d in (p['a'], p['b'])]
        ms = [mD] + mF
        sites = [sD, sD + p['a'], sD + p['b']]
        return ms, [rf'$\text{{m}}={m}\ (\text{{{r}}},\ \text{{i}}={s})$'
                    for m, r, s in zip(ms, ['D', 'F', 'F'], sites)]
    return [1, 2, 3], [r'$\text{D}$', rf'$\text{{F}}_1\ (\text{{d}}={p["a"]})$',
                       rf'$\text{{F}}_2\ (\text{{d}}={p["b"]})$']


def td_plot(ax, x, y, col, n_mark):
    """Curva de TDNEGF: circulos abiertos submuestreados (o linea discontinua)."""
    x = np.asarray(x); y = np.asarray(y)
    if TD_STYLE == 'circles':
        st = max(1, len(x) // n_mark)
        ax.plot(x[::st], y[::st], ls='none', marker='o', mfc='none', mec=col, ms=5, mew=1.0, zorder=3)
    else:
        ax.plot(x, y, color=col, zorder=3, **LS_TD)


def td_handle():
    if TD_STYLE == 'circles':
        return Line2D([], [], ls='none', marker='o', mfc='none', mec='k', ms=6, label='TDNEGF')
    return Line2D([], [], color='k', **LS_TD, label='TDNEGF')


def _leyenda_metodos(ax, con_td):
    h = [Line2D([], [], color='k', **LS_LLG, label='LLG-Floquet')]
    if con_td:
        h.insert(0, td_handle())
    ax.add_artist(ax.get_legend())
    ax.legend(handles=h, loc='center right', fontsize=14, frameon=False)


# ---------------------------------------------------------------- trayectorias
def fig_cmp(llgdir, p, td, D, last, formats):
    T = 2 * np.pi / p['Omega']
    L = pd.read_csv(os.path.join(llgdir, 'spins_t_llg.csv'))
    ms, tit = titulos(p, td)
    tL = L['t'].max()
    if D is not None:
        tD = D['t'].max()
        shift = round((tL - tD) / T) * T                 # entero de periodos: misma fase del driver
        Dw = D[D['t'] >= tD - last * T]
        Lw = L[(L['t'] >= tD - last * T + shift) & (L['t'] <= tD + shift)]
        xL = (Lw['t'] - shift) / T
        nota = f'LLG: ultimos {last:g}T de su evolucion, corrida {round(shift / T)}T'
    else:
        Lw = L[L['t'] >= tL - last * T]
        xL = Lw['t'] / T
        nota = None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for k, ax in enumerate(axes):
        for a, col in COMP:
            if D is not None:
                td_plot(ax, Dw['t'] / T, Dw[f'M{ms[k]}_{a}'], col, N_MARK_T)
            ax.plot(xL, Lw[f'M{k + 1}_{a}'], color=col, zorder=4, **LS_LLG)
        ax.axhline(0.0, color='0.5', ls=':', lw=1.0, zorder=1)
        ax.set_title(tit[k], fontsize=18)
        ax.set_xlabel(TLAB); ax.set_xlim(xL.min(), xL.max())
        _fmt_axes(ax); _sci_yaxis(ax)
    axes[0].set_ylabel(r'$M^{\alpha}_{\text{m}}(t)$')
    _alpha_legend(axes[0])
    _leyenda_metodos(axes[0], D is not None)
    if nota:
        print('  ' + nota)
    _save(fig, os.path.join(llgdir, 'figures'), 'spins_t_cmp', formats)


def fig_rot(llgdir, p, td, formats):
    T = 2 * np.pi / p['Omega']
    R = pd.read_csv(os.path.join(llgdir, 'spins_rot_llg.csv'))
    _, tit = titulos(p, td)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for k, ax in enumerate(axes):
        for a, col in COMP:
            ax.plot(R['t'] / T, R[f'M{k + 1}_{a}'], '-', color=col, lw=1.2, zorder=3)
        ax.axhline(0.0, color='0.5', ls=':', lw=1.0, zorder=1)
        ax.set_title(tit[k], fontsize=18)
        ax.set_xlabel(TLAB); ax.set_xlim(R['t'].min() / T, R['t'].max() / T)
        _fmt_axes(ax); _sci_yaxis(ax)
    axes[0].set_ylabel(r'$\bar{M}^{\alpha}_{\text{m}}(t)$')
    _alpha_legend(axes[0])
    _save(fig, os.path.join(llgdir, 'figures'), 'spins_rot_llg', formats)


# ---------------------------------------------------------------- PSD
def _psd_final(df, cols, T, nper, Omega, wmax, pad):
    """PSD de las columnas en los ultimos nper periodos (numero entero) de df."""
    t_end = df['t'].max()
    d = df[df['t'] > t_end - nper * T].sort_values('t')
    t = d['t'].to_numpy()
    return {c: psd(t, d[c].to_numpy(), Omega, 'hann', pad, wmax) for c in cols}


def _panel(ax, curvas, escala, wmax):
    """curvas: (omega, P, color, estilo). En la escala lineal la componente y va al eje derecho."""
    cy = SPIN_COLORS['sy']
    izq = [c for c in curvas if not (escala == 'lin' and c[2] == cy)]
    der = [c for c in curvas if escala == 'lin' and c[2] == cy]
    for om, Pw, col, st in izq:
        td_plot(ax, om, Pw, col, N_MARK_W) if st == 'TD' else ax.plot(om, Pw, color=col, zorder=3, **st)
    for n in (1, 2):
        ax.axvline(n, color='0.5', ls=':', lw=1.2, zorder=1)
    _ylim(ax, max(Pw.max() for _, Pw, _, _ in izq), escala)
    ax.set_xlim(0.0, wmax)
    _fmt_axes(ax)
    if escala == 'lin':
        _sci_yaxis(ax)
    if not der:
        return None
    ax.tick_params(axis='y', right=False)
    ax2 = ax.twinx()
    for om, Pw, col, st in der:
        td_plot(ax2, om, Pw, col, N_MARK_W) if st == 'TD' else ax2.plot(om, Pw, color=col, zorder=3, **st)
    _ylim(ax2, max(Pw.max() for _, Pw, _, _ in der), escala)
    ax2.tick_params(axis='y', direction='in', right=True, left=False, size=5.0, width=1.0, colors=cy)
    ax2.spines['right'].set_color(cy)
    _sci_yaxis(ax2)
    ax2.yaxis.get_offset_text().set_color(cy)
    return ax2


def fig_psd(llgdir, p, td, D, nper, wmax, pad, formats):
    T = 2 * np.pi / p['Omega']
    L = pd.read_csv(os.path.join(llgdir, 'spins_t_llg.csv'))
    ms, tit = titulos(p, td)
    PL = _psd_final(L, [f'M{k}_{a}' for k in (1, 2, 3) for a, _ in COMP], T, nper, p['Omega'], wmax, pad)
    PD = None
    if D is not None:
        PD = _psd_final(D, [f'M{m}_{a}' for m in ms for a, _ in COMP], T, nper, p['Omega'], wmax, pad)
    for esc in ('log', 'lin'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ders = []
        for k, ax in enumerate(axes):
            cur = []
            for a, col in COMP:
                if PD is not None:
                    cur.append((*PD[f'M{ms[k]}_{a}'], col, 'TD'))
                cur.append((*PL[f'M{k + 1}_{a}'], col, LS_LLG))
            ders.append(_panel(ax, cur, esc, wmax))
            ax.set_title(tit[k], fontsize=18)
            ax.set_xlabel(WLAB)
        axes[0].set_ylabel(r'PSD $M^{\alpha}_{\text{m}}$')
        _unificar_y(list(axes), mostrar=0, lado='left')
        tw = [d for d in ders if d is not None]
        if len(tw) > 1:
            _unificar_y(tw, mostrar=len(tw) - 1, lado='right')
        if ders[-1] is not None:
            ders[-1].set_ylabel(r'PSD $M^{y}_{\text{m}}$', color=SPIN_COLORS['sy'])
        _alpha_legend(axes[0])
        h = [Line2D([], [], color='k', **LS_LLG, label='LLG-Floquet')]
        if PD is not None:
            h.insert(0, td_handle())
        axes[1].legend(handles=h, loc='upper right', fontsize=13, frameon=False)
        _save(fig, os.path.join(llgdir, 'figures'), f'psd_cmp_{esc}', formats)
    print(f'  PSD: ultimos {nper:g} periodos de cada simulacion (resolucion {1 / nper:.3f} Omega)')


def main():
    ap = argparse.ArgumentParser(description='Figuras de llg_two_moments.jl (estado final frente a TDNEGF, PSD, marco rotante).')
    ap.add_argument('--tag', default='')
    ap.add_argument('--sites', default='', help='a,b (sitios del analitico de F1 y F2)')
    ap.add_argument('--llgdir', default='', help='carpeta de la corrida (defecto output/<tag>/F<a>-<b>)')
    ap.add_argument('--tdnegf', default='', help='carpeta de TDNEGF o none (defecto: la de params_llg.txt)')
    ap.add_argument('--last', type=float, default=5.0, help='periodos finales en spins_t_cmp')
    ap.add_argument('--psd-periods', type=float, default=10.0, help='periodos finales para la PSD')
    ap.add_argument('--wmax', type=float, default=3.0, help='omega/Omega maximo de la PSD')
    ap.add_argument('--pad', type=int, default=8, help='relleno con ceros de la PSD (solo dibujo)')
    ap.add_argument('--formats', default='jpg,svg,pdf')
    ap.add_argument('--td-style', choices=['circles', 'dashed'], default='circles',
                    help='TDNEGF en circulos abiertos (defecto) o en linea discontinua')
    args = ap.parse_args()
    global TD_STYLE
    TD_STYLE = args.td_style
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
    D = pd.read_csv(os.path.join(td, 'spins_t.csv')) if td != 'none' else None
    formats = [f.strip() for f in args.formats.split(',') if f.strip()]
    print(llgdir)
    fig_cmp(llgdir, p, td, D, args.last, formats)
    fig_psd(llgdir, p, td, D, args.psd_periods, args.wmax, args.pad, formats)
    fig_rot(llgdir, p, td, formats)


if __name__ == '__main__':
    main()
