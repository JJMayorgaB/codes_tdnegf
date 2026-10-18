#!/usr/bin/env python3
"""
plot_llg_elements.py -- elementos de la LLG efectiva (marco rotante) para todos
los pares de sitios (i,j) ∈ [1,N]^2 del lead:

    intercambio:  J, Γ^{xx,yy,zz,xz,xy,yz}, t^{x,y,z}
    damping:      α, Λ^{xx,yy,zz,xz,xy,yz}, r^{x,y,z}

SOLO figuras (no llama a Julia). Lee data/llg_elements_sites1-N.csv, que escriben en
el cluster (run_cluster.sh):
    julia kernel_elements.jl ... --chi0 --sites N   -> data/kernel_chi0_sites1-N.csv
    julia llg_elements.jl <ese archivo>             -> data/llg_elements_sites1-N.csv
Figuras (por defecto solo .jpg; --formats jpg,svg,pdf) en figures/llg/:
       onsite/  llg_onsite_{intercambio,damping}     X_mm frente a m (todos los elementos)
       maps/    llg_map_<elem>                        X_ij, X^S_ij, X^A_ij en el plano (i,j)
       cuts/    llg_cut_<elem>                        (izq) X_{i,i+d} frente a d, i ∈ sitios F
                                                      (der) X_{i,i+d} frente a (i+j)/2, d fijo

Uso:
    python plot_llg_elements.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005
    python plot_llg_elements.py --tag <tag> --sites 10 --fsites 1,3,5 --dmax 4
    python plot_llg_elements.py                         # sin --tag: params.txt de setup2 (como antes)
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
import plot_kernel_elements as K                       # noqa: E402  (params, etiquetas, rcParams)
from plot_kernel_elements import P, _fmt_axes          # noqa: E402


NAMES_A = ['J', 'Γxx', 'Γyy', 'Γzz', 'Γxz', 'Γxy', 'Γyz', 'tx', 'ty', 'tz']
NAMES_D = ['α', 'Λxx', 'Λyy', 'Λzz', 'Λxz', 'Λxy', 'Λyz', 'rx', 'ry', 'rz']


def tex(nm, extra=None):
    """'Γxz' -> '\\Gamma^{xz}',  'α' -> '\\alpha',  ('ty','A') -> 't^{y,A}'."""
    base = {'J': 'J', 'Γ': r'\Gamma', 't': 't', 'α': r'\alpha', 'Λ': r'\Lambda', 'r': 'r'}[nm[0]]
    sup = ','.join(x for x in (nm[1:], extra) if x)
    return base + (rf'^{{{sup}}}' if sup else '')


def ascii_name(nm):
    return nm.replace('Γ', 'G').replace('Λ', 'L').replace('α', 'alpha')


def load(path, N):
    """Dict elem -> matriz N x N con X[i-1, j-1] = X_ij (NaN si falta)."""
    d = pd.read_csv(path, encoding='utf-8')
    out = {}
    for nm in NAMES_A + NAMES_D:
        M = np.full((N, N), np.nan)
        s = d[d['elemento'] == nm]
        M[s['i'].to_numpy() - 1, s['j'].to_numpy() - 1] = s['valor'].to_numpy()
        out[nm] = M
    return out


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    for ext in K.FORMATS:
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


# ---------------------------------------------------------------- (a) on-site
def plot_onsite(X, N, outdir):
    m = np.arange(1, N + 1)
    for kind, names in (('intercambio', NAMES_A), ('damping', NAMES_D)):
        fig, axes = plt.subplots(2, 5, figsize=(5.0 * 5, 4.0 * 2), squeeze=False)
        for ax, nm in zip(axes.ravel(), names):
            ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
            ax.plot(m, np.diag(X[nm]), 'o-', color='black', ms=5, lw=1.2, zorder=3)
            ax.set_xlabel(r'$m$')
            ax.set_ylabel(rf'${tex(nm)}_{{mm}}$')
            ax.set_xticks(m)
            _fmt_axes(ax)
            P._sci_yaxis(ax)
        _save(fig, outdir, f'llg_onsite_{kind}')


# ---------------------------------------------------------------- (b) mapas (i,j)
def _map(ax, fig, M, N, lab):
    v = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M.T, origin='lower', cmap='seismic', vmin=-v, vmax=v, interpolation='nearest',
                   extent=(0.5, N + 0.5, 0.5, N + 0.5))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(lab, fontsize=16)
    P._sci_cbar(cb)
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$j$')
    ax.set_xticks(range(1, N + 1)); ax.set_yticks(range(1, N + 1))
    _fmt_axes(ax)


def plot_maps(X, N, outdir):
    for nm in NAMES_A + NAMES_D:
        M = X[nm]
        S, A = (M + M.T) / 2, (M - M.T) / 2
        fig, axes = plt.subplots(1, 3, figsize=(5.4 * 3, 4.6))
        for ax, Z, lab in zip(axes, (M, S, A),
                              (rf'${tex(nm)}_{{ij}}$', rf"${tex(nm, 'S')}_{{ij}}$", rf"${tex(nm, 'A')}_{{ij}}$")):
            _map(ax, fig, Z, N, lab)
        _save(fig, outdir, f'llg_map_{ascii_name(nm)}')


# ---------------------------------------------------------------- (c) cortes
def plot_cuts(X, N, outdir, fsites, dmax):
    cols = plt.get_cmap('tab10')
    ds = np.arange(-dmax, dmax + 1)
    for nm in NAMES_A + NAMES_D:
        M = X[nm]
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.0 * 2, 4.6))
        for c, i in enumerate(fsites):
            y = [M[i - 1, i + d - 1] if 1 <= i + d <= N else np.nan for d in ds]
            a0.plot(ds, y, 'o-', color=cols(c), ms=5, lw=1.3, label=rf'$i={i}$')
        a0.set_xlabel(r'$d = j - i$')
        a0.set_ylabel(rf'${tex(nm)}_{{i,i+d}}$')
        a0.set_xticks(ds)
        for c, d in enumerate(range(0, dmax + 1)):
            i = np.arange(1, N + 1 - d)
            a1.plot((2 * i + d) / 2, M[i - 1, i + d - 1], 'o-', color=cols(c), ms=4, lw=1.2, label=rf'$d={d}$')
        a1.set_xlabel(r'$(i+j)/2$')
        a1.set_ylabel(rf'${tex(nm)}_{{i,i+d}}$')
        for ax in (a0, a1):
            ax.axhline(0.0, color='0.6', lw=0.8, zorder=1)
            ax.legend(loc='best', frameon=False, fontsize=12, handlelength=1.6)
            _fmt_axes(ax)
            P._sci_yaxis(ax)
        _save(fig, outdir, f'llg_cut_{ascii_name(nm)}')


def main():
    ap = argparse.ArgumentParser(description='Elementos de la LLG para todos los pares (i,j) ∈ [1,N]^2 (sin Julia).')
    ap.add_argument('--tag', default=None, help='carpeta output/<tag>')
    ap.add_argument('--sites', type=int, default=10)
    ap.add_argument('--fsites', default='1,3,5', help='sitios i de los cortes frente a d (primer F de cada configuracion)')
    ap.add_argument('--dmax', type=int, default=4)
    ap.add_argument('--formats', default='jpg', help='formatos de las figuras, p. ej. jpg,svg,pdf')
    ap.add_argument('--params', default=K.DEFAULT_PARAMS, help='sin --tag: params.txt de TDNEGF')
    for k in K.DEFAULTS:
        ap.add_argument(f'--{k}', type=float, default=None)
    args = ap.parse_args()

    K.FORMATS[:] = [f.strip() for f in args.formats.split(',') if f.strip()]
    if args.tag:
        tag = args.tag
    else:
        if args.params != 'none':
            args.params = os.path.abspath(args.params)
        vals = K.read_params(args.params)
        overridden = args.params == 'none' or any(getattr(args, k) is not None for k in K.DEFAULTS)
        for k in K.DEFAULTS:
            if getattr(args, k) is not None:
                vals[k] = getattr(args, k)
        tag = K.param_tag(vals) if overridden else K.run_tag(args.params)
    datadir = os.path.join(HERE, 'output', tag, 'data')
    figdir = os.path.join(HERE, 'output', tag, 'figures', 'llg')
    N = args.sites

    fl = os.path.join(datadir, f'llg_elements_sites1-{N}.csv')
    K.need(os.path.isfile(fl), fl,
           f'julia kernel_elements.jl ... --chi0 --sites {N}  y  julia llg_elements.jl '
           f'output/{tag}/data/kernel_chi0_sites1-{N}.csv  (ver run_cluster.sh)')
    src = os.path.join(datadir, f'kernel_chi0_sites1-{N}.csv')
    K.need(K.sign_ok([src]), f'{fl} (y {os.path.basename(src)}) con el signo viejo (-i J_sd)',
           f'python fix_kernel_sign.py --tag {tag}   (en local, sin recalcular)')
    print(f'output/{tag}   |   {N}x{N} pares   |   formatos {K.FORMATS}')
    X = load(fl, N)
    fs = [int(s) for s in args.fsites.split(',') if s.strip()]
    plot_onsite(X, N, os.path.join(figdir, 'onsite'))
    plot_maps(X, N, os.path.join(figdir, 'maps'))
    plot_cuts(X, N, os.path.join(figdir, 'cuts'), fs, args.dmax)


if __name__ == '__main__':
    main()
