#!/usr/bin/env python3
"""
plot_time_kernel.py -- paneles de chi^{mu nu}_{ij}(t,t') calculados por
time_kernel.jl.

Para cada (mu,nu) salen DOS figuras de n x n paneles (n = numero de sitios):
    time_kernel_{mu}{nu}_{sitios}_Re.{jpg,svg,pdf}   Re chi
    time_kernel_{mu}{nu}_{sitios}_Im.{jpg,svg,pdf}   Im chi
Panel (fila j, columna i), como en
    (i=1,j=1) | (i=2,j=1) | ...
    (i=1,j=2) | (i=2,j=2) | ...
Eje x = t, eje y = t' (en periodos del drive). Cada panel lleva su propia barra
de color, seismic centrada en cero.

Uso:
    python plot_time_kernel.py --mu x --nu y --sites 1,2,3,4
    python plot_time_kernel.py --all --sites 1,7,13,20
    python plot_time_kernel.py --mu x --nu y --sites 1,2,3,4 f11.npy f21.npy ... (16)
Los 16 archivos explicitos van en el orden de lectura de la figura:
(i=1,j=1), (i=2,j=1), ..., (i=4,j=1), (i=1,j=2), ...
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')        # sin ventana Qt: solo guardar (evita que se acumule memoria)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# importarlo tambien fija los rcParams comunes a todas las figuras
from inbedding_leads_plot import _fmt_axes, _sci_cbar    # noqa: E402

DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'output', 'time_kernel', 'data')      # .npy del cluster
DEFAULT_FIG = os.path.join(SCRIPT_DIR, 'output', 'time_kernel', 'figures')   # figuras
COMPS = ('x', 'y', 'z')
PANEL = (5.0, 4.0)                  # tamano de cada panel individual
DPI = 300                           # reconstruct_time_kernel.py lo sube para ventanas largas
FORMATS = ('jpg', 'svg', 'pdf')   # reconstruct_time_kernel.py --formats lo cambia
_CMAP = plt.get_cmap('seismic').with_extremes(bad='white')   # nan (zona oculta) en blanco


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in FORMATS:
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=DPI)
        print(f'  -> {path}')
    plt.close(fig)


def _load_meta(indir, tag):
    path = os.path.join(indir, f'time_kernel_{tag}_meta.json')
    if not os.path.isfile(path):
        raise SystemExit(f'no encuentro {path} (lo escribe time_kernel.jl)')
    with open(path) as f:
        return json.load(f)


def _paths(indir, mu, nu, labels, tag):
    """Orden de lectura de la figura: j por filas, i por columnas."""
    return [os.path.join(indir, f'time_kernel_{mu}{nu}_{tag}_i{i}_j{j}.npy')
            for j in labels for i in labels]


def _titulo(parte, mu, nu):
    return (r'$\text{' + parte + r'}\,\chi^{\text{' + mu + nu +
            r'}}_{\text{ij}}(t,t^{\prime})$')


def _barra_global(fig, ax, im, vmax, titulo):
    """
    Titulo de la figura y barra de color comun, dentro del triangulo vacio
    (t' > t) del panel (1,1), en la esquina superior izquierda:
        titulo alineado a la izquierda, arriba
        barra horizontal debajo, tambien a la izquierda, con el multiplicador
        x10^p justo al final de la barra
    Todo queda con x < y (coordenadas del panel), o sea sin tocar los datos.
    """
    ax.text(0.04, 0.975, titulo, transform=ax.transAxes, ha='left', va='top',
            fontsize=22)
    cax = ax.inset_axes([0.04, 0.70, 0.38, 0.07])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    # matplotlib rasteriza la barra (>= 50 niveles); en pdf/svg eso abre un
    # lienzo del tamano de la figura entera. Como vector pesa poco.
    cb.solids.set_rasterized(False)
    p = int(np.floor(np.log10(vmax)))
    cb.locator = mticker.MaxNLocator(nbins=4, symmetric=True)
    cb.formatter = mticker.FuncFormatter(lambda v, _: '$' + f'{v / 10.0 ** p:.3g}' + '$')
    cb.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cax.tick_params(labelsize=17, direction='in', length=5)
    cax.text(1.04, 0.5, r'$\times 10^{' + f'{p}' + r'}$', transform=cax.transAxes,
             ha='left', va='center', fontsize=17)


def plot_parte(data, t, labels, mu, nu, parte, outdir, tag):
    """Una figura n x n para Re o Im."""
    n = len(labels)
    # ejes compartidos: toda la malla usa la misma ventana (t,t'), asi que los
    # numeros de los ticks solo van en la columna izquierda y la fila de abajo
    fig, axes = plt.subplots(n, n, figsize=(PANEL[0] * n, PANEL[1] * n),
                             squeeze=False, sharex=True, sharey=True)
    f = np.real if parte == 'Re' else np.imag

    # UNA escala de color para los n x n paneles, para poder comparar amplitudes
    vmax = max(np.nanmax(np.abs(f(Z) if np.iscomplexobj(Z) else Z)) for Z in data.values())   # nan = zona oculta
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    for r, j in enumerate(labels):
        for c, i in enumerate(labels):
            ax = axes[r, c]
            Z = data[(i, j)]
            if np.iscomplexobj(Z):
                Z = f(Z)
            Z = np.asarray(Z, np.float32)        # Z[a,b] = chi(t_a, t'_b); float32 ahorra RAM
            # imshow en vez de pcolormesh: la malla es uniforme, asi que es lo
            # mismo que shading='nearest', pero entra al pdf/svg como UNA imagen
            # de 601x601. pcolormesh rasterizado le pide al backend mixto un
            # lienzo del tamano de toda la figura por panel y se queda sin RAM.
            # C[y,x]: y = t' (indice b), x = t (indice a) -> Z.T; origin='lower'.
            h = 0.5 * (t[1] - t[0])
            im = ax.imshow(Z.T, origin='lower', aspect='auto', interpolation='nearest',
                           extent=(t[0] - h, t[-1] + h, t[0] - h, t[-1] + h),
                           cmap=_CMAP, vmin=-vmax, vmax=vmax)
            # i encabeza las columnas (arriba) y j las filas (a la izquierda,
            # girada 90 grados), en vez de un titulo (i,j) por panel
            if r == 0:
                ax.set_title(r'$\text{i}=' + f'{i}' + r'$', fontsize=22, pad=12)
            if c == 0:
                # con ticks decimales (ventanas < 2T) el rotulo j se corre mas a la izquierda
                ax.text(-0.25 if t[-1] >= 2 else -0.42, 0.5, r'$\text{j}=' + f'{j}' + r'$', transform=ax.transAxes,
                        rotation=90, ha='center', va='center', fontsize=22)
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(t[0], t[-1])
            # ticks enteros solo si la ventana abarca varios periodos (zoom: no)
            entero = t[-1] >= 2
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=entero, nbins=4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=entero, nbins=4))
            _fmt_axes(ax)
            ax.set_axisbelow(False)              # ticks por encima del mapa
            if r == n - 1:
                ax.set_xlabel(r'$\text{Time t}\ (2\pi/\Omega)$')
            if c == 0:
                ax.set_ylabel(r'$\text{Time t}^{\prime}\ (2\pi/\Omega)$')
            ax.label_outer()                     # ticks quedan, numeros solo afuera

    _barra_global(fig, axes[0, 0], im, vmax, _titulo(parte, mu, nu))
    _save(fig, outdir, f'time_kernel_{mu}{nu}_{tag}_{parte}')


def plot_mn(mu, nu, labels, indir, outdir, files=None):
    tag = '-'.join(str(s) for s in labels)
    meta = _load_meta(indir, tag)
    T = 2 * np.pi / meta['Omega']
    t = np.arange(meta['Nt']) * meta['dt'] / T         # en periodos

    paths = files if files else _paths(indir, mu, nu, labels, tag)
    if len(paths) != len(labels) ** 2:
        raise SystemExit(f'hacen falta {len(labels) ** 2} componentes, llegaron {len(paths)}')
    faltan = [q for q in paths if not os.path.isfile(q)]
    if faltan:
        raise SystemExit('faltan archivos:\n  ' + '\n  '.join(faltan))

    keys = [(i, j) for j in labels for i in labels]
    data = {k: np.load(q) for k, q in zip(keys, paths)}

    print(f'  chi^{mu}{nu}   max|Re|   max|Im|')
    for (i, j), Z in data.items():
        print(f'    ({i},{j})   {np.abs(Z.real).max():.3e}  {np.abs(Z.imag).max():.3e}')

    for parte in ('Re', 'Im'):
        plot_parte(data, t, labels, mu, nu, parte, outdir, tag)


def main():
    ap = argparse.ArgumentParser(description='Paneles de chi^{mu nu}_{ij}(t,t\').')
    ap.add_argument('files', nargs='*',
                    help='las n^2 componentes en orden de lectura de la figura '
                         '(opcional; por defecto se arman desde --indir)')
    ap.add_argument('--mu', choices=COMPS)
    ap.add_argument('--nu', choices=COMPS)
    ap.add_argument('--all', action='store_true',
                    help='las 9 combinaciones (mu,nu) que existan en --indir')
    ap.add_argument('--sites', default='1,2,3,4')
    ap.add_argument('--indir', default=DEFAULT_DIR)
    ap.add_argument('--outdir', default=DEFAULT_FIG)
    args = ap.parse_args()

    labels = [int(s) for s in args.sites.split(',')]
    tag = '-'.join(str(s) for s in labels)
    # cada conjunto de sitios en su subcarpeta: <data>/<tag>/ y <figures>/<tag>/
    indir = args.indir
    if os.path.isdir(os.path.join(indir, tag)):
        indir = os.path.join(indir, tag)
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    print(f'  datos:   {indir}\n  figuras: {outdir}')

    if args.all:
        if args.files:
            raise SystemExit('--all no admite archivos explicitos')
        hechos = 0
        for mu in COMPS:
            for nu in COMPS:
                if all(os.path.isfile(q) for q in _paths(indir, mu, nu, labels, tag)):
                    plot_mn(mu, nu, labels, indir, outdir)
                    hechos += 1
                else:
                    print(f'  (sin datos completos para {mu}{nu}, lo salto)')
        print(f'{hechos} combinaciones (mu,nu) graficadas')
    else:
        if args.mu is None or args.nu is None:
            raise SystemExit('hace falta --mu y --nu, o --all')
        plot_mn(args.mu, args.nu, labels, indir, outdir, args.files or None)


if __name__ == '__main__':
    main()
