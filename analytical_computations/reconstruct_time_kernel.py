#!/usr/bin/env python3
"""
reconstruct_time_kernel.py -- reconstruye los planos chi^{mu nu}_{ij}(t,t') a
partir de los armonicos que guarda kernel_harmonics.jl,

    chi(t,t') = sum_{k=-2..2} e^{-i k Omega t} chi_k(t - t'),

y saca las mismas figuras que plot_time_kernel.py (4x4 en (i,j), Re e Im,
barra de color global en el panel (1,1)), pero con la malla de los armonicos.

Ventana: (t,t') in [t0, t0 + W]^2, con paso dt = dtau y W = tau_max por
defecto (--window para una mas chica). Si la ventana tiene mas puntos que
pixeles en la figura (--max-pixels), se promedian bloques de la malla fina en
vez de submuestrear, para no volver a meter aliasing. tau_max es la mas grande posible: en ese cuadrado |t - t'| <= tau_max, y todos los tau = t - t' caen
exactos sobre la malla de los armonicos, asi que la reconstruccion es exacta
(sin interpolar). La causalidad NO se impone: el triangulo t' > t usa los
chi_k(tau < 0) calculados.

Uso:
    # 1 periodo, SIN promediar (a 600 dpi caben ~1650 unidades de tiempo sin aliasing)
    python reconstruct_time_kernel.py --all --sites 1,2,3,4 --window 1257 --max-pixels 100000 --dpi 600
    # ventana completa [0,3T], promediada por bloques (vista general)
    python reconstruct_time_kernel.py --all --sites 1,2,3,4
    python reconstruct_time_kernel.py --all --sites 1,7,13,20 --t0 0.5    # arranca a mitad de periodo
    # solo triangulo inferior, sin la franja |t-t'| < 3 (cola lejos de la diagonal)
    python reconstruct_time_kernel.py --mu x --nu x --sites 1,2,3,4 --window 60 --tau-min 3
"""

import argparse
import gc
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import plot_time_kernel                    # noqa: E402
from plot_time_kernel import plot_parte    # noqa: E402  (mismo formato de figura)

DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'output', 'kernel_harmonics', 'full', 'data_3T')
DEFAULT_FIG = os.path.join(SCRIPT_DIR, 'output', 'kernel_harmonics', 'full', 'maps_3T_avg')
COMPS = ('x', 'y', 'z')


def reconstruir(A, tau, Omega, t, f=1, filas=256, parte=None):
    """
    A[tau, k+2, i_idx, j_idx] (k = -2..2)  ->  chi[(c, r)] con chi[a, b] = chi(t_a, t'_b).
    t es la malla (t_a = t'_a) con el mismo paso que tau.

    f > 1: promedia bloques f x f de la malla fina (coarse-graining). Se usa
    cuando la ventana tiene mas puntos que pixeles en la figura: dibujar la
    malla fina punto a punto seria volver a submuestrear (aliasing); el
    promedio por bloques muestra el contenido de baja frecuencia sin inventar
    patrones. Se calcula por tandas de filas para no llenar la memoria.

    parte = 'Re' o 'Im': guarda solo esa parte en float32 (4x menos RAM que
    complex128); es lo que usa main para graficar. None: complejo completo.
    """
    dtau = tau[1] - tau[0]
    i0 = int(np.argmin(np.abs(tau)))
    if abs(tau[i0]) > 1e-9 * dtau:
        raise SystemExit('la malla tau no contiene tau = 0')
    N = len(t) - (len(t) % f)                                  # multiplo de f
    if i0 - (N - 1) < 0 or i0 + (N - 1) >= len(tau):
        raise SystemExit('la malla tau no cubre |t - t\'| en la ventana; '
                         '¿datos viejos sin tau < 0? corre de nuevo kernel_harmonics.jl')
    ks = np.arange(-2, 3)
    E = np.exp(-1j * np.outer(t[:N], ks) * Omega)             # (N, 5): e^{-ik Omega t_a}
    n = A.shape[2]
    Nb = N // f
    filas = max(f, (filas // f) * f)
    out = {}
    for c in range(n):
        for r in range(n):
            Akr = A[:, :, c, r]
            Z = np.empty((Nb, Nb), complex if parte is None else np.float32)
            for a0 in range(0, N, filas):
                a = np.arange(a0, min(a0 + filas, N))
                idx = i0 + a[:, None] - np.arange(N)[None, :]
                blk = np.einsum('abk,ak->ab', Akr[idx], E[a])  # (len(a), N)
                if f > 1:
                    blk = blk.reshape(len(a) // f, f, Nb, f).mean(axis=(1, 3))
                if parte is not None:
                    blk = blk.real if parte == 'Re' else blk.imag
                Z[a0 // f:(a0 + len(a)) // f] = blk
            out[(c, r)] = Z
    return out


def main():
    ap = argparse.ArgumentParser(description='chi(t,t\') reconstruido desde los armonicos chi_k(tau).')
    ap.add_argument('--mu', choices=COMPS)
    ap.add_argument('--nu', choices=COMPS)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--sites', default='1,2,3,4')
    ap.add_argument('--indir', default=DEFAULT_DIR)
    ap.add_argument('--outdir', default=DEFAULT_FIG)
    ap.add_argument('--window', type=float, default=None,
                    help='largo de la ventana (unidades de tiempo, <= tau_max); por defecto tau_max')
    ap.add_argument('--max-pixels', type=int, default=1500,
                    help='si la ventana tiene mas puntos por eje, se promedian bloques '
                         'para quedar <= este numero (defecto 1500, ~ pixeles del panel)')
    ap.add_argument('--dpi', type=int, default=300,
                    help='dpi de las figuras (600 para ventanas de ~1T sin promediar)')
    ap.add_argument('--tau-min', type=float, default=None,
                    help="oculta la franja |t - t'| < tau_min y el triangulo t' > t; la escala de "
                         "color se calcula solo con lo que queda (para ver la cola lejos de la diagonal)")
    ap.add_argument('--formats', default='jpg,svg,pdf',
                    help='formatos de salida separados por coma (p.ej. jpg para los mapas de 1T a 600 dpi)')
    ap.add_argument('--skip-existing', action='store_true',
                    help='salta las (mu,nu) cuyas figuras Re e Im ya existen (para retomar)')
    ap.add_argument('--t0', type=float, default=0.0,
                    help='inicio de la ventana, en periodos del drive (defecto 0)')
    args = ap.parse_args()
    plot_time_kernel.DPI = args.dpi
    plot_time_kernel.FORMATS = tuple(args.formats.split(','))

    labels = [int(s) for s in args.sites.split(',')]
    tag = '-'.join(str(s) for s in labels)
    indir = args.indir
    if os.path.isdir(os.path.join(indir, tag)):
        indir = os.path.join(indir, tag)
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(indir, f'kh_{tag}_meta.json')) as fh:
        meta = json.load(fh)
    tau = np.load(os.path.join(indir, f'kh_{tag}_tau.npy'))
    Omega = meta['Omega']
    T = 2 * np.pi / Omega
    dtau = tau[1] - tau[0]
    W = tau.max() if args.window is None else min(args.window, tau.max())
    N = int(round(W / dtau)) + 1
    t = args.t0 * T + dtau * np.arange(N)
    f = int(np.ceil(N / args.max_pixels))
    Nb = (N - N % f) // f
    tb = t[:Nb * f].reshape(Nb, f).mean(axis=1)                # centro de cada bloque
    print(f'  datos:   {indir}\n  figuras: {outdir}')
    print(f'  ventana (t,t\') in [{t[0]:.2f}, {t[-1]:.2f}]^2 = [{t[0]/T:.4f}, {t[-1]/T:.4f}] T,'
          f'  dt = {dtau:.4g}, {N} x {N} puntos')
    if f > 1:
        print(f'  promedio por bloques {f}x{f} (ancho {f*dtau:.3g}) -> {Nb} x {Nb} en la figura')

    mns = [(mu, nu) for mu in COMPS for nu in COMPS] if args.all else [(args.mu, args.nu)]
    if None in mns[0]:
        raise SystemExit('hace falta --mu y --nu, o --all')
    for mu, nu in mns:
        fn = os.path.join(indir, f'kh_{mu}{nu}_{tag}.npy')
        if not os.path.isfile(fn):
            print(f'  (no hay {os.path.basename(fn)}, lo salto)')
            continue
        if args.skip_existing and all(os.path.isfile(os.path.join(outdir, f'time_kernel_{mu}{nu}_{tag}_{p}.{e}'))
                                      for p in ('Re', 'Im') for e in plot_time_kernel.FORMATS):
            print(f'  chi^{mu}{nu}: figuras ya existen, lo salto')
            continue
        A = np.load(fn)
        for parte in ('Re', 'Im'):                 # una parte a la vez (float32): menos RAM
            R = reconstruir(A, tau, Omega, t, f, parte=parte)
            if args.tau_min is not None:           # solo t - t' >= tau_min
                oculto = (tb[:, None] - tb[None, :]) < args.tau_min
                for v in R.values():
                    v[oculto] = np.nan
            data = {(labels[c], labels[r]): v for (c, r), v in R.items()}
            print(f'  chi^{mu}{nu}: max|{parte}| = {max(np.nanmax(np.abs(v)) for v in data.values()):.3e}')
            plot_parte(data, tb / T, labels, mu, nu, parte, outdir, tag)
            del R, data                            # liberar antes de la siguiente figura
            gc.collect()


if __name__ == '__main__':
    main()
