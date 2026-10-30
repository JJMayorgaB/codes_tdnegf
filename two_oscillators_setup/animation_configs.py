#!/usr/bin/env python3
"""
animation_configs.py -- animaciones que comparan las configuraciones de los F para
un mismo conjunto de parametros (corridas de run_config.jl / run_all.sh).
(animation.py se conserva para comparar corridas arbitrarias, p. ej. y contra z.)

Para cada conjunto <param_tag> hace tres videos en cada uno de los dos estilos:
    animations/<eje>_precession/DFF/<param_tag>.mp4       columnas F1-2 | F3-6 | F5-9
    animations/<eje>_precession/FDF/<param_tag>.mp4       columnas Fm1-1 | Fm3-3 | Fm4-4
    animations/<eje>_precession/DFF_FDF/<param_tag>.mp4   las 6: DFF arriba, FDF abajo
    (y los mismos con bloch_<param_tag>.mp4 para el estilo de esfera de Bloch)
Cada configuracion tiene dos paneles:
    arriba : estilo chain (como animation.py): vista 3D de la cadena, un vector
             M_m por LMM sobre su sitio i, con estela de la punta (D = driver en
             rojo, F = libres en azul/verde); puntos grises chicos en los sitios
             desnudos entre los LMMs. La vista abarca su propia cadena, con la
             misma escala en i, y, z (las flechas no se deforman).
             estilo bloch (como animation_bloch_sphere.py): los 3 M_m desde el
             origen sobre la misma esfera de Bloch, con la estela de sus puntas.
    abajo  : M^x_m(t) de los 3 LMMs con un cursor en el tiempo actual

Uso:
    python animation_configs.py gso0p1_jsd0p2_th10deg_Om0p005_buf4_adyn0p0
    python animation_configs.py <param_tag> --which DFF,FDF --frame-skip 10 --t-min 6283
    python animation_configs.py <param_tag> --style bloch      # solo la esfera de Bloch
    python animation_configs.py                  # todos los conjuntos que encuentre
Un video que ya existe no se rehace (salvo con --overwrite); si falta alguna de
sus corridas, se omite con un aviso.
"""

import argparse
import os
import re
from collections import deque

import numpy as np
import pandas as pd
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib import animation

HERE = os.path.dirname(os.path.abspath(__file__))
ROLE_COLOR = {'D': '#c1272d'}
FREE_COLORS = ('#1f4e9c', '#2a9d5c')
TRAIL = {'chain': 60, 'bloch': 120}
ARROW = {'chain': 0.8, 'bloch': 1.0}
PREFIX = {'chain': '', 'bloch': 'bloch_'}

CONFIGS = {'DFF': ('F1-2', 'F3-6', 'F5-9'),
           'FDF': ('Fm1-1', 'Fm3-3', 'Fm4-4')}
VIDEOS = {'DFF': ('DFF',), 'FDF': ('FDF',), 'DFF_FDF': ('DFF', 'FDF')}


def leer(run, layout, sep):
    geo = pd.read_csv(os.path.join(run, 'geometry.csv'))
    txt = open(os.path.join(run, 'params.txt'), encoding='utf-8').read()
    Omega = float(re.search(r'Ω = ([0-9.eE+-]+)', txt).group(1))
    d = pd.read_csv(os.path.join(run, 'spins_t.csv'))
    ms = list(geo['m'])
    M = np.stack([d[[f'M{m}_x', f'M{m}_y', f'M{m}_z']].to_numpy() for m in ms], axis=1)
    colores, k = [], 0
    for r in geo['role']:
        if r == 'D':
            colores.append(ROLE_COLOR['D'])
        else:
            colores.append(FREE_COLORS[k % 2]); k += 1
    return dict(t=d['t'].to_numpy(), M=M, sites=geo['site'].to_numpy(), roles=list(geo['role']),
                colors=colores, T=2 * np.pi / Omega, label=f'{layout}  {sep}')


def _esfera(ax3):
    """Esfera de Bloch de animation_bloch_sphere.py: malla tenue, ecuador, meridianos y ejes."""
    u, v = np.linspace(0, 2 * np.pi, 49), np.linspace(0, np.pi, 25)
    ax3.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v)), color='0.85', lw=0.4,
                       rstride=4, cstride=4)
    ph = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(ph), np.sin(ph), 0 * ph, color='0.6', lw=0.8)
    ax3.plot(np.cos(ph), 0 * ph, np.sin(ph), color='0.75', lw=0.6)
    ax3.plot(0 * ph, np.cos(ph), np.sin(ph), color='0.75', lw=0.6)
    for e, lab in ((np.eye(3)[0], 'x'), (np.eye(3)[1], 'y'), (np.eye(3)[2], 'z')):
        ax3.plot([-1.15 * e[0], 1.15 * e[0]], [-1.15 * e[1], 1.15 * e[1]],
                 [-1.15 * e[2], 1.15 * e[2]], color='0.5', lw=0.8)
        ax3.text(1.25 * e[0], 1.25 * e[1], 1.25 * e[2], lab, fontsize=13, ha='center', va='center')
    ax3.set_xlim(-1, 1); ax3.set_ylim(-1, 1); ax3.set_zlim(-1, 1)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_axis_off()


def columna(fig, gs_top, gs_bot, d, sel, window, style):
    """Paneles de una configuracion; devuelve el estado que usa update()."""
    ax3 = fig.add_subplot(gs_top, projection='3d')
    nm = len(d['sites'])
    if style == 'bloch':
        xs = np.zeros(nm)                                   # todos desde el origen
        _esfera(ax3)
        ax3.set_title(d['label'], fontsize=14)
    else:
        xs = d['sites'].astype(float)
        _cadena(ax3, xs, d)
    lw = 1.2 if style == 'bloch' else 1.0
    alpha = 0.8 if style == 'bloch' else 0.7
    trails = [ax3.plot([], [], [], '-', color=d['colors'][m], lw=lw, alpha=alpha)[0] for m in range(nm)]

    ax2 = fig.add_subplot(gs_bot)
    lines = [ax2.plot([], [], '-', color=d['colors'][m], lw=1.4,
                      label=rf'm={m + 1} ({d["roles"][m]})')[0] for m in range(nm)]
    cursor = ax2.axvline(0, color='0.4', lw=0.8, ls=':')
    amp = np.abs(d['M'][sel][:, :, 0]).max()
    ax2.set_ylim(-1.1 * amp, 1.1 * amp if amp > 0 else 1)
    ax2.set_xlabel(r'Time $(2\pi/\Omega)$'); ax2.set_ylabel(r'$M^x_m(t)$')
    ax2.legend(loc='upper right', fontsize=11, frameon=False)
    return dict(ax3=ax3, xs=xs, quivers=[None] * nm, trails=trails,
                buf=[deque(maxlen=TRAIL[style]) for _ in range(nm)], ax2=ax2, lines=lines,
                cursor=cursor, d=d, window=window, style=style)


def _cadena(ax3, xs, d):
    """Vista 3D de la cadena de animation.py, con los sitios desnudos entre los LMMs."""
    x0, x1 = xs.min() - 1, xs.max() + 1
    ax3.set_xlim(x0, x1); ax3.set_ylim(-1, 1); ax3.set_zlim(-1, 1)
    ax3.set_box_aspect((x1 - x0, 2, 2), zoom=1.0)
    ax3.set_xlabel('i'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
    ax3.set_xticks(np.arange(int(x0) + 1, int(x1)))
    ax3.set_yticks([-1, 0, 1]); ax3.set_zticks([-1, 0, 1])
    bare = np.setdiff1d(np.arange(int(xs.min()), int(xs.max()) + 1), d['sites'])
    ax3.plot(bare, 0 * bare, 0 * bare, 'o', color='0.7', ms=3)
    ax3.plot(xs, 0 * xs, 0 * xs, 'o', color='0.4', ms=5)
    ax3.set_title(d['label'], fontsize=14)


def video(runs, nrow, ncol, out_path, args, tag, style):
    """runs: lista de filas, cada fila una lista de (run, layout, sep)."""
    D = [[leer(*r) for r in fila] for fila in runs]
    t = D[0][0]['t']
    for d in (x for fila in D for x in fila):
        if len(d['t']) != len(t) or np.abs(d['t'] - t).max() > 1e-6:
            raise SystemExit('las corridas no comparten la malla temporal (mismo t_final y OUT_STRIDE)')
    sel = np.ones(len(t), bool)
    if args.t_min is not None: sel &= t >= args.t_min
    if args.t_max is not None: sel &= t <= args.t_max
    frames = np.flatnonzero(sel)[::args.frame_skip]
    if len(frames) == 0:
        raise SystemExit('no hay tiempos en el rango pedido')
    T = D[0][0]['T']

    fig = plt.figure(figsize=(5.0 * ncol, 4.0 * 2 * nrow))
    gs = fig.add_gridspec(2 * nrow, ncol)
    cols = [columna(fig, gs[2 * r, c], gs[2 * r + 1, c], d, sel, args.window, style)
            for r, fila in enumerate(D) for c, d in enumerate(fila)]
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    def update(fi):
        i = frames[fi]
        out = []
        for st in cols:
            d, xs, sty = st['d'], st['xs'], st['style']
            a = ARROW[sty]
            lwq, alr = (2.2, 0.12) if sty == 'bloch' else (2.0, 0.15)
            for m in range(len(xs)):
                v = d['M'][i, m]
                if st['quivers'][m] is not None:
                    st['quivers'][m].remove()
                st['quivers'][m] = st['ax3'].quiver(xs[m], 0, 0, a * v[0], a * v[1], a * v[2],
                                                    color=d['colors'][m], lw=lwq, arrow_length_ratio=alr)
                st['buf'][m].append((xs[m] + a * v[0], a * v[1], a * v[2]))
                p = np.array(st['buf'][m])
                st['trails'][m].set_data(p[:, 0], p[:, 1]); st['trails'][m].set_3d_properties(p[:, 2])
            tt = d['t'] / T
            lo = max(tt[frames[0]], tt[i] - st['window'])
            w = (tt >= lo) & (tt <= tt[i])
            for m, ln in enumerate(st['lines']):
                ln.set_data(tt[w], d['M'][w, m, 0])
            st['ax2'].set_xlim(lo, max(lo + st['window'], tt[i]))
            st['cursor'].set_xdata([tt[i], tt[i]])
            out += st['lines'] + st['trails'] + [st['cursor']]
        fig.suptitle(rf'{tag}      $t = {t[i] / T:.3f}\ T$', fontsize=16)
        return out

    if args.frame is not None:                       # un solo cuadro de prueba (png)
        update(min(args.frame, len(frames) - 1))
        png = out_path[:-4] + '_frame.png'
        fig.savefig(png, dpi=args.dpi)
        print(f'  cuadro de prueba -> {png}')
        plt.close(fig)
        return
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    print(f'  {len(frames)} cuadros -> {out_path}')
    ani.save(out_path, writer=writer, dpi=args.dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Animaciones por conjunto de parametros: DFF, FDF y las 6.')
    ap.add_argument('tags', nargs='*', help='param_tag (sin nada: todos los que encuentre)')
    ap.add_argument('--axis', default='y', help='eje de precesion (carpeta <eje>_precession)')
    ap.add_argument('--which', default='DFF,FDF,DFF_FDF', help='videos a hacer')
    ap.add_argument('--style', default='chain,bloch',
                    help='chain (cadena, como animation.py) y/o bloch (esfera, como animation_bloch_sphere.py)')
    ap.add_argument('--t-min', type=float, default=None)
    ap.add_argument('--t-max', type=float, default=None)
    ap.add_argument('--frame-skip', type=int, default=5, help='cada cuantas filas del CSV un cuadro')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--dpi', type=int, default=110)
    ap.add_argument('--window', type=float, default=1.0,
                    help='ancho (en periodos) de la ventana movil del panel inferior')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--frame', type=int, default=None,
                    help='solo guarda ese cuadro como png (prueba rapida del diseno)')
    args = ap.parse_args()

    root = os.path.join(HERE, f'{args.axis}_precession', 'output')
    tags = args.tags
    if not tags:
        tags = sorted({t for lay, seps in CONFIGS.items() for s in seps
                       if os.path.isdir(os.path.join(root, lay, s))
                       for t in os.listdir(os.path.join(root, lay, s))})
    which = [w.strip() for w in args.which.split(',') if w.strip()]
    styles = [w.strip() for w in args.style.split(',') if w.strip()]
    for st in styles:
        if st not in PREFIX:
            raise SystemExit(f'estilo desconocido: {st} (chain o bloch)')
    for tag in tags:
        print(f'== {tag}')
        for w, sty in ((w, sty) for w in which for sty in styles):
            out_path = os.path.join(HERE, 'animations', f'{args.axis}_precession', w,
                                    f'{PREFIX[sty]}{tag}.mp4')
            if os.path.exists(out_path) and not args.overwrite and args.frame is None:
                print(f'  ya existe: {out_path}'); continue
            runs = [[(os.path.join(root, lay, s, tag), lay, s) for s in CONFIGS[lay]] for lay in VIDEOS[w]]
            falta = [r for fila in runs for r, _, _ in fila
                     if not os.path.isfile(os.path.join(r, 'spins_t.csv'))]
            if falta:
                print(f'  AVISO: {w} omitido, faltan corridas:\n    ' + '\n    '.join(falta)); continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            video(runs, len(runs), len(runs[0]), out_path, args, tag, sty)
    print('listo')


if __name__ == '__main__':
    main()
