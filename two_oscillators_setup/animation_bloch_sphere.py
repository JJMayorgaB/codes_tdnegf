#!/usr/bin/env python3
"""
animation_bloch_sphere.py -- igual que animation.py, pero en el panel de arriba
los 3 LMMs se proyectan sobre la MISMA esfera de Bloch (todos desde el origen),
para ver directamente sus angulos relativos y el desfase de las precesiones.

Cada corrida es una columna:
    arriba : esfera de Bloch unitaria con los 3 vectores M_m y la estela de sus
             puntas sobre la superficie (D = driver en rojo, F = libres en azul/verde)
    abajo  : M^x_m(t) de los 3 LMMs con un cursor en el tiempo actual

Uso:
    python animation_bloch_sphere.py z_precession/output/setup1_<tag>
    python animation_bloch_sphere.py z_precession/output/setup1_<tag> z_precession/output/setup2_<tag> \
                        y-precession/output/setup1_<tag> y-precession/output/setup2_<tag> \
                        --t-min 6283 --frame-skip 10 --out comparacion_setups.mp4
Con una corrida el video queda en <corrida>/animation_bloch.mp4; con varias, en
two_oscillators_setup/animations/ (o donde diga --out).
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
TRAIL = 120


def leer(run):
    geo = pd.read_csv(os.path.join(run, 'geometry.csv'))
    txt = open(os.path.join(run, 'params.txt'), encoding='utf-8').read()
    Omega = float(re.search(r'Ω = ([0-9.eE+-]+)', txt).group(1))
    setup = re.search(r'setup = (\S+)', txt).group(1)
    axis = re.search(r'eje de precesion = (\S+)', txt).group(1)
    d = pd.read_csv(os.path.join(run, 'spins_t.csv'))
    M = np.stack([d[[f'M{m}_x', f'M{m}_y', f'M{m}_z']].to_numpy() for m in (1, 2, 3)], axis=1)
    colores, k = [], 0
    for r in geo['role']:
        if r == 'D':
            colores.append(ROLE_COLOR['D'])
        else:
            colores.append(FREE_COLORS[k % 2]); k += 1
    return dict(t=d['t'].to_numpy(), M=M, sites=geo['site'].to_numpy(), roles=list(geo['role']),
                colors=colores, T=2 * np.pi / Omega, label=f'{setup}, eje {axis}')


def main():
    ap = argparse.ArgumentParser(description='Animacion de los LMMs de los setups de dos osciladores.')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--t-min', type=float, default=None)
    ap.add_argument('--t-max', type=float, default=None)
    ap.add_argument('--frame-skip', type=int, default=5, help='cada cuantas filas del CSV un cuadro')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--dpi', type=int, default=110)
    ap.add_argument('--window', type=float, default=1.0,
                    help='ancho (en periodos) de la ventana movil del panel inferior')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    runs = [os.path.abspath(r) for r in args.runs]
    D = [leer(r) for r in runs]
    t = D[0]['t']
    for d in D[1:]:
        if len(d['t']) != len(t) or np.abs(d['t'] - t).max() > 1e-6:
            raise SystemExit('las corridas no comparten la malla temporal (mismo t_final y OUT_STRIDE)')
    sel = np.ones(len(t), bool)
    if args.t_min is not None: sel &= t >= args.t_min
    if args.t_max is not None: sel &= t <= args.t_max
    frames = np.flatnonzero(sel)[::args.frame_skip]
    if len(frames) == 0:
        raise SystemExit('no hay tiempos en el rango pedido')
    T = D[0]['T']

    nc = len(D)
    fig = plt.figure(figsize=(5.0 * nc, 8.0))
    arts = []
    for c, d in enumerate(D):
        ax3 = fig.add_subplot(2, nc, c + 1, projection='3d')
        # esfera de Bloch: malla tenue, ecuador, meridianos y ejes x, y, z
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
        ax3.set_title(d['label'], fontsize=14)
        xs = np.zeros(3)                                 # todos desde el origen
        quivers = [None] * 3
        trails = [ax3.plot([], [], [], '-', color=d['colors'][m], lw=1.2, alpha=0.8)[0] for m in range(3)]
        buf = [deque(maxlen=TRAIL) for _ in range(3)]

        ax2 = fig.add_subplot(2, nc, nc + c + 1)
        lines = [ax2.plot([], [], '-', color=d['colors'][m], lw=1.4,
                          label=rf'm={m + 1} ({d["roles"][m]})')[0] for m in range(3)]
        cursor = ax2.axvline(0, color='0.4', lw=0.8, ls=':')
        amp = np.abs(d['M'][sel][:, :, 0]).max()
        ax2.set_ylim(-1.1 * amp, 1.1 * amp if amp > 0 else 1)
        ax2.set_xlabel(r'Time $(2\pi/\Omega)$'); ax2.set_ylabel(r'$M^x_m(t)$')
        ax2.legend(loc='upper right', fontsize=11, frameon=False)
        arts.append((ax3, xs, quivers, trails, buf, ax2, lines, cursor, d))
    fig.tight_layout()

    def update(fi):
        i = frames[fi]
        out = []
        for ax3, xs, quivers, trails, buf, ax2, lines, cursor, d in arts:
            for m in range(3):
                v = d['M'][i, m]
                if quivers[m] is not None:
                    quivers[m].remove()
                quivers[m] = ax3.quiver(0, 0, 0, v[0], v[1], v[2],
                                        color=d['colors'][m], lw=2.2, arrow_length_ratio=0.12)
                buf[m].append((v[0], v[1], v[2]))           # estela sobre la superficie
                p = np.array(buf[m])
                trails[m].set_data(p[:, 0], p[:, 1]); trails[m].set_3d_properties(p[:, 2])
            tt = d['t'] / T
            lo = max(tt[frames[0]], tt[i] - args.window)
            w = (tt >= lo) & (tt <= tt[i])
            for m in range(3):
                lines[m].set_data(tt[w], d['M'][w, m, 0])
            ax2.set_xlim(lo, max(lo + args.window, tt[i]))
            cursor.set_xdata([tt[i], tt[i]])
            out += lines + trails + [cursor]
        fig.suptitle(rf'$t = {t[i] / T:.3f}\ T$', fontsize=16)
        return out

    if args.out is not None:
        out_path = os.path.abspath(args.out)
    elif nc == 1:
        out_path = os.path.join(runs[0], 'animation_bloch.mp4')
    else:
        os.makedirs(os.path.join(HERE, 'animations'), exist_ok=True)
        out_path = os.path.join(HERE, 'animations', 'bloch_compare_' + '__'.join(os.path.basename(r) for r in runs)[:180] + '.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    print(f'{len(frames)} cuadros -> {out_path}')
    ani.save(out_path, writer=writer, dpi=args.dpi)
    print('listo')


if __name__ == '__main__':
    main()
