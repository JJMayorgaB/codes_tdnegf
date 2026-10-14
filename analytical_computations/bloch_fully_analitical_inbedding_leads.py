#!/usr/bin/env python3
"""
bloch_fully_analitical_inbedding_leads.py -- animacion en la esfera de Bloch de la
trayectoria analitica de fully_analitical_inbedding_leads.jl (precesion en y +
Rashba en y, lead derecho). Equivale al setup D-F-F: el driver es la impureza
(sitio 0) y los sitios 1 y 2 del lead ocupan el lugar de los F, pero aqui lo que
se muestra en ellos es el espin ELECTRONICO <sigma>_i(t), no un LMM.

Panel de arriba: esfera de Bloch unitaria con todos los vectores desde el origen
    - driver M_D(t) = (sin th sin Om t, cos th, sin th cos Om t), en rojo
    - sitios 1 y 2: direccion <sigma>_i / |<sigma>_i|, con la flecha y la estela
      coloreadas por la magnitud |<sigma>_i(t)| (colorbar comun); estela con
      circulos (sitio 1) y triangulos (sitio 2)
Panel de abajo: <sigma^x>_i(t) sin normalizar (eje izquierdo) y M_D^x(t) (eje
derecho, rojo discontinuo), con ventana movil y cursor.

Uso:
    python bloch_fully_analitical_inbedding_leads.py
    python bloch_fully_analitical_inbedding_leads.py --theta 10 --Omega 0.005 --frame-skip 2
Lee output/fa_inbedding_rho_t.csv y escribe output/fa_inbedding_bloch.mp4.
"""

import argparse
import os
from collections import deque

import numpy as np
import pandas as pd
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_COLOR = '#c1272d'
SITE_LINE = ('#1f4e9c', '#2a9d5c')          # solo para el panel de abajo
SITE_MARK = ('o', '^')
TRAIL = 120
CMAP = cm.viridis


def leer(path):
    d = pd.read_csv(path)
    d = d[d['lead'] == 'R']
    sites = sorted(int(s) for s in d['site'].unique())[:2]
    if len(sites) < 2:
        raise SystemExit(f'{path}: se necesitan los sitios 0 y 1 (sitios 1 y 2 del lead)')
    S = []
    for s in sites:
        ds = d[d['site'] == s].sort_values('t')
        S.append(ds[['sx', 'sy', 'sz']].to_numpy())
    t = d[d['site'] == sites[0]].sort_values('t')['t'].to_numpy()
    return t, np.stack(S, axis=1)               # (Nt, 2, 3)


def esfera(ax):
    u, v = np.linspace(0, 2 * np.pi, 49), np.linspace(0, np.pi, 25)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)), color='0.85', lw=0.4,
                      rstride=4, cstride=4)
    ph = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(ph), np.sin(ph), 0 * ph, color='0.6', lw=0.8)
    ax.plot(np.cos(ph), 0 * ph, np.sin(ph), color='0.75', lw=0.6)
    ax.plot(0 * ph, np.cos(ph), np.sin(ph), color='0.75', lw=0.6)
    for e, lab in ((np.eye(3)[0], 'x'), (np.eye(3)[1], 'y'), (np.eye(3)[2], 'z')):
        ax.plot([-1.15 * e[0], 1.15 * e[0]], [-1.15 * e[1], 1.15 * e[1]],
                [-1.15 * e[2], 1.15 * e[2]], color='0.5', lw=0.8)
        ax.text(1.25 * e[0], 1.25 * e[1], 1.25 * e[2], lab, fontsize=13, ha='center', va='center')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser(description='Esfera de Bloch del espin electronico en los sitios 1 y 2.')
    ap.add_argument('--rho-csv', default=os.path.join(HERE, 'output', 'fa_inbedding_rho_t.csv'))
    ap.add_argument('--Omega', type=float, default=0.005)
    ap.add_argument('--theta', type=float, default=10.0, help='angulo del cono del driver (grados)')
    ap.add_argument('--frame-skip', type=int, default=1, help='cada cuantas filas del CSV un cuadro')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--dpi', type=int, default=110)
    ap.add_argument('--window', type=float, default=1.0,
                    help='ancho (en periodos) de la ventana movil del panel inferior')
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'fa_inbedding_bloch.mp4'))
    args = ap.parse_args()

    t, S = leer(args.rho_csv)
    T = 2 * np.pi / args.Omega
    th = np.radians(args.theta)
    MD = np.stack([np.sin(th) * np.sin(args.Omega * t), np.cos(th) * np.ones_like(t),
                   np.sin(th) * np.cos(args.Omega * t)], axis=1)
    mag = np.linalg.norm(S, axis=2)                       # (Nt, 2)
    dirs = S / np.where(mag > 0, mag, 1.0)[:, :, None]
    norm = colors.Normalize(vmin=mag.min(), vmax=mag.max())
    frames = np.arange(0, len(t), args.frame_skip)

    fig = plt.figure(figsize=(6.2, 8.0))
    ax3 = fig.add_subplot(2, 1, 1, projection='3d')
    esfera(ax3)
    ax3.text2D(0.0, 0.02, r'rojo: $\mathbf{M}_D$;   1, 2: $\langle\hat{\boldsymbol{\sigma}}\rangle_{i}/|\langle\hat{\boldsymbol{\sigma}}\rangle_{i}|$',
               transform=ax3.transAxes, fontsize=11)
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=ax3, shrink=0.6, pad=0.08)
    cb.set_label(r'$|\langle\hat{\boldsymbol{\sigma}}\rangle_{i}|$', fontsize=13)

    quiv = [None] * 3                                     # driver, sitio 1, sitio 2
    labels = [None] * 2
    tr_D = ax3.plot([], [], [], '-', color=DRIVER_COLOR, lw=1.2, alpha=0.8)[0]
    tr_S = [ax3.scatter([], [], [], s=6, marker=SITE_MARK[i], depthshade=False) for i in range(2)]
    for sc in tr_S:                                       # la estela usa la misma escala que la colorbar
        sc.set_cmap(CMAP); sc.set_norm(norm)
    buf_D = deque(maxlen=TRAIL)
    buf_S = [deque(maxlen=TRAIL) for _ in range(2)]

    ax2 = fig.add_subplot(2, 1, 2)
    ln = [ax2.plot([], [], '-', color=SITE_LINE[i], lw=1.4,
                   label=rf'$\langle\hat\sigma^x\rangle_{{{i + 1}}}$')[0] for i in range(2)]
    amp = np.abs(S[:, :, 0]).max()
    ax2.set_ylim(-1.15 * amp, 1.15 * amp if amp > 0 else 1)
    ax2.set_xlabel(r'Time $(2\pi/\Omega)$'); ax2.set_ylabel(r'$\langle\hat\sigma^x\rangle_i(t)$')
    axr = ax2.twinx()
    lnD, = axr.plot([], [], '--', color=DRIVER_COLOR, lw=1.2, label=r'$M_D^x$')
    axr.set_ylim(-1.15 * np.sin(th), 1.15 * np.sin(th))
    axr.set_ylabel(r'$M_D^x(t)$', color=DRIVER_COLOR)
    cursor = ax2.axvline(0, color='0.4', lw=0.8, ls=':')
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=11, frameon=False, ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)                         # espacio para el titulo con el tiempo

    tt = t / T

    def update(fi):
        i = frames[fi]
        for q in quiv:
            if q is not None:
                q.remove()
        for lb in labels:
            if lb is not None:
                lb.remove()
        v = MD[i]
        quiv[0] = ax3.quiver(0, 0, 0, *v, color=DRIVER_COLOR, lw=2.2, arrow_length_ratio=0.12)
        buf_D.append(tuple(v)); p = np.array(buf_D)
        tr_D.set_data(p[:, 0], p[:, 1]); tr_D.set_3d_properties(p[:, 2])
        for s in range(2):
            u = dirs[i, s]
            col = CMAP(norm(mag[i, s]))
            quiv[s + 1] = ax3.quiver(0, 0, 0, *u, color=col, lw=2.2, arrow_length_ratio=0.12)
            labels[s] = ax3.text(*(1.12 * u), f'{s + 1}', fontsize=12, color='k',
                                 ha='center', va='center')
            buf_S[s].append((*u, mag[i, s])); q = np.array(buf_S[s])
            tr_S[s]._offsets3d = (q[:, 0], q[:, 1], q[:, 2])
            tr_S[s].set_array(q[:, 3])
        lo = max(tt[frames[0]], tt[i] - args.window)
        w = (tt >= lo) & (tt <= tt[i])
        for s in range(2):
            ln[s].set_data(tt[w], S[w, s, 0])
        lnD.set_data(tt[w], MD[w, 0])
        ax2.set_xlim(lo, max(lo + args.window, tt[i]))
        cursor.set_xdata([tt[i], tt[i]])
        fig.suptitle(rf'$t = {tt[i]:.3f}\ T$', fontsize=15)
        return ln + [lnD, tr_D, cursor] + tr_S

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000)
    print(f'{len(frames)} cuadros -> {args.out}')
    ani.save(args.out, writer=writer, dpi=args.dpi)
    print('listo')


if __name__ == '__main__':
    main()
