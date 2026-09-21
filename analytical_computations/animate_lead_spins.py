#!/usr/bin/env python3
"""
Animacion estilo quiver de los espines electronicos del lead, a partir de
inbedding_rho_t.csv (generado por sweep_leads en inbedding_leads.jl).

DIFERENCIA IMPORTANTE con animate_oscillators.py: alla los espines son
clasicos y unitarios, |S|=1, asi que se dibujan crudos. Aca son espines
ELECTRONICOS, |<sigma>| ~ 1e-2 en la superficie y cayendo un factor ~20 al
entrar al lead. Dibujados crudos, los sitios profundos son invisibles. Por eso
las flechas se normalizan a longitud unitaria (solo direccion) y la magnitud
se codifica en el COLOR, en escala logaritmica. Con --scale-mode raw se
dibujan sin normalizar, escalados por un factor comun.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import imageio_ffmpeg
import matplotlib
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib import animation, colors

HAS_LATEX = shutil.which('latex') is not None

plt.rcParams.update({
    'text.usetex': HAS_LATEX,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}',
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'] if HAS_LATEX else ['DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.axisbelow': True,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, 'output', 'inbedding_rho_t.csv')
CMAP = 'viridis'
TRACK_COLORS = ['#c1272d', '#2266aa', '#7e2bb6', '#2a9d5c']


def _fmt2d(ax, xlabel=None, ylabel=None):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=6.0, width=1.0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def load_lead(csv_path, lead, nsites, t_min=None):
    df = pd.read_csv(csv_path)
    df = df[df['lead'] == lead]
    if df.empty:
        raise SystemExit(f'no hay datos para el lead {lead} en {csv_path}')
    if t_min is not None:
        df = df[df['t'] >= t_min]
        if df.empty:
            raise SystemExit(f'no quedan datos con t >= {t_min}')
    sites = sorted(df['site'].unique())[:nsites]
    t = np.sort(df[df['site'] == sites[0]]['t'].unique())
    S = np.zeros((len(t), len(sites), 3))
    for j, n in enumerate(sites):
        s = df[df['site'] == n].sort_values('t')
        S[:, j, 0] = s['sx'].to_numpy()
        S[:, j, 1] = s['sy'].to_numpy()
        S[:, j, 2] = s['sz'].to_numpy()
    return t, np.array(sites), S


def build_figure(t, sites, S, Omega, scale_mode, lead):
    nsite = len(sites)
    T = 2 * np.pi / Omega

    # Coordenada comprimida para dibujar (igual que en animate_oscillators.py:
    # con los sitios reales la caja 3D sale larguisima). Los ticks muestran el
    # numero de sitio real.
    x = np.linspace(0.0, 12.0, nsite)
    x_lo, x_hi = x[0] - 1.0, x[-1] + 1.0

    mag = np.linalg.norm(S, axis=2)                      # (Nt, nsite)

    # color = magnitud, en log (abarca mas de una decada con la profundidad)
    vmax = float(mag.max())
    vmin = max(float(mag[mag > 0].min()) if np.any(mag > 0) else vmax / 100,
               vmax / 1e3)

    if scale_mode == 'raw':
        V = S / max(vmax, 1e-300)                        # escala comun
    else:
        # Direccion unitaria, PERO solo por encima del ruido numerico. En los
        # primeros cuadros rho(0)=0 y |<sigma>| ~ 1e-61 es basura de punto
        # flotante: dividir por eso da una flecha de tamano completo apuntando
        # al azar. El piso es precision de maquina RELATIVA al mayor valor de
        # la corrida, no el vmin del colorbar: asi solo desaparecen las flechas
        # que son ruido, y cualquier magnitud fisica por pequena que sea
        # (sitios profundos, nodos de la oscilacion 2k_F) se dibuja normal.
        noise = vmax * 1e-12
        shrink = np.minimum(mag, noise) / noise
        V = S / np.maximum(mag, 1e-300)[:, :, None] * shrink[:, :, None]
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(CMAP)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # sitios rastreados en los paneles 2D: superficie y tres mas, espaciados
    track_idx = sorted(set(np.linspace(0, nsite - 1, 4).astype(int).tolist()))

    fig = plt.figure(figsize=(13.0, 8.5))
    ax3d = fig.add_axes([0.02, 0.36, 0.60, 0.60], projection='3d')
    cax = fig.add_axes([0.645, 0.52, 0.011, 0.32])
    axtip = fig.add_axes([0.76, 0.46, 0.22, 0.40])
    axts = fig.add_axes([0.08, 0.06, 0.66, 0.26])

    # --- panel 3D: cadena de flechas -------------------------------------
    ax3d.set_xlim(x_lo, x_hi)
    ax3d.set_ylim(-1.1, 1.1)
    ax3d.set_zlim(-1.15, 1.15)
    ax3d.set_proj_type('ortho')
    ax3d.set_box_aspect((x_hi - x_lo, 2.2, 2.3))
    ax3d.view_init(elev=22, azim=-62)
    # El CSV guarda el sitio 0-based (n=0 es la superficie, convencion del
    # codigo Julia). Aqui se muestra 1-based y con el simbolo i, que es la
    # notacion del paper. El relabel es solo de presentacion.
    ax3d.set_xlabel('Lead site $i$', labelpad=20)
    ax3d.set_ylabel(r'$\langle\sigma^{y}_{i}\rangle$', labelpad=6)
    ax3d.set_zlabel(r'$\langle\sigma^{z}_{i}\rangle$', labelpad=2)
    step = max(1, nsite // 6)
    ax3d.set_xticks(x[::step])
    ax3d.set_xticklabels([str(s + 1) for s in sites[::step]], fontsize=9)
    ax3d.set_yticks([-1, 0, 1])
    ax3d.set_zticks([-1, 0, 1])
    unidad = 'direccion (norma 1)' if scale_mode == 'unit' else 'escala comun'
    ax3d.set_title(rf'Lead {lead}: electronic spins')

    # El texto que cambia cada cuadro va como texto plano, NO por LaTeX: si no,
    # es una llamada a latex.exe por cuadro (mismo motivo que en
    # animate_oscillators.py).
    dyn_text = fig.text(0.5, 0.995, '', ha='center', va='top', fontsize=13,
                        usetex=False, fontfamily='DejaVu Serif')

    ax3d.plot(x, np.zeros(nsite), np.zeros(nsite), color='0.8', lw=1.0, zorder=1)
    ax3d.scatter(x, np.zeros(nsite), np.zeros(nsite), s=16, color='#3a3a45',
                 edgecolors='white', linewidths=0.4, depthshade=False, zorder=2)
    for c, j in zip(TRACK_COLORS, track_idx):
        ax3d.scatter([x[j]], [0.0], [0.0], s=70, color=c, edgecolors='white',
                     linewidths=0.7, depthshade=False, zorder=3)
    quiver_holder = {'outline': None, 'fill': None}

    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_title(r'$|\langle\boldsymbol{\sigma}_{i}\rangle|$', pad=10, fontsize=11)
    cbar.ax.tick_params(direction='in', labelsize=9, pad=2)

    # --- panel de la punta del espin, plano (sigma_x, sigma_y) ------------
    axtip.set_aspect('equal')
    axtip.axhline(0, color='0.85', lw=0.8)
    axtip.axvline(0, color='0.85', lw=0.8)
    _fmt2d(axtip, xlabel=r'$\langle\sigma^{x}_{i}\rangle$',
           ylabel=r'$\langle\sigma^{y}_{i}\rangle$')
    lim = 1.15 * float(np.abs(S[:, track_idx, :2]).max())
    axtip.set_xlim(-lim, lim)
    axtip.set_ylim(-lim, lim)
    axtip.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
    axtip.xaxis.get_offset_text().set_fontsize(9)
    axtip.yaxis.get_offset_text().set_fontsize(9)
    tip_lines, tip_dots = {}, {}
    for c, j in zip(TRACK_COLORS, track_idx):
        (ln,) = axtip.plot([], [], '-', color=c, lw=1.0, alpha=0.6)
        (dot,) = axtip.plot([], [], 'o', color=c, ms=6)
        tip_lines[j], tip_dots[j] = ln, dot

    # --- panel temporal: <sigma_x>(t) de los sitios rastreados ------------
    for c, j in zip(TRACK_COLORS, track_idx):
        axts.plot(t / T, S[:, j, 0], '-', color=c, lw=1.2,
                  label=rf'$i={sites[j] + 1}$')
    axts.set_xlim((t / T).min(), (t / T).max())
    _fmt2d(axts, xlabel=r'$t\, (2\pi/\Omega)$',
           ylabel=r'$\langle\sigma^{x}_{i}\rangle(t)$')
    axts.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    axts.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                borderpad=0.4, handlelength=1.8, labelspacing=0.6, ncol=1,
                fontsize=16)
    ts_cursor = axts.axvline(t[0] / T, color='k', lw=1.5)

    def draw(idx):
        if quiver_holder['outline'] is not None:
            quiver_holder['outline'].remove()
        if quiver_holder['fill'] is not None:
            quiver_holder['fill'].remove()

        cols = cmap(norm(mag[idx, :]))
        # contorno oscuro primero, relleno de color encima y mas delgado, para
        # que las flechas claras no se pierdan sobre el fondo blanco
        quiver_holder['outline'] = ax3d.quiver(
            x, np.zeros(nsite), np.zeros(nsite),
            V[idx, :, 0], V[idx, :, 1], V[idx, :, 2],
            color='#333333', linewidth=3.4, arrow_length_ratio=0.22)
        quiver_holder['fill'] = ax3d.quiver(
            x, np.zeros(nsite), np.zeros(nsite),
            V[idx, :, 0], V[idx, :, 1], V[idx, :, 2],
            colors=cols, linewidth=1.9, arrow_length_ratio=0.20)

        for j in track_idx:
            tip_lines[j].set_data(S[:idx + 1, j, 0], S[:idx + 1, j, 1])
            tip_dots[j].set_data([S[idx, j, 0]], [S[idx, j, 1]])

        ts_cursor.set_xdata([t[idx] / T, t[idx] / T])
        dyn_text.set_text(f't/T = {t[idx]/T:.3f}')
        return ()

    return fig, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'output'))
    ap.add_argument('--lead', default='R')
    ap.add_argument('--nsites', type=int, default=20)
    ap.add_argument('--Omega', type=float, default=0.005)
    ap.add_argument('--scale-mode', choices=['unit', 'raw'], default='unit',
                    help='unit: flechas normalizadas (magnitud en el color). '
                         'raw: sin normalizar, escala comun.')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--frame-skip', type=int, default=2)
    ap.add_argument('--dpi', type=int, default=100)
    # TDNEGF escribe la evolucion completa (12567 tiempos = ~3.5 min de video).
    # Estos dos recortan la ventana ANTES de construir la figura, asi que los
    # limites de los ejes y el color se calculan solo con lo que se anima.
    corte = ap.add_mutually_exclusive_group()
    corte.add_argument('--t-min', type=float, default=None,
                       help='anima solo t >= t_min')
    corte.add_argument('--last-periods', type=float, default=None, metavar='N',
                       help='anima los ultimos N periodos: t_min = t_max - N*T')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --last-periods necesita t_max; se lee solo la columna t para no cargar
    # el CSV entero dos veces (TDNEGF escribe ~5e5 filas).
    t_min, tag = args.t_min, ''
    if args.last_periods is not None:
        t_max = pd.read_csv(args.csv, usecols=['t'])['t'].max()
        t_min = t_max - args.last_periods * 2 * np.pi / args.Omega
        tag = f'_last{args.last_periods:g}T'
    elif t_min is not None:
        tag = f'_tmin{t_min:g}'

    t, sites, S = load_lead(args.csv, args.lead, args.nsites, t_min=t_min)
    print(f'  lead {args.lead}: {len(sites)} sitios ({sites[0]}..{sites[-1]}), '
          f'{len(t)} pasos'
          + (f'  (t >= {t_min:.1f}, {(t[-1]-t[0])*args.Omega/(2*np.pi):.2f} periodos)'
             if t_min is not None else ''))
    mag = np.linalg.norm(S, axis=2)
    print(f'  |<sigma>|: max={mag.max():.3e}  min={mag[mag>0].min():.3e}  '
          f'(rango {mag.max()/max(mag[mag>0].min(),1e-300):.1f}x)')

    fig, draw = build_figure(t, sites, S, args.Omega, args.scale_mode, args.lead)
    frame_idx = list(range(0, len(t), args.frame_skip))

    def update(fi):
        return draw(frame_idx[fi])

    n_frames = len(frame_idx)
    print(f'  frames: {n_frames}, fps: {args.fps}, '
          f'duracion~{n_frames/args.fps:.1f}s')
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    out_path = os.path.join(args.outdir, f'anim_lead_spins_{args.lead}{tag}.mp4')
    # -pix_fmt yuv420p es obligatorio para que reproduzca en Windows (mismo
    # motivo que en animate_oscillators.py)
    writer = animation.FFMpegWriter(
        fps=args.fps, bitrate=2600, codec='libx264',
        extra_args=['-preset', 'veryfast', '-threads', '1',
                    '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.0',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
    ani.save(out_path, writer=writer, dpi=args.dpi)
    print(f'  -> {out_path}  ({n_frames} cuadros)')


if __name__ == '__main__':
    main()
