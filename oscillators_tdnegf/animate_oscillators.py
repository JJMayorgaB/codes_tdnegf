#!/usr/bin/env python3

import argparse
import os
import re
from collections import deque

import numpy as np
import pandas as pd
import imageio_ffmpeg
import matplotlib
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
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
DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'output')

CMAP = 'seismic'
TRAIL_LEN = 20  # 3D-panel trail length, only for tracked spins

# Protocol time milestones (see oscillators.jl)
T_ON_G3 = 250.0   # group3 uniform precession turns on
T_ON_G1 = 1000.0  # group1 traveling wave turns on

# groups (spin index, 1-based) -- see oscillators.jl
G1 = list(range(1, 6))     # traveling wave
G3 = [11]                  # uniform precession (driver)
DRIVEN = G1 + G3

# tracked spins: (spin index 1-based, label, color)
TRACKED = [
    (1,  r'g1, site 2 (Left, Wave)',    '#7e2bb6'),
    (8,  r'g2, site 16 (Free)',         '#2266aa'),
    (11, r'g3, site 22 (Driver)',       '#c1272d'),
    (16, r'g4, site 32 (Right, Free)',  '#2a9d5c'),
]


def detect_spin_columns(columns):
    pat = re.compile(r'^S(\d+)_x_site(\d+)$')
    entries = []
    for c in columns:
        m = pat.match(c)
        if m:
            entries.append((int(m.group(1)), int(m.group(2))))
    entries.sort()
    if not entries:
        raise ValueError('No S{m}_x_site{site} columns found')
    return entries


def load_chain(csv_path):
    df = pd.read_csv(csv_path)
    entries = detect_spin_columns(df.columns)
    t = df['t'].to_numpy()
    spins = np.array([m for m, _ in entries], dtype=int)
    sites = np.array([site for _, site in entries], dtype=int)
    S = np.zeros((len(df), len(entries), 3))
    for j, (m, site) in enumerate(entries):
        S[:, j, 0] = df[f'S{m}_x_site{site}'].to_numpy()
        S[:, j, 1] = df[f'S{m}_y_site{site}'].to_numpy()
        S[:, j, 2] = df[f'S{m}_z_site{site}'].to_numpy()
    return t, spins, sites, S


def _fmt2d(ax, xlabel=None, ylabel=None):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                    left=True, right=True, size=6.0, width=1.0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _milestones(ax):
    ax.axvline(T_ON_G3, color='k', lw=1.0, ls=':', alpha=0.7)
    ax.axvline(T_ON_G1, color='k', lw=1.0, ls='--', alpha=0.7)


def build_figure(t, spins, sites, S, title_label):
    nspin = len(sites)
    # Compressed plotting coordinate (real sites span 2..32, which makes the
    # 3D box extremely elongated and the chain render as a thin sliver).
    # Ticks still show the real site numbers.
    x = np.linspace(0.0, 12.0, nspin)
    x_lo, x_hi = x[0] - 1.0, x[-1] + 1.0
    driven_idx = [j for j, m in enumerate(spins) if m in DRIVEN]

    # índice (0-based, en el orden de `spins`) de cada espín rastreado
    track_idx = [int(np.where(spins == m)[0][0]) for m, _, _ in TRACKED]

    vmax = max(1e-3, float(np.nanmax(np.abs(S[:, :, 0]))))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap(CMAP)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(13.0, 8.5))
    ax3d  = fig.add_axes([0.02, 0.34, 0.56, 0.62], projection='3d')
    cax   = fig.add_axes([0.605, 0.50, 0.010, 0.32])
    axtip = fig.add_axes([0.75, 0.46, 0.24, 0.42])
    axts  = fig.add_axes([0.08, 0.06, 0.68, 0.26])

    # --- 3D panel: chain of arrows + trails (tracked spins only) --------
    ax3d.set_xlim(x_lo, x_hi)
    ax3d.set_ylim(-1.1, 1.1)
    ax3d.set_zlim(-1.15, 1.15)
    ax3d.set_proj_type('ortho')
    ax3d.set_box_aspect((x_hi - x_lo, 2.2, 2.3))
    ax3d.view_init(elev=22, azim=-62)
    ax3d.set_xlabel('Electronic Site', labelpad=10)
    ax3d.set_ylabel(r'$S_y$', labelpad=6)
    ax3d.set_zlabel(r'$S_z$', labelpad=2)
    ax3d.set_xticks(x[::4])
    ax3d.set_xticklabels([str(s) for s in sites[::4]], fontsize=9)
    ax3d.set_yticks([-1, 0, 1])
    ax3d.set_zticks([-1, 0, 1])
    ax3d.set_title(title_label)

    # The "t=... | stage" part changes every frame; routing it through real
    # LaTeX (like set_title did before) means one latex.exe subprocess call
    # per frame, and MiKTeX intermittently chokes under that many rapid
    # calls in a row. Keep it as plain (non-LaTeX) text instead -- title_label
    # and every other label still render through real LaTeX, just once.
    dyn_text = fig.text(0.5, 0.995, '', ha='center', va='top', fontsize=13,
                         usetex=False, fontfamily='DejaVu Serif')

    ax3d.plot(x, np.zeros(nspin), np.zeros(nspin), color='0.8', lw=1.0, zorder=1)
    ax3d.scatter(x, np.zeros(nspin), np.zeros(nspin), s=16, color='#3a3a45',
                 edgecolors='white', linewidths=0.4, depthshade=False, zorder=2)
    for m, lab, col in TRACKED:
        j = int(np.where(spins == m)[0][0])
        ax3d.scatter([x[j]], [0.0], [0.0], s=70, color=col, edgecolors='white',
                     linewidths=0.7, depthshade=False, zorder=3)

    trails = {j: deque(maxlen=TRAIL_LEN) for j in track_idx}
    trail_collections = {}
    track_colors = {j: to_rgba(col) for (m, lab, col), j in
                     zip(TRACKED, track_idx)}
    for j in track_idx:
        lc = Line3DCollection([], linewidths=[], colors=[])
        ax3d.add_collection3d(lc, autolim=False)
        trail_collections[j] = lc
    quiver_holder = {'outline': None, 'fill': None}

    cbar = fig.colorbar(sm, cax=cax, format='%.2f')
    cbar.ax.set_title(r'$S_x$', pad=10, fontsize=12)
    cbar.ax.tick_params(direction='in', labelsize=9, pad=2)

    # --- spin-tip panel in the (S_x,S_y) plane -----------------------------
    axtip.set_xlim(-1.05, 1.05)
    axtip.set_ylim(-1.05, 1.05)
    axtip.set_aspect('equal')
    axtip.axhline(0, color='0.85', lw=0.8)
    axtip.axvline(0, color='0.85', lw=0.8)
    _fmt2d(axtip, xlabel=r'$S_x$', ylabel=r'$S_y$')
    tip_lines = {}
    tip_dots = {}
    for m, lab, col in TRACKED:
        j = int(np.where(spins == m)[0][0])
        (ln,) = axtip.plot([], [], '-', color=col, lw=1.0, alpha=0.6)
        (dot,) = axtip.plot([], [], 'o', color=col, ms=6)
        tip_lines[j] = ln
        tip_dots[j] = dot

    # --- S_x^i(t) panel for the tracked spins ------------------------------
    ts_lines = {}
    for m, lab, col in TRACKED:
        j = int(np.where(spins == m)[0][0])
        (ln,) = axts.plot(t, S[:, j, 0], '-', color=col, lw=1.2, label=lab)
        ts_lines[j] = ln
    _milestones(axts)
    axts.set_xlim(t.min(), t.max())
    axts.set_ylim(-1.05, 1.05)
    _fmt2d(axts, xlabel=r'$t\ (\hbar/\gamma)$', ylabel=r'$S_x^i(t)$')
    axts.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
                loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                borderpad=0.4, handlelength=1.8, labelspacing=0.6, ncol=1,
                fontsize=11)
    ts_cursor = axts.axvline(t[0], color='k', lw=1.5)

    def draw(idx):
        if quiver_holder['outline'] is not None:
            quiver_holder['outline'].remove()
        if quiver_holder['fill'] is not None:
            quiver_holder['fill'].remove()

        for j in track_idx:
            trails[j].append((x[j] + S[idx, j, 0], S[idx, j, 1], S[idx, j, 2]))
            pts = list(trails[j])
            if len(pts) > 1:
                n_seg = len(pts) - 1
                pts_arr = np.asarray(pts)
                segs = np.stack([pts_arr[:-1], pts_arr[1:]], axis=1)
                a = np.arange(1, n_seg + 1) / n_seg
                rgba = np.tile(np.array(track_colors[j]), (n_seg, 1))
                rgba[:, 3] = 0.15 + 0.65 * a
                trail_collections[j].set_segments(segs)
                trail_collections[j].set_colors(rgba)
                trail_collections[j].set_linewidths(0.8 + 1.4 * a)

        colors_now = cmap(norm(S[idx, :, 0]))
        colors_now = np.array(colors_now)
        colors_now[driven_idx] = to_rgba('#7e2bb6')
        # dark outline first (so near-white S_x~0 arrows stay visible on the
        # white background), colored fill on top, slightly thinner.
        quiver_holder['outline'] = ax3d.quiver(
            x, np.zeros(nspin), np.zeros(nspin),
            S[idx, :, 0], S[idx, :, 1], S[idx, :, 2],
            color='#333333', linewidth=3.4, arrow_length_ratio=0.22)
        quiver_holder['fill'] = ax3d.quiver(
            x, np.zeros(nspin), np.zeros(nspin),
            S[idx, :, 0], S[idx, :, 1], S[idx, :, 2],
            colors=colors_now, linewidth=1.9, arrow_length_ratio=0.20)

        for j in track_idx:
            tip_lines[j].set_data(S[:idx + 1, j, 0], S[:idx + 1, j, 1])
            tip_dots[j].set_data([S[idx, j, 0]], [S[idx, j, 1]])

        ts_cursor.set_xdata([t[idx], t[idx]])

        if t[idx] < T_ON_G3:
            stage = 'Relaxation (Drive off)'
        elif t[idx] < T_ON_G1:
            stage = 'g3 (Uniform precession) On'
        else:
            stage = 'g1 (Traveling wave) + g3 On'
        dyn_text.set_text(f't = {t[idx]:.1f}   |   {stage}')
        return ()

    return fig, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, choices=['kpos', 'kneg'])
    ap.add_argument('--data-dir', default=DEFAULT_DIR)
    ap.add_argument('--outdir', default=DEFAULT_DIR)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--frame-skip', type=int, default=5,
                     help='use 1 out of every N CSV rows as a frame')
    ap.add_argument('--t-min', type=float, default=None)
    ap.add_argument('--t-max', type=float, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    csv_path = os.path.join(args.data_dir, f'oscillators_trace_{args.run}.csv')
    t, spins, sites, S = load_chain(csv_path)
    print(f'  [{args.run}] read {csv_path}  ({len(t)} steps, {len(sites)} spins)')

    sel = np.ones(len(t), dtype=bool)
    if args.t_min is not None:
        sel &= t >= args.t_min
    if args.t_max is not None:
        sel &= t <= args.t_max
    t, S = t[sel], S[sel]

    frame_idx = list(range(0, len(t), args.frame_skip))
    label = r'$k{>}0$ (kpos)' if args.run == 'kpos' else r'$k{<}0$ (kneg)'
    fig, draw = build_figure(t, spins, sites, S, label)

    def update(fi):
        return draw(frame_idx[fi])

    n_frames = len(frame_idx)
    print(f'  frames: {n_frames}, fps: {args.fps}, duration~{n_frames/args.fps:.1f}s')
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    out_path = os.path.join(args.outdir, f'anim_oscillators_{args.run}.mp4')
    writer = animation.FFMpegWriter(
        fps=args.fps, bitrate=2600, codec='libx264',
        extra_args=['-preset', 'veryfast', '-threads', '2'])
    ani.save(out_path, writer=writer, dpi=100)
    print(f'  → {out_path}  ({n_frames} cuadros)')


if __name__ == '__main__':
    main()
