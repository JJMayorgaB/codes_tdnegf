#!/usr/bin/env python3

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HAS_LATEX = shutil.which('latex') is not None

plt.rcParams.update({
    'text.usetex': HAS_LATEX,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'] if HAS_LATEX else ['DejaVu Serif'],
    'font.size': 20,
    'axes.labelsize': 22,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.1,
    'axes.axisbelow': True,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LDOS_CSV = os.path.join(SCRIPT_DIR, 'output', 'inbedding_ldos.csv')
DEFAULT_RHO_CSV = os.path.join(SCRIPT_DIR, 'output', 'inbedding_rho_t.csv')

# Prefijo de los archivos de salida. Se cambia con --prefix para no confundir
# las figuras del analitico con las de TDNEGF, que salen del mismo script.
PREFIX = 'inbedding'


def _fmt_axes(ax):
    ax.tick_params(axis='both', direction='in', bottom=True, top=True,
                   left=True, right=True, size=5.0, width=1.0)
    for axis in (ax.xaxis, ax.yaxis):
        fmt = axis.get_major_formatter()
        if isinstance(fmt, mticker.ScalarFormatter):
            fmt.set_useMathText(False)


def _site_colors(sites):
    """Color por sitio: degradado oscuro->claro segun se entra al lead."""
    cmap = plt.get_cmap('viridis')
    return {n: cmap(0.12 + 0.72 * i / max(1, len(sites) - 1))
            for i, n in enumerate(sites)}


def _pick_sites(available, n=4):
    """
    Elige n sitios para graficar, con espaciado ~geometrico.

    El CSV puede traer muchos mas sitios de los que conviene superponer (p.ej.
    sweep_leads(sites=0:19) para la animacion). El espaciado geometrico da
    superficie + profundidades crecientes, que es donde esta el contraste.
    """
    av = sorted(available)
    if len(av) <= n:
        return av
    idx = sorted({min(len(av) - 1, int(round(f)) - 1)
                  for f in np.geomspace(1, len(av), n)})
    while len(idx) < n:                       # relleno si el redondeo colapso
        extra = next(i for i in range(len(av)) if i not in idx)
        idx = sorted(idx + [extra])
    return [av[i] for i in idx]


def _sci_yaxis(ax, fontsize=15):
    """
    Manda el exponente (y el offset, si hace falta) a la esquina del panel y
    deja los ticks con pocas cifras.

    Las diagonales de rho viven pegadas a 1/2 con variaciones de 1e-3: sin
    offset harian falta 5 decimales por tick. El offset automatico de
    matplotlib no dispara aqui (exige media/rango > 1e4 y esto es ~3e2), asi
    que se calcula uno redondo a mano cuando los datos estan lejos del cero
    comparados con su propio rango. Para las componentes de fuera de diagonal,
    centradas en cero, basta el multiplicador.
    """
    lo, hi = ax.get_ylim()
    span, mid = hi - lo, 0.5 * (lo + hi)
    off = False
    if span > 0 and abs(mid) > 10 * span:
        step = 10.0 ** np.floor(np.log10(abs(mid)))
        off = round(mid / step) * step
    fmt = mticker.ScalarFormatter(useOffset=off, useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_offset_position('right')
    ax.yaxis.get_offset_text().set_fontsize(fontsize)


def _site_legend(fig, colors):
    """
    Leyenda de sitios en una sola fila, arriba de todos los paneles.

    Va a nivel de figura (no de eje) para no tapar curvas ni competir con el
    exponente que _sci_yaxis pone en la esquina. Queda por encima del area de
    los ejes; se guarda completa porque _save usa bbox_inches='tight'.
    """
    # Los CSV guardan el sitio 0-based (convencion del codigo Julia: n=0 es la
    # superficie). En las figuras se muestra 1-based y con el simbolo i, que es
    # la notacion del paper. El relabel vive SOLO aqui, los datos no se tocan.
    _top_legend(fig, [Line2D([], [], color=c, lw=2.0, label=rf'$i={n + 1}$')
                      for n, c in colors.items()])


def _top_legend(fig, handles):
    """Misma leyenda pero con handles arbitrarios (p.ej. componentes de espin)."""
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.975),
               ncol=len(handles), frameon=False, fontsize=16,
               handlelength=1.8, columnspacing=2.0, handletextpad=0.6)


def _save(fig, outdir, name):
    fig.tight_layout()
    for ext in ('jpg', 'svg', 'pdf'):
        path = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def plot_ldos(df, outdir, lead, wmax=None):
    """Panel 1x2: LDOS de espin up y de espin down, lado a lado."""
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = [('LDOS_up', r'$A^{\uparrow}_i(\omega)\ (1/\gamma)$'),
            ('LDOS_dn', r'$A^{\downarrow}_i(\omega)\ (1/\gamma)$')]

    for ax, (col, ylab) in zip(axes, cols):
        for n in sites:
            s = df[df['site'] == n].sort_values('omega')
            w, y = s['omega'].to_numpy(), s[col].to_numpy()
            if wmax is not None:
                sel = np.abs(w) <= wmax
                w, y = w[sel], y[sel]
            ax.plot(w, y, '-', color=colors[n], lw=1.4, zorder=3)
        ax.set_xlabel(r'$\omega\ (\gamma)$')
        ax.set_ylabel(ylab)
        ax.set_xlim(w.min(), w.max())
        ax.set_ylim(bottom=0.0)
        _fmt_axes(ax)

    _site_legend(fig, colors)
    _save(fig, outdir, f'{PREFIX}_ldos_{lead}')


def plot_rho_t(df, outdir, lead, Omega):
    """Panel 2x2: las cuatro componentes de la matriz densidad."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    cols = [('n_up',        r'$\rho_{\uparrow\uparrow}(t)$'),
            ('n_dn',        r'$\rho_{\downarrow\downarrow}(t)$'),
            ('Re_rho_updn', r'$\text{Re}\,\rho_{\uparrow\downarrow}(t)$'),
            ('Im_rho_updn', r'$\text{Im}\,\rho_{\uparrow\downarrow}(t)$')]

    for ax, (col, ylab) in zip(axes.flat, cols):
        for n in sites:
            s = df[df['site'] == n].sort_values('t')
            x = s['t'].to_numpy() / T
            ax.plot(x, s[col].to_numpy(), '-', color=colors[n], lw=1.4, zorder=3)
        ax.set_ylabel(ylab)
        ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax)
        _sci_yaxis(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
    _site_legend(fig, colors)
    _save(fig, outdir, f'{PREFIX}_rho_t_{lead}')


def plot_spin_t(df, outdir, lead, Omega):
    """Panel 2x2: un sitio por panel, las tres componentes de espin en cada uno."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    comps = [('sx', 'red',   r'$\langle\sigma_x\rangle$'),
             ('sy', 'blue',  r'$\langle\sigma_y\rangle$'),
             ('sz', 'black', r'$\langle\sigma_z\rangle$')]

    for ax, n in zip(axes.flat, sites):
        s = df[df['site'] == n].sort_values('t')
        x = s['t'].to_numpy() / T
        for col, color, _ in comps:
            ax.plot(x, s[col].to_numpy(), '-', color=color, lw=1.5, zorder=3)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        ax.set_title(rf'$i={n + 1}$', fontsize=18)
        ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax)
        _sci_yaxis(ax)

    for ax in axes.flat[len(sites):]:          # si hay menos de 4 sitios
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\langle\boldsymbol{\hat{\sigma}}\rangle_i(t)$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')

    _top_legend(fig, [Line2D([], [], color=c, lw=2.0, label=lab)
                      for _, c, lab in comps])
    _save(fig, outdir, f'{PREFIX}_spin_t_{lead}')


def plot_occupation_t(df, outdir, lead, Omega):
    """Panel 1x2: ocupacion de espin up y de espin down, lado a lado."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = [('n_up', r'$n^{\uparrow}(t)$'),
            ('n_dn', r'$n^{\downarrow}(t)$')]

    for ax, (col, ylab) in zip(axes, cols):
        for n in sites:
            s = df[df['site'] == n].sort_values('t')
            x = s['t'].to_numpy() / T
            ax.plot(x, s[col].to_numpy(), '-', color=colors[n], lw=1.4, zorder=3)
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
        ax.set_ylabel(ylab)
        ax.set_xlim(x.min(), x.max())
        #ax.set_ylim(0.0, 1.0)
        _fmt_axes(ax)
        # Con el ylim fijo comentado estos paneles autoescalan, y n_up/n_dn son
        # los mismos datos que las diagonales de rho: pegados a 1/2. Sin esto
        # volverian los ticks de cinco decimales.
        _sci_yaxis(ax)

    _site_legend(fig, colors)
    _save(fig, outdir, f'{PREFIX}_occ_t_{lead}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ldos-csv', default=DEFAULT_LDOS_CSV)
    ap.add_argument('--rho-csv', default=DEFAULT_RHO_CSV)
    ap.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'output'))
    ap.add_argument('--wmax', type=float, default=None,
                    help='recorta el eje de omega (|omega|<=wmax) en el plot de LDOS')
    ap.add_argument('--Omega', type=float, default=0.005,
                    help='frecuencia de driving usada en el barrido (para el eje t/T)')
    ap.add_argument('--lead', default='R', help='lead a graficar (por defecto R)')
    ap.add_argument('--sites', default=None,
                    help='sitios a graficar, separados por coma (p.ej. 0,1,4,16). '
                         'Por defecto se eligen 4 con espaciado geometrico.')
    ap.add_argument('--prefix', default='inbedding',
                    help='prefijo de los archivos de salida (usa tdnegf para el test)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    global PREFIX
    PREFIX = args.prefix

    lead = args.lead.strip()
    df_rho = pd.read_csv(args.rho_csv)
    dr = df_rho[df_rho['lead'] == lead]
    if dr.empty:
        raise SystemExit(f'no hay datos para el lead {lead} en {args.rho_csv}')

    # El CSV de LDOS es OPCIONAL: TDNEGF propaga en el tiempo y nunca calcula
    # A(ω), asi que sus salidas no lo tienen. Los otros tres paneles si sirven
    # igual para el analitico y para TDNEGF.
    dl = None
    if os.path.exists(args.ldos_csv):
        df_ldos = pd.read_csv(args.ldos_csv)
        dl = df_ldos[df_ldos['lead'] == lead]
        if dl.empty:
            dl = None
    else:
        print(f'  (sin {os.path.basename(args.ldos_csv)}: me salto el panel de LDOS)')

    # El CSV puede traer muchos sitios; los paneles solo aguantan unos pocos.
    avail = set(dr['site'].unique())
    if dl is not None:
        avail |= set(dl['site'].unique())
    sel = ([int(s) for s in args.sites.split(',')] if args.sites is not None
           else _pick_sites(sorted(avail)))
    print(f'  lead {lead}, sitios del CSV: {sel}   ->  en las figuras: '
          f'{[s + 1 for s in sel]}  (i = n+1)')
    dr = dr[dr['site'].isin(sel)]

    if dl is not None:
        plot_ldos(dl[dl['site'].isin(sel)], args.outdir, lead, wmax=args.wmax)
    plot_rho_t(dr, args.outdir, lead, Omega=args.Omega)
    plot_spin_t(dr, args.outdir, lead, Omega=args.Omega)
    plot_occupation_t(dr, args.outdir, lead, Omega=args.Omega)


if __name__ == '__main__':
    main()
