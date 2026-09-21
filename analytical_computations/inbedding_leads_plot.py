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

# Sufijo del nombre, lo pone --t-min. Asi una figura recortada a la cola no
# sobrescribe la de la evolucion completa.
SUFFIX = ''


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


def _sci_cbar(cb, fontsize=14):
    """
    Mismo criterio que _sci_yaxis pero para la barra de color.

    Hace falta sobre todo en el panel de sigma_0: n_tot vive pegado a 1 con
    variaciones de 1e-5, asi que sin offset cada tick serian seis decimales.
    """
    lo, hi = cb.mappable.get_clim()
    span, mid = hi - lo, 0.5 * (lo + hi)
    off = False
    if span > 0 and abs(mid) > 10 * span:
        step = 10.0 ** np.floor(np.log10(abs(mid)))
        off = round(mid / step) * step
    fmt = mticker.ScalarFormatter(useOffset=off, useMathText=True)
    fmt.set_powerlimits((0, 0))
    # OJO: hay que asignar a cb.formatter / cb.locator, NO a cb.ax.yaxis. La
    # Colorbar guarda los suyos aparte y update_ticks() los reimpone sobre el
    # eje, asi que un set_major_formatter directo se pierde al dibujar.
    cb.formatter = fmt
    cb.locator = mticker.MaxNLocator(4)
    cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_fontsize(fontsize)
    cb.ax.tick_params(labelsize=fontsize)


def _grid(df, col, T):
    """Pasa el CSV largo a una malla (sitio, tiempo) lista para pcolormesh."""
    p = df.pivot_table(index='site', columns='t', values=col)
    return p.index.to_numpy(), p.columns.to_numpy() / T, p.to_numpy()


def _heatmap(ax, df, col, label, T):
    """
    Un cuadrante: eje y = indice de sitio i, eje x = t/T, color = observable.

    Escala lineal y datos crudos. La barra de color autoescala al rango real
    del observable, que es lo que deja leer los paneles casi constantes.
    Colormap divergente y centrado en cero solo si el dato cambia de signo;
    si no, secuencial.
    """
    sites, t, Z = _grid(df, col, T)
    if Z.min() < 0.0 < Z.max():
        vmax = np.abs(Z).max()
        kw = dict(cmap='seismic', vmin=-vmax, vmax=vmax)
    else:
        kw = dict(cmap='inferno', vmin=Z.min(), vmax=Z.max())

    # shading='nearest' dibuja una celda por sitio: el retículo es discreto y
    # no hay senal entre sitios, asi que nada de interpolacion bilineal.
    im = ax.pcolormesh(t, sites + 1, Z, shading='nearest', rasterized=True, **kw)
    # loc='left': el exponente de la barra de color se dibuja arriba a la
    # derecha, justo donde caeria un titulo centrado.
    ax.set_title(label, fontsize=18, pad=6, loc='left')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    _fmt_axes(ax)
    # rcParams trae axes.axisbelow=True, que deja los ticks por detras de los
    # artistas. Con curvas no importa, pero el pcolormesh es una superficie
    # llena y se los come: los ticks 'in' desaparecen. Hay que subirlos.
    ax.set_axisbelow(False)
    _sci_cbar(ax.figure.colorbar(im, ax=ax, pad=0.025, fraction=0.046))


def plot_rho_map(df, outdir, lead, Omega):
    """Panel 2x2: las cuatro componentes de la matriz densidad, como heatmaps."""
    T = 2 * np.pi / Omega
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    cols = [('n_up',        r'$\rho^{\uparrow\uparrow}_{i}$'),
            ('n_dn',        r'$\rho^{\downarrow\downarrow}_{i}$'),
            ('Re_rho_updn', r'$\text{Re}\,\rho^{\uparrow\downarrow}_{i}$'),
            ('Im_rho_updn', r'$\text{Im}\,\rho^{\uparrow\downarrow}_{i}$')]

    for ax, (col, lab) in zip(axes.flat, cols):
        _heatmap(ax, df, col, lab, T)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$i$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
    _save(fig, outdir, f'{PREFIX}_rho_map_{lead}')


def plot_spin_map(df, outdir, lead, Omega):
    """
    Panel 2x2 en la base de Pauli:  sigma_0 | sigma_x  /  sigma_y | sigma_z.

    sigma_0 es la poblacion electronica n_tot = Tr rho. Sale plano en el
    tiempo a precision de maquina: el cono rota M pero no bombea carga, solo
    espin. Las bandas horizontales de ese cuadrante son ese resultado.
    """
    T = 2 * np.pi / Omega
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    cols = [('n_tot', r'$\langle\sigma^{0}_{i}\rangle$'),
            ('sx',    r'$\langle\sigma^{x}_{i}\rangle$'),
            ('sy',    r'$\langle\sigma^{y}_{i}\rangle$'),
            ('sz',    r'$\langle\sigma^{z}_{i}\rangle$')]

    for ax, (col, lab) in zip(axes.flat, cols):
        _heatmap(ax, df, col, lab, T)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$i$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$t\, (2\pi/\Omega)$')
    _save(fig, outdir, f'{PREFIX}_spin_map_{lead}')


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
        path = os.path.join(outdir, f'{name}{SUFFIX}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f'  -> {path}')
    plt.close(fig)


def plot_ldos(df, outdir, lead, wmax=None):
    """Panel 1x2: LDOS de espin up y de espin down, lado a lado."""
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = [('LDOS_up', r'$A^{\uparrow}_{i}(\omega)\ (1/\gamma)$'),
            ('LDOS_dn', r'$A^{\downarrow}_{i}(\omega)\ (1/\gamma)$')]

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


def plot_occupation_t(df, outdir, lead, Omega):
    """Panel 1x2: ocupacion de espin up y de espin down, lado a lado."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = [('n_up', r'$n^{\uparrow}_{i}(t)$'),
            ('n_dn', r'$n^{\downarrow}_{i}(t)$')]

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
    corte = ap.add_mutually_exclusive_group()
    corte.add_argument('--t-min', type=float, default=None,
                       help='recorta las figuras temporales a t >= t_min, para dejar '
                            'solo la cola estacionaria. El valor queda en el nombre '
                            'del archivo, asi no pisa la figura de la evolucion completa.')
    corte.add_argument('--last-periods', type=float, default=None, metavar='N',
                       help='igual que --t-min pero contando N periodos hacia atras '
                            'desde el ultimo tiempo del CSV: t_min = t_max - N*T. '
                            'Pensado para TDNEGF, donde t_final depende de la corrida '
                            'y calcular el corte a mano es incomodo.')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    global PREFIX, SUFFIX
    PREFIX = args.prefix

    lead = args.lead.strip()
    df_rho = pd.read_csv(args.rho_csv)
    dr = df_rho[df_rho['lead'] == lead]
    if dr.empty:
        raise SystemExit(f'no hay datos para el lead {lead} en {args.rho_csv}')

    # El recorte temporal solo aplica a los observables en el tiempo; la LDOS
    # esta resuelta en omega y no tiene eje temporal que recortar.
    # --last-periods necesita t_max, asi que el corte se resuelve despues de
    # leer el CSV; --t-min es el valor absoluto y no depende de los datos.
    t_min = args.t_min
    if args.last_periods is not None:
        T = 2 * np.pi / args.Omega
        t_min = dr['t'].max() - args.last_periods * T
        SUFFIX = f'_last{args.last_periods:g}T'
    elif t_min is not None:
        SUFFIX = f'_tmin{t_min:g}'

    if t_min is not None:
        dr = dr[dr['t'] >= t_min]
        if dr.empty:
            raise SystemExit(f'no quedan datos con t >= {t_min}')
        print(f'  recorte temporal: t >= {t_min:g}  '
              f'({dr["t"].min():.1f} a {dr["t"].max():.1f}, '
              f'{(dr["t"].max() - dr["t"].min()) * args.Omega / (2 * np.pi):.2f} periodos, '
              f'{dr["t"].nunique()} puntos)')

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
    # Los heatmaps usan TODOS los sitios del CSV (el sitio es un eje, no una
    # curva superpuesta); --sites solo recorta los paneles de curvas.
    dr_all = dr
    dr = dr[dr['site'].isin(sel)]
    ns = sorted(dr_all['site'].unique())
    print(f'  heatmaps: {len(ns)} sitios, i = {ns[0] + 1}..{ns[-1] + 1}')

    if dl is not None:
        plot_ldos(dl[dl['site'].isin(sel)], args.outdir, lead, wmax=args.wmax)
    plot_rho_map(dr_all, args.outdir, lead, Omega=args.Omega)
    plot_spin_map(dr_all, args.outdir, lead, Omega=args.Omega)
    plot_occupation_t(dr, args.outdir, lead, Omega=args.Omega)


if __name__ == '__main__':
    main()
