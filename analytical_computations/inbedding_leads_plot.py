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
# (no hay DEFAULT_LDOS_CSV: la LDOS se resuelve junto al --rho-csv, ver main)
DEFAULT_RHO_CSV = os.path.join(SCRIPT_DIR, 'output', 'inbedding_rho_t.csv')

# Prefijo de los archivos de salida. Se cambia con --prefix para no confundir
# las figuras del analitico con las de TDNEGF, que salen del mismo script.
PREFIX = 'inbedding'

# Sufijo del nombre, lo pone --t-min. Asi una figura recortada a la cola no
# sobrescribe la de la evolucion completa.
SUFFIX = ''

# Color por componente de espin, unico para todas las figuras de curvas. Los
# heatmaps no lo usan: ahi el color codifica el VALOR, no la componente.
# comparison_tdnegf_floquet_plot.py lo importa de aqui para no desincronizarse.
SPIN_COLORS = {'sx': 'blue', 'sy': 'green', 'sz': 'red'}


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
    # Posicion por defecto: arriba a la izquierda, encima de los ticks del eje
    # y, que es la escala a la que multiplica. En los heatmaps el exponente va
    # sobre la barra de color (ver _sci_cbar), que es otro eje.
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
    cols = [('n_up',        r'$\rho^{\uparrow\uparrow}_{\text{ii}}$'),
            ('n_dn',        r'$\rho^{\downarrow\downarrow}_{\text{ii}}$'),
            ('Re_rho_updn', r'$\text{Re}\,\rho^{\uparrow\downarrow}_{\text{ii}}$'),
            ('Im_rho_updn', r'$\text{Im}\,\rho^{\uparrow\downarrow}_{\text{ii}}$')]

    for ax, (col, lab) in zip(axes.flat, cols):
        _heatmap(ax, df, col, lab, T)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\text{Site i}$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
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
    cols = [('n_tot', r'$\langle\hat{\sigma}^{\text{0}}_{\text{i}}\rangle$'),
            ('sx',    r'$\langle\hat{\sigma}^{\text{x}}_{\text{i}}\rangle$'),
            ('sy',    r'$\langle\hat{\sigma}^{\text{y}}_{\text{i}}\rangle$'),
            ('sz',    r'$\langle\hat{\sigma}^{\text{z}}_{\text{i}}\rangle$')]

    for ax, (col, lab) in zip(axes.flat, cols):
        _heatmap(ax, df, col, lab, T)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\text{Site i}$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
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
    _top_legend(fig, [Line2D([], [], color=c, lw=2.0, label=rf'$\text{{i}}={n + 1}$')
                      for n, c in colors.items()])


def _top_legend(fig, handles):
    """Misma leyenda pero con handles arbitrarios (p.ej. componentes de espin)."""
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.975),
               ncol=len(handles), frameon=False, fontsize=18,
               handlelength=1.8, columnspacing=1.5, handletextpad=0.6)


def _legend_en_hueco(ax, handles, labels, pad=0.025, **kw):
    """
    Coloca la leyenda en la banda horizontal mas ancha que no cruza ninguna
    curva, en vez de dejarsela a loc='best'.

    loc='best' puntua solo las 9 anclas estandar (esquinas, centros de borde,
    centro) y si todas chocan escoge la menos mala, que en estos paneles cae
    justo encima de una curva. Aqui se prueban tres franjas verticales
    (izquierda / centro / derecha), se recogen las y de todas las curvas que
    pasan por esa franja, y se busca el mayor salto entre y consecutivas: ese
    salto es, por construccion, una banda libre en TODA la franja. La leyenda
    se centra ahi.
    """
    kw.setdefault('frameon', False)
    leg = ax.legend(handles, labels, loc='lower left', **kw)

    # Hay que dibujar para poder medir la caja, y hacerlo despues de
    # tight_layout: si no, el tamano relativo de la leyenda cambia luego.
    ax.figure.tight_layout()
    ax.figure.canvas.draw()
    inv = ax.transAxes.inverted()
    bb = leg.get_window_extent().transformed(inv)
    w, h = bb.width, bb.height

    curvas = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        # <3 puntos = axhline/axvline y similares: su x no esta en transData,
        # medirlos daria basura.
        if not ln.get_visible() or len(xy) < 3:
            continue
        a = inv.transform(ax.transData.transform(xy))
        curvas.append(a[np.isfinite(a).all(axis=1)])

    mejor = None
    for xl in (pad, 0.5 - w / 2, 1.0 - w - pad):
        if xl < 0.0 or xl + w > 1.0:
            continue
        ys = [a[(a[:, 0] >= xl - pad) & (a[:, 0] <= xl + w + pad), 1]
              for a in curvas]
        ys = np.concatenate(ys) if ys else np.empty(0)
        # los bordes del panel cuentan como ocupados
        bordes = np.sort(np.concatenate([[0.0], np.clip(ys, 0.0, 1.0), [1.0]]))
        saltos = np.diff(bordes)
        k = int(np.argmax(saltos))
        if mejor is None or saltos[k] > mejor[0]:
            mejor = (saltos[k], xl, bordes[k] + (saltos[k] - h) / 2)

    if mejor is not None:
        _, xl, yb = mejor
        leg.set_bbox_to_anchor((xl, np.clip(yb, pad, 1.0 - h - pad)),
                               transform=ax.transAxes)
    return leg


def _alpha_legend(ax, fontsize=16):
    """
    Cajita "alpha = x, y, z" con cada letra del color de su componente.

    Va como leyenda y no como ax.text para que matplotlib le busque el hueco
    con loc='best'. handlelength=0 porque no hay simbolo que mostrar: el color
    de la LETRA es lo que identifica la componente, asi que una linea de
    muestra al lado seria redundante.
    """
    lab = [r'$\alpha=$', r'x,', r'y,', r'z']
    col = ['black', SPIN_COLORS['sx'], SPIN_COLORS['sy'], SPIN_COLORS['sz']]
    _legend_en_hueco(ax, [Line2D([], [], ls='none') for _ in lab], lab,
                     ncol=len(lab), fontsize=fontsize, handlelength=0.0,
                     handletextpad=0.0, columnspacing=0.45, labelcolor=col)


def _panel_site_legend(ax, colors, fontsize=15):
    """Leyenda de sitios en UNA columna, dentro del panel, en loc='best'."""
    _legend_en_hueco(ax,
                     [Line2D([], [], color=c, lw=2.0) for c in colors.values()],
                     [rf'$\text{{i}}={n + 1}$' for n in colors],
                     ncol=1, fontsize=fontsize, handlelength=1.6,
                     labelspacing=0.35, handletextpad=0.6)


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
    cols = [('LDOS_up', r'$A^{\uparrow}_{\text{i}}(\omega)\ (1/\gamma)$'),
            ('LDOS_dn', r'$A^{\downarrow}_{\text{i}}(\omega)\ (1/\gamma)$')]

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
    """
    Panel 2x2 de curvas: las cuatro componentes de la matriz densidad, con unos
    pocos sitios superpuestos.

    Complementa a plot_rho_map, no lo sustituye: el heatmap da la vista global
    de los 20 sitios, esta da el detalle cuantitativo de unos pocos.
    """
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    cols = [('n_up',        r'$\rho^{\uparrow\uparrow}_{\text{ii}}(t)$'),
            ('n_dn',        r'$\rho^{\downarrow\downarrow}_{\text{ii}}(t)$'),
            ('Re_rho_updn', r'$\text{Re}\,\rho^{\uparrow\downarrow}_{\text{ii}}(t)$'),
            ('Im_rho_updn', r'$\text{Im}\,\rho^{\uparrow\downarrow}_{\text{ii}}(t)$')]

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
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
    _panel_site_legend(axes[0, 0], colors)
    _save(fig, outdir, f'{PREFIX}_rho_t_{lead}')


def plot_spin_t(df, outdir, lead, Omega):
    """Panel 2x2 de curvas: un sitio por panel, las tres componentes de espin."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    comps = [('sx', SPIN_COLORS['sx'], r'$\langle\hat{\sigma}^{\text{x}}_{\text{i}}\rangle$'),
             ('sy', SPIN_COLORS['sy'], r'$\langle\hat{\sigma}^{\text{y}}_{\text{i}}\rangle$'),
             ('sz', SPIN_COLORS['sz'], r'$\langle\hat{\sigma}^{\text{z}}_{\text{i}}\rangle$')]

    for ax, n in zip(axes.flat, sites):
        s = df[df['site'] == n].sort_values('t')
        x = s['t'].to_numpy() / T
        for col, color, _ in comps:
            ax.plot(x, s[col].to_numpy(), '-', color=color, lw=1.5, zorder=3)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.0, zorder=1)
        ax.set_title(rf'$\text{{i}}={n + 1}$', fontsize=18)
        ax.set_xlim(x.min(), x.max())
        _fmt_axes(ax)
        _sci_yaxis(ax)

    for ax in axes.flat[len(sites):]:          # si hay menos de 4 sitios
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\langle\hat{\sigma}^{\alpha}_{\text{i}}\rangle(t)$')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')

    _alpha_legend(axes[0, 0])
    _save(fig, outdir, f'{PREFIX}_spin_t_{lead}')


def plot_occupation_t(df, outdir, lead, Omega):
    """Panel 1x2: ocupacion de espin up y de espin down, lado a lado."""
    T = 2 * np.pi / Omega
    sites = sorted(df['site'].unique())
    colors = _site_colors(sites)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = [('n_up', r'$n^{\uparrow}_{\text{i}}(t)$'),
            ('n_dn', r'$n^{\downarrow}_{\text{i}}(t)$')]

    for ax, (col, ylab) in zip(axes, cols):
        for n in sites:
            s = df[df['site'] == n].sort_values('t')
            x = s['t'].to_numpy() / T
            ax.plot(x, s[col].to_numpy(), '-', color=colors[n], lw=1.4, zorder=3)
        ax.set_xlabel(r'$\text{Time}\, (2\pi/\Omega)$')
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
    ap.add_argument('--ldos-csv', default=None,
                    help='CSV de LDOS. Por defecto se busca junto al --rho-csv, como '
                         '<dir>/<prefix>_ldos.csv, y si no existe se omite el panel. '
                         'OJO: no hay fallback a una ruta fija. Antes lo habia y apuntaba '
                         'al analitico, asi que correr con --prefix tdnegf producia un '
                         'tdnegf_ldos_*.jpg que era la LDOS analitica mal etiquetada. '
                         'TDNEGF propaga en el tiempo y nunca calcula A(omega).')
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
    # La LDOS se busca JUNTO al rho-csv y con el mismo prefijo, nunca en una
    # ruta fija: si no, una corrida de TDNEGF se lleva la LDOS del analitico.
    ldos_csv = args.ldos_csv
    if ldos_csv is None:
        ldos_csv = os.path.join(os.path.dirname(os.path.abspath(args.rho_csv)),
                                f'{PREFIX}_ldos.csv')

    dl = None
    if os.path.exists(ldos_csv):
        df_ldos = pd.read_csv(ldos_csv)
        dl = df_ldos[df_ldos['lead'] == lead]
        if dl.empty:
            dl = None
    else:
        print(f'  (sin {os.path.basename(ldos_csv)}: me salto el panel de LDOS)')

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
    # Curvas (pocos sitios, detalle) Y heatmaps (todos los sitios, vista
    # global). Son complementarios, se generan siempre los dos.
    plot_rho_t(dr, args.outdir, lead, Omega=args.Omega)
    plot_spin_t(dr, args.outdir, lead, Omega=args.Omega)
    plot_rho_map(dr_all, args.outdir, lead, Omega=args.Omega)
    plot_spin_map(dr_all, args.outdir, lead, Omega=args.Omega)
    plot_occupation_t(dr, args.outdir, lead, Omega=args.Omega)


if __name__ == '__main__':
    main()
