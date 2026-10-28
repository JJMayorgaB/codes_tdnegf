#!/usr/bin/env python3
"""
check_kernel_results.py -- revisa los resultados del kernel ya calculados (run_cluster.sh)
sin recalcular nada. Para cada conjunto output/<tag>/data/ (y su logs/<tag>.log):

  1. integridad   archivos completos (armonicos, 100 kernel_chi, 100 kernel_tau, chi0,
                  llg); mismo numero de filas en cada familia; sin NaN/Inf; chi_sign=1;
                  encabezados coherentes entre si y con el nombre de la carpeta
  2. log          inicio y fin, sin errores, numero de archivos escritos; peores valores
                  de realidad y causalidad que imprimio kernel_elements.jl; τ_mem
  3. consistencia --chi0 frente a --kernel (dos corridas del mismo calculo);
                  fila ω = 0 de kernel_chi frente al resumen de --kernel;
                  ventanas γ y Ω en los ω comunes; simetria χ̌_0(-ω) = χ̌_0(ω)* (χ̌ e i∂χ̌);
                  i∂_ωχ̌ (derivada exacta) frente a la diferencia finita de χ̌ en la ventana Ω;
                  realidad del k = 0 en ω = 0;
                  kernel en τ: causalidad (τ < -2), realidad del k = 0 y ∫dτ χ̌_k(τ) = χ̌_k(0)
                  (con e^{-2(η_ω-η_τ)|τ|} para comparar con el mismo ensanchamiento)
  4. LLG          llg_elements recalculado desde kernel_chi0 (mismo signo en ambos);
                  J_mm > 0 y α_mm < 0 (signo fisico); relacion de la torsion de gauge
                  atan2(t^{y,A}, J^S + Γ^{xx,S}) = 2 θ_SO d (mod π) en todos los pares

Cada prueba da OK / AVISO / FALLA (INFO: solo informativo). El resumen va a la pantalla
y a output/check_report.txt.

Uso:
    python check_kernel_results.py                      # todos los output/setup2_*/data
    python check_kernel_results.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005
    python check_kernel_results.py --skip-tau           # sin los kernel_tau (mas rapido)
"""

import argparse
import glob
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
K = ('xx0', 'yy0', 'xz0', 'xy1', 'yx1', 'xx2')
K0 = ('xx0', 'yy0', 'xz0')
NAMES_A = ['J', 'Γxx', 'Γyy', 'Γzz', 'Γxz', 'Γxy', 'Γyz', 'tx', 'ty', 'tz']
NAMES_D = ['α', 'Λxx', 'Λyy', 'Λzz', 'Λxz', 'Λxy', 'Λyz', 'rx', 'ry', 'rz']
LEVELS = ('OK', 'INFO', 'AVISO', 'FALLA')


# ---------------------------------------------------------------- utilidades
class Report:
    def __init__(self, tag):
        self.tag = tag
        self.items = []

    def add(self, level, check, msg):
        self.items.append((level, check, msg))

    def grade(self, check, value, warn, fail, fmt='{:.1e}', extra=''):
        """value <= warn: OK; <= fail: AVISO; si no FALLA (NaN = FALLA)."""
        if value is None or not np.isfinite(value):
            lvl = 'FALLA'
        else:
            lvl = 'OK' if value <= warn else ('AVISO' if value <= fail else 'FALLA')
        self.add(lvl, check, fmt.format(value) + (f'  (aviso > {warn:.0e}, falla > {fail:.0e})' if lvl != 'OK' else '')
                 + (f'  {extra}' if extra else ''))

    def count(self):
        return {l: sum(1 for x in self.items if x[0] == l) for l in LEVELS}


def meta_of(path):
    with open(path, encoding='utf-8') as fh:
        head = fh.readline()
    if not head.startswith('#'):
        return {}
    return dict(kv.split('=', 1) for kv in head.lstrip('#').split() if '=' in kv)


def tag_params(tag):
    m = re.search(r'gso([0-9p]+)_jsd([0-9p]+)_th([0-9]+)deg_Om([0-9p]+)', tag)
    if m is None:
        return None
    f = lambda s: float(s.replace('p', '.'))
    return dict(gso=f(m.group(1)), jsd=f(m.group(2)), th=float(m.group(3)), Om=f(m.group(4)))


def z(df, re_col, im_col):
    return df[re_col].to_numpy() + 1j * df[im_col].to_numpy()


def finite_frame(df):
    num = df.select_dtypes(include=[np.number]).to_numpy()
    return np.isfinite(num).all()


# ---------------------------------------------------------------- 1. integridad
def check_integrity(R, d, N):
    pairs = [(i, j) for i in range(1, N + 1) for j in range(1, N + 1)]
    need = ['kernel_elements_1-1.csv', f'kernel_chi0_sites1-{N}.csv', f'kernel_chi0_kernel_sites1-{N}.csv',
            f'llg_elements_sites1-{N}.csv', f'llg_elements_SA_sites1-{N}.csv']
    need += [f'kernel_chi_m{i}n{j}.csv' for i, j in pairs] + [f'kernel_tau_m{i}n{j}.csv' for i, j in pairs]
    miss = [f for f in need if not os.path.isfile(os.path.join(d, f))]
    R.add('OK' if not miss else 'FALLA', 'archivos', f'{len(need) - len(miss)}/{len(need)} presentes'
          + (f'; faltan {len(miss)}: {", ".join(miss[:6])}{" ..." if len(miss) > 6 else ""}' if miss else ''))

    # encabezados: chi_sign=1 y parametros iguales a los de la carpeta
    tp = tag_params(R.tag)
    bad_sign, bad_par, metas = [], [], {}
    for f in need:
        p = os.path.join(d, f)
        if not os.path.isfile(p) or f.startswith('llg_'):
            continue
        m = meta_of(p)
        metas[f] = m
        if f != 'kernel_elements_1-1.csv' and m.get('chi_sign') != '1':
            bad_sign.append(f)
        if tp is not None and m:
            try:
                ok = (abs(float(m['gammaSO']) - tp['gso']) < 6e-5 and
                      abs(float(m['J_sd']) - tp['jsd']) < 1e-9 and
                      abs(math.degrees(float(m['theta'])) - tp['th']) < 0.51 and
                      abs(float(m['Omega']) - tp['Om']) < 1e-12 and
                      abs(float(m['gamma']) - math.sqrt(1 - float(m['gammaSO']) ** 2)) < 1e-10)
            except (KeyError, ValueError):
                ok = False
            if not ok:
                bad_par.append(f)
    R.add('OK' if not bad_sign else 'FALLA', 'chi_sign=1',
          'todos corregidos' if not bad_sign else f'{len(bad_sign)} sin corregir (corre fix_kernel_sign.py): {bad_sign[:4]}')
    R.add('OK' if not bad_par else 'FALLA', 'parametros vs carpeta',
          'coinciden (γ = sqrt(1-γ_SO²), γ_SO, J_sd, θ, Ω)' if not bad_par else f'no coinciden en {len(bad_par)}: {bad_par[:4]}')
    # mismas mallas dentro de cada familia
    for fam, keys in (('kernel_chi', ('eta', 'nOmega', 'wide', 'nwide', 'narrow', 'margin')),
                      ('kernel_tau', ('eta', 'dtau', 'taumax', 'log2N'))):
        vals = {tuple(m.get(k) for k in keys) for f, m in metas.items() if f.startswith(fam + '_m')}
        R.add('OK' if len(vals) <= 1 else 'FALLA', f'mallas {fam}',
              ', '.join(f'{k}={v}' for k, v in zip(keys, next(iter(vals)))) if len(vals) == 1 else
              (f'{len(vals)} configuraciones distintas' if vals else 'sin archivos'))
    return metas


# ---------------------------------------------------------------- 2. log
def check_log(R, logpath, N):
    if not os.path.isfile(logpath):
        R.add('AVISO', 'log', f'no encuentro {os.path.relpath(logpath, HERE)}')
        return
    txt = open(logpath, encoding='utf-8', errors='replace').read()
    R.add('OK' if '(inicio' in txt and '(fin' in txt else 'FALLA', 'log inicio/fin',
          'completo' if '(fin' in txt else 'no termino (falta la linea de fin)')
    errs = [l for l in txt.splitlines() if re.search(r'ERROR|Error|error:|Stacktrace|FALLA|Warning', l)
            and 'FALLA]' not in l]
    R.add('OK' if not errs else 'FALLA', 'log errores', 'ninguno' if not errs else f'{len(errs)} lineas: {errs[0][:120]}')
    n_chi = len(re.findall(r'-> \S*kernel_chi_m\d+n\d+\.csv', txt))
    n_tau = len(re.findall(r'-> \S*kernel_tau_m\d+n\d+\.csv', txt))
    R.add('OK' if (n_chi, n_tau) == (N * N, N * N) else 'FALLA', 'log archivos escritos',
          f'kernel_chi {n_chi}/{N * N}, kernel_tau {n_tau}/{N * N}')
    caus = [float(x) for x in re.findall(r'τ<-2 ([0-9.eE+-]+)', txt)]
    if caus:
        R.grade('log causalidad τ<-2 (peor)', max(caus), 1e-4, 1e-3)
    gib = [float(x) for x in re.findall(r'\(-2≤τ<0: ([0-9.eE+-]+)', txt)]
    if gib:
        R.add('INFO', 'log Gibbs -2≤τ<0 (peor)', f'{max(gib):.1e}  (oscilaciones del corte en ω, esperadas)')
    mem = []
    for x in re.findall(r'Ω τ_mem = ([0-9.eE+-]+|Inf|inf)', txt):
        try:
            mem.append(float(x))
        except ValueError:
            mem.append(float('inf'))
    if mem:
        mem = np.array(mem)
        R.add('INFO', 'log Ω τ_mem (1%)', f'mediana {np.median(mem):.3f}, max {mem.max():.3f}, '
              f'{int((mem >= 0.999 * 2 * np.pi).sum())} pares sin decaer al 1 % antes de T '
              '(oscilacion de los estados ligados, η_τ pequeño)')


# ---------------------------------------------------------------- 3. consistencia
def read_chi0(path):
    d = pd.read_csv(path, comment='#')
    return d


def check_chi0(R, d, N):
    pa, pb = os.path.join(d, f'kernel_chi0_sites1-{N}.csv'), os.path.join(d, f'kernel_chi0_kernel_sites1-{N}.csv')
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        return None
    a, b = read_chi0(pa), read_chi0(pb)
    if len(a) != 6 * N * N or not finite_frame(a):
        R.add('FALLA', 'kernel_chi0 filas/NaN', f'{len(a)} filas (esperadas {6 * N * N})')
    m = a.merge(b, on=['m', 'n', 'comp'], suffixes=('_a', '_b'))
    for q, lab in (('chi0', 'χ̌(0)'), ('idchi0', 'i∂χ̌(0)')):
        za, zb = z(m, f'Re_{q}_a', f'Im_{q}_a'), z(m, f'Re_{q}_b', f'Im_{q}_b')
        R.grade(f'--chi0 vs --kernel {lab}', np.abs(za - zb).max() / np.abs(za).max(), 1e-3, 1e-2,
                extra='(max|Δ| / max|valor| del conjunto)')
    # realidad del k = 0 en ω = 0, relativa al maximo del par
    worst = [0.0, 0.0]
    for (i, j), g in a.groupby(['m', 'n']):
        g0 = g[g['comp'].isin(K0)]
        for w, q in enumerate(('chi0', 'idchi0')):
            zz = z(g0, f'Re_{q}', f'Im_{q}')
            worst[w] = max(worst[w], np.abs(zz.imag).max() / max(np.abs(zz).max(), 1e-300))
    R.grade('realidad k=0 en ω=0: χ̌', worst[0], 1e-6, 1e-3, extra='(max|Im| / max|χ̌_0| del par)')
    R.grade('realidad k=0 en ω=0: i∂χ̌', worst[1], 1e-3, 1e-2, extra='(max|Im| / max|i∂χ̌_0| del par)')
    return a, b


def check_kernel_files(R, d, N, chi0k):
    sym, ov, fd, row0, rows, nan = 0.0, 0.0, 0.0, 0.0, set(), []
    fdw = ('', 0.0)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            p = os.path.join(d, f'kernel_chi_m{i}n{j}.csv')
            if not os.path.isfile(p):
                continue
            kc = pd.read_csv(p, comment='#')
            rows.add((len(kc), int((kc['window'] == 'gamma').sum()), int((kc['window'] == 'Omega').sum())))
            if not finite_frame(kc):
                nan.append((i, j))
            ga, om = kc[kc['window'] == 'gamma'].sort_values('omega'), kc[kc['window'] == 'Omega'].sort_values('omega')
            # simetria χ̌_0(-ω) = χ̌_0(ω)* en las dos ventanas
            for w in (ga, om):
                for k in K0:
                    for q in ('chi', 'idchi'):
                        zz = z(w, f'Re_{q}_{k}', f'Im_{q}_{k}')
                        sym = max(sym, np.abs(zz - np.conj(zz[::-1])).max() / max(np.abs(zz).max(), 1e-300))
            # ventanas en los ω comunes
            cols = [c for c in kc.columns if c not in ('window', 'omega')]
            gi, oi = ga.set_index('omega'), om.set_index('omega')
            com = gi.index.intersection(oi.index)
            if len(com):
                scale = np.abs(oi[cols].to_numpy()).max()
                ov = max(ov, np.abs(gi.loc[com, cols].to_numpy() - oi.loc[com, cols].to_numpy()).max() / scale)
            # derivada exacta frente a diferencia finita (ventana Ω, paso dω)
            wv = om['omega'].to_numpy()
            h = wv[1] - wv[0]
            for k in K:
                zz = z(om, f'Re_chi_{k}', f'Im_chi_{k}')
                dz = z(om, f'Re_idchi_{k}', f'Im_idchi_{k}')
                e = np.abs(1j * (zz[2:] - zz[:-2]) / (2 * h) - dz[1:-1]).max() / max(np.abs(dz).max(), 1e-300)
                if e > fd:
                    fd, fdw = e, (f'({i},{j}) {k}', e)
            # fila ω = 0 frente al resumen de --kernel
            if chi0k is not None:
                r0 = om[om['omega'] == 0.0]
                g = chi0k[(chi0k['m'] == i) & (chi0k['n'] == j)].set_index('comp')
                if len(r0) == 1 and len(g) == 6:
                    for k in K:
                        a = r0.iloc[0][f'Re_chi_{k}'] + 1j * r0.iloc[0][f'Im_chi_{k}']
                        b = g.loc[k, 'Re_chi0'] + 1j * g.loc[k, 'Im_chi0']
                        row0 = max(row0, abs(a - b) / max(abs(b), 1e-12))
    R.add('OK' if len(rows) == 1 else 'FALLA', 'kernel_chi filas',
          (lambda r: f'{r[0]} filas ({r[1]} ventana γ, {r[2]} ventana Ω) en todos')(next(iter(rows))) if len(rows) == 1
          else f'{len(rows)} tamaños distintos: {sorted(rows)[:3]}')
    R.add('OK' if not nan else 'FALLA', 'kernel_chi NaN/Inf', 'ninguno' if not nan else f'en {nan[:5]}')
    R.grade('simetria χ̌_0(-ω) = χ̌_0(ω)*', sym, 1e-3, 1e-2, extra='(χ̌ e i∂χ̌, ventanas γ y Ω)')
    R.grade('ventanas γ y Ω en ω comunes', ov, 1e-10, 1e-6)
    R.grade('i∂χ̌ exacta vs diferencia finita', fd, 3e-2, 3e-1, extra=f'(ventana Ω; peor {fdw[0]})')
    if chi0k is not None:
        R.grade('kernel_chi(ω=0) vs resumen --kernel', row0, 1e-9, 1e-6)


def check_tau(R, d, N, chi0a, metas):
    caus, reim, integ, rows, nan = 0.0, 0.0, 0.0, set(), []
    worst = ''
    eta_w = float(next((m['eta'] for f, m in metas.items() if f.startswith('kernel_chi0_sites')), 'nan'))
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            p = os.path.join(d, f'kernel_tau_m{i}n{j}.csv')
            if not os.path.isfile(p):
                continue
            eta_t = float(metas.get(f'kernel_tau_m{i}n{j}.csv', {}).get('eta', 'nan'))
            kt = pd.read_csv(p, comment='#')
            rows.add(len(kt))
            if not finite_frame(kt):
                nan.append((i, j))
            tau = kt['tau'].to_numpy()
            dt = tau[1] - tau[0]
            vals = kt[[c for c in kt.columns if c != 'tau']].to_numpy()
            mx = np.abs(vals).max()
            caus = max(caus, np.abs(vals[tau < -2.0]).max() / mx)
            for k in K0:
                zz = z(kt, f'Re_chi_{k}', f'Im_chi_{k}')
                reim = max(reim, np.abs(zz.imag).max() / max(np.abs(zz).max(), 1e-300))
            if chi0a is not None and np.isfinite(eta_w) and np.isfinite(eta_t):
                wgt = np.exp(-2.0 * max(eta_w - eta_t, 0.0) * np.abs(tau))
                g = chi0a[(chi0a['m'] == i) & (chi0a['n'] == j)].set_index('comp')
                ref = np.array([g.loc[k, 'Re_chi0'] + 1j * g.loc[k, 'Im_chi0'] for k in K])
                got = np.array([np.trapezoid(z(kt, f'Re_chi_{k}', f'Im_chi_{k}') * wgt, dx=dt) for k in K])
                e = np.abs(got - ref).max() / np.abs(ref).max()
                if e > integ:
                    integ, worst = e, f'({i},{j})'
    if not rows:
        return
    R.add('OK' if len(rows) == 1 else 'FALLA', 'kernel_tau filas',
          f'{next(iter(rows))} filas en todos' if len(rows) == 1 else f'{len(rows)} tamaños distintos')
    R.add('OK' if not nan else 'FALLA', 'kernel_tau NaN/Inf', 'ninguno' if not nan else f'en {nan[:5]}')
    R.grade('causalidad τ<-2', caus, 1e-4, 1e-3, extra='(max|χ̌(τ<-2)| / max|χ̌|)')
    R.grade('realidad k=0 en τ', reim, 1e-8, 1e-4)
    if chi0a is not None:
        R.grade('∫dτ χ̌_k(τ) = χ̌_k(0)', integ, 3e-2, 2e-1,
                extra=f'(error / max|χ̌(0)| del par; peor {worst}; diferencia de η entre ω y τ)')


# ---------------------------------------------------------------- 4. LLG
def llg_from_chi0(x):
    """Mismas formulas que llg_elements.jl (x: dict comp -> complejo)."""
    a, b, c = x['xx0'].real, x['yy0'].real, x['xz0'].real
    xy1, yx1, xx2 = x['xy1'], x['yx1'], x['xx2']
    return [(2 * a + b) / 3, (a - b) / 3 + 2 * xx2.real, -2 * (a - b) / 3, (a - b) / 3 - 2 * xx2.real,
            2 * xx2.imag, xy1.real + yx1.real, xy1.imag + yx1.imag, yx1.imag - xy1.imag, -c, xy1.real - yx1.real]


def check_llg(R, d, N, chi0a, metas):
    pl = os.path.join(d, f'llg_elements_sites1-{N}.csv')
    if not os.path.isfile(pl) or chi0a is None:
        return None
    L = pd.read_csv(pl)
    if not finite_frame(L):
        R.add('FALLA', 'llg NaN/Inf', 'hay valores no finitos')
    val = {(r.i, r.j, r.elemento): r.valor for r in L.itertuples()}
    # recalculo desde kernel_chi0 (detecta que llg y chi0 tengan signos distintos)
    err, scale = 0.0, 0.0
    for (i, j), g in chi0a.groupby(['m', 'n']):
        g = g.set_index('comp')
        for names, q in ((NAMES_A, 'chi0'), (NAMES_D, 'idchi0')):
            x = {k: g.loc[k, f'Re_{q}'] + 1j * g.loc[k, f'Im_{q}'] for k in K}
            for nm, v in zip(names, llg_from_chi0(x)):
                ref = val.get((i, j, nm), np.nan)
                err = max(err, abs(v - ref)); scale = max(scale, abs(ref))
    R.grade('llg_elements vs kernel_chi0', err / scale if scale else np.nan, 1e-9, 1e-6,
            extra='(max|Δ| / max|valor|; si es ~2 uno de los dos tiene el signo viejo)')
    # signo fisico on-site
    Jmm = np.array([val[(m, m, 'J')] for m in range(1, N + 1)])
    amm = np.array([val[(m, m, 'α')] for m in range(1, N + 1)])
    R.add('OK' if (Jmm > 0).all() else 'FALLA', 'J_mm > 0 (on-site paramagnetico)',
          f'J_mm ∈ [{Jmm.min():.4f}, {Jmm.max():.4f}]')
    R.add('OK' if (amm < 0).all() else 'FALLA', 'α_mm < 0 (damping, no antidamping)',
          f'α_mm ∈ [{amm.min():.4f}, {amm.max():.4f}]')
    # torsion de gauge: atan2(t^{y,A}, J^S + Γ^{xx,S}) = 2 θ_SO d (mod π)
    m0 = next((m for f, m in metas.items() if f.startswith('kernel_chi0_sites')), None)
    if m0 is None:
        return val
    thso = math.atan2(float(m0['gammaSO']), float(m0['gamma']))
    vs, rec = [], []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            S = lambda nm: (val[(i, j, nm)] + val[(j, i, nm)]) / 2
            A = lambda nm: (val[(i, j, nm)] - val[(j, i, nm)]) / 2
            x, y = S('J') + S('Γxx'), A('ty')
            vs.append(math.hypot(x, y))
            dif = (math.atan2(y, x) - 2 * thso * (j - i) + math.pi / 2) % math.pi - math.pi / 2
            rec.append((abs(dif), i, j, math.hypot(x, y)))
    vmax = max(vs)
    rec = [r for r in rec if r[3] > 1e-3 * vmax]
    w = max(rec)
    R.grade('torsion de gauge: ángulo vs 2θ_SO d', w[0], 1e-3, 1e-2, fmt='{:.1e} rad',
            extra=f'({len(rec)} pares i<j; peor ({w[1]},{w[2]}))')
    return val


# ---------------------------------------------------------------- principal
def check_set(tag, N, skip_tau):
    R = Report(tag)
    d = os.path.join(HERE, 'output', tag, 'data')
    metas = check_integrity(R, d, N)
    check_log(R, os.path.join(HERE, 'logs', f'{tag}.log'), N)
    ab = check_chi0(R, d, N)
    chi0a, chi0k = (ab if ab is not None else (None, None))
    check_kernel_files(R, d, N, chi0k)
    if not skip_tau:
        check_tau(R, d, N, chi0a, metas)
    val = check_llg(R, d, N, chi0a, metas)
    return R, val


def main():
    ap = argparse.ArgumentParser(description='Revisa los resultados del kernel (sin recalcular).')
    ap.add_argument('--tag', action='append', default=[], help='output/<tag> (se puede repetir; defecto: todos)')
    ap.add_argument('--sites', type=int, default=10)
    ap.add_argument('--skip-tau', action='store_true', help='no lee los kernel_tau (mucho mas rapido)')
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'check_report.txt'))
    args = ap.parse_args()
    tags = args.tag or sorted(os.path.basename(os.path.dirname(p))
                              for p in glob.glob(os.path.join(HERE, 'output', 'setup*', 'data')))
    if not tags:
        raise SystemExit('no hay carpetas output/setup*/data')

    lines, table, info = [], [], []
    t0 = time.time()
    for tag in tags:
        t1 = time.time()
        print(f'== {tag} ...', flush=True)
        R, val = check_set(tag, args.sites, args.skip_tau)
        lines.append('=' * 100 + f'\n{tag}   ({time.time() - t1:.0f} s)\n' + '=' * 100)
        for lvl, chk, msg in R.items:
            lines.append(f'  [{lvl:5s}] {chk:42s} {msg}')
        c = R.count()
        table.append((tag, c))
        if val is not None:
            g = lambda i, j, nm: val.get((i, j, nm), np.nan)
            info.append((tag, g(1, 1, 'J'), g(1, 2, 'J'), (g(1, 2, 'ty') - g(2, 1, 'ty')) / 2, g(1, 1, 'α'),
                         (g(1, 3, 'α') + g(3, 1, 'α')) / 2))
        print(f'   OK {c["OK"]}  AVISO {c["AVISO"]}  FALLA {c["FALLA"]}', flush=True)

    out = ['check_kernel_results.py   ' + time.strftime('%Y-%m-%d %H:%M') +
           f'   ({len(tags)} conjuntos, {time.time() - t0:.0f} s{", sin kernel_tau" if args.skip_tau else ""})', '']
    out.append(f'{"conjunto":46s} {"OK":>4s} {"AVISO":>6s} {"FALLA":>6s}')
    for tag, c in table:
        out.append(f'{tag:46s} {c["OK"]:4d} {c["AVISO"]:6d} {c["FALLA"]:6d}')
    if info:
        out += ['', 'valores de referencia (comparar entre conjuntos):',
                f'{"conjunto":46s} {"J_11":>10s} {"J_12":>10s} {"t^yA_12":>10s} {"α_11":>10s} {"α^S_13":>10s}']
        for t, *v in info:
            out.append(f'{t:46s} ' + ' '.join(f'{x:+10.3e}' for x in v))
    out += [''] + lines
    txt = '\n'.join(out) + '\n'
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print('\n' + '\n'.join(out[:len(table) + 4 + (len(info) + 3 if info else 0)]))
    print(f'\ndetalle en {args.out}')
    fails = sum(c['FALLA'] for _, c in table)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
