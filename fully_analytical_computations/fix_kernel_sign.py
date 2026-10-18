#!/usr/bin/env python3
"""
fix_kernel_sign.py -- corrige el signo global del kernel en CSV ya calculados.

El prefactor correcto es +i J_sd (⟨σ⟩ = -i Tr[σ 𝒢^<] y 𝒢^< = G^< - J_sd ∫[...]); los
CSV escritos antes con -i J_sd tienen TODO el kernel con el signo opuesto. Como el
kernel es lineal en ese prefactor, basta cambiar el signo de los valores, sin
recalcular. Los CSV corregidos (y los nuevos de kernel_elements.jl) llevan
chi_sign=1 en el encabezado; un archivo que ya lo tiene no se vuelve a tocar, asi
que correr el script dos veces es seguro.

Que cambia (en output/<tag>/data/):
    kernel_chi_m{i}n{j}.csv      todas las columnas de χ̌ e i∂_ωχ̌ (no window ni omega)
    kernel_chi0_*.csv            Re/Im de χ̌(0) e i∂_ωχ̌(0)
    kernel_tau_m{i}n{j}.csv      todas las columnas de χ̌(τ) (no tau)
    llg_elements_<stem>.csv      columna valor   (escala = |valor| no cambia)
    llg_elements_SA_<stem>.csv   columnas S y A
  los llg_elements solo si su kernel_chi0_<stem>.csv de origen estaba sin corregir.
No cambia kernel_elements_*.csv (armonicos de G, no dependen de ese prefactor).

El cambio es de texto: se quita o se pone el '-' de cada numero, sin reescribirlo,
asi que las cifras significativas quedan exactamente iguales (los ceros exactos no
se tocan).

Uso:
    python fix_kernel_sign.py --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005
    python fix_kernel_sign.py --all              # todas las carpetas output/*/data y output/conv/*
    python fix_kernel_sign.py --all --dry-run    # solo dice que haria
"""

import argparse
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = 'chi_sign=1'


def flip(s):
    t = s.strip()
    tail = s[len(s.rstrip()):]                      # p. ej. '\r' si el archivo tiene CRLF
    try:
        if float(t) == 0.0:
            return s
    except ValueError:
        return s
    if t.startswith('-'):
        return t[1:] + tail
    if t.startswith('+'):
        return '-' + t[1:] + tail
    return '-' + t + tail


def has_mark(path):
    with open(path, encoding='utf-8') as fh:
        head = fh.readline()
    return head.startswith('#') and MARK in head.split()


def fix_file(path, keep, add_mark, dry):
    """Invierte el signo de las columnas que no estan en keep (por nombre)."""
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    i = 0
    if lines[0].startswith('#'):
        if add_mark:
            lines[0] = lines[0].rstrip() + ' ' + MARK
        i = 1
    cols = lines[i].split(',')
    idx = [k for k, c in enumerate(cols) if c not in keep]
    n = 0
    for r in range(i + 1, len(lines)):
        if not lines[r]:
            continue
        f = lines[r].split(',')
        for k in idx:
            f[k] = flip(f[k])
        lines[r] = ','.join(f)
        n += 1
    if not dry:
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='') as fh:
            fh.write('\n'.join(lines))
        os.replace(tmp, path)
    return n, [cols[k] for k in idx]


def fix_dir(d, dry):
    print(f'== {os.path.relpath(d, HERE)}')
    done = skipped = 0
    for pat, keep in (('kernel_chi_m*n*.csv', {'window', 'omega'}),
                      ('kernel_tau_m*n*.csv', {'tau'}),
                      ('kernel_chi0_*.csv', {'m', 'n', 'comp'})):
        for p in sorted(glob.glob(os.path.join(d, pat))):
            if has_mark(p):
                skipped += 1
                continue
            stem = os.path.basename(p)[len('kernel_chi0_'):-4] if pat.startswith('kernel_chi0') else None
            fix_file(p, keep, True, dry)
            done += 1
            # llg_elements derivados de este kernel_chi0 (sin encabezado: van con su origen)
            if stem is not None:
                for q, kp in ((os.path.join(d, f'llg_elements_{stem}.csv'), {'i', 'j', 'tipo', 'elemento', 'escala'}),
                              (os.path.join(d, f'llg_elements_SA_{stem}.csv'), {'i', 'j', 'tipo', 'elemento'})):
                    if os.path.isfile(q):
                        _, c = fix_file(q, kp, False, dry)
                        print(f'   {os.path.basename(q)}  (columnas {", ".join(c)})')
                        done += 1
    print(f'   {"se corregirian" if dry else "corregidos"}: {done}   ya tenian {MARK}: {skipped}')


def main():
    ap = argparse.ArgumentParser(description='Corrige el signo global del kernel (-i J_sd -> +i J_sd) en CSV ya calculados.')
    ap.add_argument('--tag', action='append', default=[], help='carpeta output/<tag>/data (se puede repetir)')
    ap.add_argument('--all', action='store_true', help='todas las output/*/data y output/conv/*')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    dirs = [os.path.join(HERE, 'output', t, 'data') for t in args.tag]
    if args.all:
        dirs += sorted(glob.glob(os.path.join(HERE, 'output', '*', 'data')))
        dirs += sorted(d for d in glob.glob(os.path.join(HERE, 'output', 'conv', '*')) if os.path.isdir(d))
    if not dirs:
        raise SystemExit('nada que hacer: usa --tag <tag> o --all')
    for d in dict.fromkeys(dirs):
        if not os.path.isdir(d):
            print(f'   no existe: {d}')
            continue
        fix_dir(d, args.dry_run)


if __name__ == '__main__':
    main()
