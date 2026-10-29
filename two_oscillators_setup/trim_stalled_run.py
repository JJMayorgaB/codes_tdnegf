#!/usr/bin/env python3
"""
trim_stalled_run.py -- recorta una corrida de TDNEGF en la que el integrador de los
electrones se detuvo antes de t_final (p. ej. por maxiters) y el bucle siguio escribiendo
filas con el mismo t (los espines seguian moviendose con ρ congelada: datos invalidos).

Busca en spins_t.csv el primer t que no crece respecto al anterior (t_corte) y deja en
TODOS los CSV temporales solo las filas con t < t_corte:
    spins_t.csv, sites_rho_t.csv, bond_rho_t.csv, lead_currents_t.csv
Los originales se guardan como <archivo>.full.csv (no se vuelve a recortar si ya existen).
En params.txt se reemplaza t_final por el ultimo t valido (para que psd_setup.py elija la
ventana correcta) y se deja una linea con el t_final original; copia en params.full.txt.

Uso:
    python trim_stalled_run.py y_precession/output/DFF/F1-2/gso0p1_jsd0p2_th10deg_Om0p005_buf4_adyn0p0_np80
"""

import os
import re
import shutil
import sys

ARCHIVOS = ('spins_t.csv', 'sites_rho_t.csv', 'bond_rho_t.csv', 'lead_currents_t.csv')


def t_corte(path):
    """Primer t que no es mayor que el anterior (None si t crece en todo el archivo)."""
    prev = None
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            t = float(line.split(',', 1)[0])
            if prev is not None and t <= prev:
                return t, prev
            prev = t
    return None, prev


def recortar(path, tc):
    full = path[:-4] + '.full.csv'
    if not os.path.exists(full):
        shutil.move(path, full)
    n_in = n_out = 0
    with open(full, encoding='utf-8') as fi, open(path, 'w', encoding='utf-8', newline='') as fo:
        fo.write(next(fi))
        for line in fi:
            n_in += 1
            if float(line.split(',', 1)[0]) < tc - 1e-9:
                fo.write(line)
                n_out += 1
    return n_in, n_out


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = sys.argv[1]
    sp = os.path.join(run, 'spins_t.csv')
    src = sp[:-4] + '.full.csv' if os.path.exists(sp[:-4] + '.full.csv') else sp
    tc, tprev = t_corte(src)
    if tc is None:
        print(f'{run}: t crece en todo spins_t.csv (ultimo t = {tprev}); no hay nada que recortar')
        return
    txt = open(os.path.join(run, 'params.txt'), encoding='utf-8').read()
    T = 2 * 3.141592653589793 / float(re.search(r'Ω = ([0-9.eE+-]+)', txt).group(1))
    t_on = float(re.search(r't_on = t_relax = ([0-9.eE+-]+)', txt).group(1))
    print(f'{run}\n  el integrador se detuvo en t = {tc:.4f} ({tc / T:.2f} T, {(tc - t_on) / T:.2f} periodos despues de t_on)')
    t_last = None
    for f in ARCHIVOS:
        p = os.path.join(run, f)
        if not os.path.exists(p) and not os.path.exists(p[:-4] + '.full.csv'):
            print(f'  (no esta {f})')
            continue
        n_in, n_out = recortar(p, tc)
        print(f'  {f}: {n_out} de {n_in} filas')
    with open(os.path.join(run, 'spins_t.csv'), encoding='utf-8') as fh:
        for line in fh:
            pass
        t_last = float(line.split(',', 1)[0])
    # params.txt: t_final -> ultimo t valido
    pfull = os.path.join(run, 'params.full.txt')
    if not os.path.exists(pfull):
        shutil.copy(os.path.join(run, 'params.txt'), pfull)
    base = open(pfull, encoding='utf-8').read()
    m = re.search(r't_final = ([0-9.eE+-]+)\s+\(([0-9.]+) periodos de driver\)', base)
    nuevo = base.replace(m.group(0), f't_final = {t_last}   ({(t_last - t_on) / T:.2f} periodos de driver)')
    nuevo += (f'RECORTADA con trim_stalled_run.py: el integrador se detuvo en t = {tc}; '
              f't_final original = {m.group(1)} ({m.group(2)} periodos)\n')
    open(os.path.join(run, 'params.txt'), 'w', encoding='utf-8').write(nuevo)
    print(f'  params.txt: t_final = {t_last:.4f} ({t_last / T:.2f} T); original en params.full.txt')


if __name__ == '__main__':
    main()
