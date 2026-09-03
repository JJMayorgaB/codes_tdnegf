#!/usr/bin/env python3
"""Config compartida por plot_spin_currents.py y animate_oscillators.py.

La carpeta de salida y los tiempos del protocolo se derivan de oscillators.jl
(fuente única de verdad) y del params.txt de cada corrida, para que los scripts
nunca queden desincronizados de los datos que están graficando.
"""

import math
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JL_PATH = os.path.join(SCRIPT_DIR, 'oscillators.jl')
BASE_OUT = os.path.join(SCRIPT_DIR, 'output')

_NAMES = {'gso': 'γso', 'jsd': 'j_sd', 'theta': 'θ_max', 'Omega': 'Ω',
          't_rise': 't_rise', 't_on_g3': 't_on_g3', 't_on_g1': 't_on_g1',
          't_relax': 't_relax', 't_final': 't_final'}


def _eval_julia(expr):
    expr = expr.split('#')[0].strip()
    expr = expr.replace('sqrt(', 'math.sqrt(').replace('deg2rad(', 'math.radians(')
    return float(eval(expr, {'__builtins__': {}}, {'math': math}))


def parse_julia_consts(jl_path=JL_PATH):
    with open(jl_path, encoding='utf-8') as f:
        src = f.read()
    out = {}
    for key, jname in _NAMES.items():
        m = re.search(r'^const\s+' + re.escape(jname) + r'\s*=\s*(.+)$', src, re.M)
        if m:
            out[key] = _eval_julia(m.group(1))
    return out


def _fmt(x):
    return repr(round(float(x), 4)).replace('.', 'p').replace('-', 'm')


def run_tag(c):
    """Réplica exacta de run_tag() en oscillators.jl."""
    return (f"gso{_fmt(c['gso'])}_jsd{_fmt(c['jsd'])}"
            f"_th{round(math.degrees(c['theta']))}deg_Om{_fmt(c['Omega'])}")


def load_params_txt(run_dir):
    path = os.path.join(run_dir, 'params.txt')
    if not os.path.isfile(path):
        return {}
    vals = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            m = re.match(r'\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)', v)
            if m:
                vals[k.strip()] = float(m.group(1))
    return vals


def protocol_times(run_dir):
    """(t_on_g3, t_on_g1, Omega) del params.txt de la corrida; si falta alguno,
    se completa desde oscillators.jl."""
    p = load_params_txt(run_dir)

    def pick(*keys):
        for k in keys:
            if k in p:
                return p[k]
        return None

    t3, t1, om = pick('t_on_g3'), pick('t_on_g1'), pick('Ω', 'Omega')
    if None in (t3, t1, om):
        c = parse_julia_consts()
        t3 = c['t_on_g3'] if t3 is None else t3
        t1 = c['t_on_g1'] if t1 is None else t1
        om = c['Omega'] if om is None else om
    return t3, t1, om


def resolve_run_dir(run_tag_arg=None, base_out=BASE_OUT):
    """Carpeta de la corrida: --run-tag explícito > la que corresponde a los
    parámetros actuales de oscillators.jl > la única subcarpeta que haya."""
    if run_tag_arg:
        return os.path.join(base_out, run_tag_arg)

    try:
        d = os.path.join(base_out, run_tag(parse_julia_consts()))
        if os.path.isdir(d):
            return d
    except Exception:
        pass

    subdirs = ([d for d in sorted(os.listdir(base_out))
                if os.path.isdir(os.path.join(base_out, d))]
               if os.path.isdir(base_out) else [])
    if len(subdirs) == 1:
        return os.path.join(base_out, subdirs[0])

    raise SystemExit('No se pudo determinar la carpeta de la corrida: pasa --run-tag. '
                     'Subcarpetas en output/: ' + (', '.join(subdirs) or '(ninguna)'))
