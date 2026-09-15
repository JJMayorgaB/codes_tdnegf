#!/usr/bin/env python3
"""Config compartida por plot_spin_currents.py y animate_oscillators.py.

Variante SIN g1: la cadena es g2 (libres) | g3 (driver) | g4 (libres), asi que
no hay t_on_g1 ni barrido en k. La carpeta de salida y los tiempos del protocolo
se derivan de oscillators.jl y del params.txt de cada corrida, para que los
scripts nunca queden desincronizados de los datos que estan graficando.
"""

import math
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JL_PATH = os.path.join(SCRIPT_DIR, 'oscillators.jl')
BASE_OUT = os.path.join(SCRIPT_DIR, 'output')

_NAMES = {'gso': 'γso', 'jsd': 'j_sd', 'theta': 'θ_max', 'Omega': 'Ω',
          't_rise': 't_rise', 't_on_g3': 't_on_g3',
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
    m = re.search(r'^const\s+N_SPINS\s*=\s*(\d+)', src, re.M)
    if m:
        out['N_SPINS'] = int(m.group(1))
    return out


def parse_groups(jl_path=JL_PATH):
    """Grupos de espines definidos en el const GROUPS de oscillators.jl.
    Aqui solo existen g2, g3 y g4 (g1 fue eliminado)."""
    with open(jl_path, encoding='utf-8') as f:
        src = f.read()
    groups = {}
    for name in ('g2', 'g3', 'g4'):
        m = re.search(name + r'\s*=\s*(\d+)\s*:\s*(\d+)', src)
        if m:
            groups[name] = list(range(int(m.group(1)), int(m.group(2)) + 1))
    return groups


def elec_site(m, jl_path=JL_PATH):
    """Sitio electronico del espin m, leyendo elec_site() de oscillators.jl.

    El mapeo depende de N_BUF (sitios desnudos entre el lead y el primer momento),
    asi que no se puede hardcodear como 2*m.
    """
    with open(jl_path, encoding='utf-8') as f:
        src = f.read()
    expr = re.search(r'elec_site\(m::Int\)\s*=\s*(.+?)\s*$', src, re.M)
    if not expr:
        return 2 * m
    env = {'m': m}
    nbuf = re.search(r'^const\s+N_BUF\s*=\s*(\d+)', src, re.M)
    env['N_BUF'] = int(nbuf.group(1)) if nbuf else 0
    return int(eval(expr.group(1).split('#')[0].strip(), {'__builtins__': {}}, env))


def _fmt(x):
    return repr(round(float(x), 4)).replace('.', 'p').replace('-', 'm')


def run_tag(c):
    """Replica exacta de param_tag() en oscillators.jl."""
    return (f"n{int(c['N_SPINS'])}_gso{_fmt(c['gso'])}_jsd{_fmt(c['jsd'])}"
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


def protocol(run_dir):
    """Tiempos del protocolo de una corrida, como dict.

    Combina el params.txt de la carpeta con los const de oscillators.jl. Las
    claves que no existan en ninguna de las dos fuentes valen None.
    """
    p = load_params_txt(run_dir)
    try:
        c = parse_julia_consts()
    except Exception:
        c = {}

    def pick(jl_key, *txt_keys):
        for k in txt_keys:
            if k in p:
                return p[k]
        return c.get(jl_key)

    return {
        't_on_g3': pick('t_on_g3', 't_on_g3'),
        't_rise':  pick('t_rise', 't_rise'),
        't_relax': pick('t_relax', 't_relax'),
        't_final': pick('t_final', 't_final'),
        'Omega':   pick('Omega', 'Ω', 'Omega'),
    }


def resolve_run_dir(run_tag_arg=None, base_out=BASE_OUT, prefix='steady_state_'):
    """Carpeta de la corrida: --run-tag explicito > la que corresponde a los
    parametros actuales de oscillators.jl > la unica subcarpeta que haya."""
    if run_tag_arg:
        d = os.path.join(base_out, run_tag_arg)
        if os.path.isdir(d):
            return d
        return os.path.join(base_out, prefix + run_tag_arg)

    try:
        d = os.path.join(base_out, prefix + run_tag(parse_julia_consts()))
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
