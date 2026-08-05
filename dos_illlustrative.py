#!/usr/bin/env python3


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── estilo ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[utf8]{inputenc}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'axes.axisbelow': True,
})

C_WFB  = plt.cm.seismic(0.88)   # rojo
C_SOFT = plt.cm.seismic(0.12)   # azul
C_SEMI = 'black'

# ── parámetros ────────────────────────────────────────────────────────────────
t      = 1.0          # hopping del lead  ->  eje en w/t
e_c    = 2.0 * t      # borde de banda (semicírculo de semiancho 2t)
rho_0  = 1.0          # altura máxima común
T_edge = 0.04 * t     # ancho del borde suave; = 0.4*Gamma con e_c/Gamma = 20

w = np.linspace(-3.5 * t, 3.5 * t, 4000)

fermi = lambda x: 1.0 / (1.0 + np.exp(np.clip(x / T_edge, -500, 500)))

rho_wfb  = np.full_like(w, rho_0)
rho_soft = rho_0 * (fermi(w - e_c) - fermi(w + e_c))
rho_semi = np.where(np.abs(w) < e_c,
                    rho_0 * np.sqrt(np.clip(1.0 - (w / e_c) ** 2, 0.0, None)),
                    0.0)

# ── figura ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 3.5))

ax.plot(w / t, rho_wfb,  '-',  color=C_WFB,  lw=1.5, label=r'WFB')
ax.plot(w / t, rho_soft, '-',  color=C_SOFT, lw=1.5, label=r'soft cutoff')
ax.plot(w / t, rho_semi, '-',  color=C_SEMI, lw=1.5, label=r'TDNEGF')

ax.set_xlabel(r'$\omega/t$')
ax.set_ylabel(r'$\rho(\omega)/\rho_0$', labelpad=10)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_yticks([0.0, 0.5, 1.0])
ax.tick_params(axis='both', direction='in', bottom=True, top=True, right=True, left=True, width=1.0, length=3.5)
ax.legend(frameon=True, edgecolor='black', framealpha=0.0, fancybox=False,
          borderpad=0.5, handlelength=1.8, labelspacing=0.3, loc='upper right', ncol=3, bbox_to_anchor=(1.0,1.15))

plt.tight_layout()
plt.savefig('output/dos_models.png', bbox_inches='tight', dpi=300)
plt.savefig('output/dos_models.svg', bbox_inches='tight')
print('output/dos_models.png')