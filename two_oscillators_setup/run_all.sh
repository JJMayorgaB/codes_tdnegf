#!/usr/bin/env bash
# run_all.sh -- corridas de TDNEGF con precesion en y:
#   8 conjuntos de parametros (los de fully_analytical_computations/run_cluster.sh)
#   x 6 configuraciones:  DFF  F1-2, F3-6, F5-9   y   FDF  Fm1-1, Fm3-3, Fm4-4
# = 48 corridas, lanzadas por tandas (MAXJOBS a la vez, BLAS hilos cada una).
# Se salta las corridas que ya tienen spins_t.csv con el protocolo actual
# (t_on = 5 (2π/0.005) para todo Ω). Las hechas con otro protocolo se borran y se
# corren de nuevo en la misma carpeta (no se guardan datos viejos).
#
# Salida:  y_precession/output/{DFF,FDF}/F…/<param_tag>/
# Logs:    y_precession/output/logs/{DFF,FDF}/F…/<param_tag>.log
# Al final, las animaciones de cada conjunto (animation_configs.py):
#   animations/y_precession/DFF/<param_tag>.mp4       columnas F1-2 | F3-6 | F5-9
#   animations/y_precession/FDF/<param_tag>.mp4       columnas Fm1-1 | Fm3-3 | Fm4-4
#   animations/y_precession/DFF_FDF/<param_tag>.mp4   las 6 (DFF arriba, FDF abajo)
# cada uno en dos estilos: <param_tag>.mp4 (cadena) y bloch_<param_tag>.mp4 (esfera de Bloch)
#
# Uso (en tmux, desde two_oscillators_setup; antes, una vez: bash migrate_layout.sh):
#   bash run_all.sh                      # 16 corridas a la vez, 8 hilos de BLAS cada una
#   MAXJOBS=12 BLAS=8 bash run_all.sh
#   bash run_all.sh 6 7                  # solo esos conjuntos
#   NOANIM=1 bash run_all.sh             # sin animaciones
set -u
cd "$(dirname "$0")"

MAXJOBS=${MAXJOBS:-16}
BLAS=${BLAS:-8}
JULIA=${JULIA:-julia}
AX=y
OUT=${AX}_precession/output

#      GSO                JSD  THETA OMEGA
SETS=("0.1                0.2  10    0.005"
      "0.1                0.2  30    0.005"
      "0.5                0.2  10    0.005"
      "0.1                0.2  10    0.02"
      "0.5                0.2  30    0.02"
      "0.2588190451025208 0.2  10    0.005"
      "0.1950903220161283 0.2  10    0.005"
      "0.1                0.5  10    0.005")

#        carpeta        FPOS
CONFIGS=("DFF/F1-2      1,2"
         "DFF/F3-6      3,6"
         "DFF/F5-9      5,9"
         "FDF/Fm1-1     -1,1"
         "FDF/Fm3-3     -3,3"
         "FDF/Fm4-4     -4,4")

# mismo param_tag que common.jl (round(x; digits = 4), '.' -> 'p'; buf4, adyn0p0)
tag() {
  python3 -c "import sys
f=lambda x: str(round(float(x),4)).replace('.','p').replace('-','m')
g,j,th,om=sys.argv[1:]
print(f'gso{f(g)}_jsd{f(j)}_th{round(float(th))}deg_Om{f(om)}_buf4_adyn0p0')" "$@"
}

if [ $# -gt 0 ]; then idx=("$@"); else idx=(1 2 3 4 5 6 7 8); fi

echo "conjuntos: ${idx[*]}   corridas a la vez: $MAXJOBS   hilos BLAS: $BLAS"
n_lanzadas=0
for k in "${idx[@]}"; do
  read -r g j th om <<< "${SETS[$((k-1))]}"
  t=$(tag "$g" "$j" "$th" "$om")
  for cf in "${CONFIGS[@]}"; do
    read -r dir fpos <<< "$cf"
    if [ -f "$OUT/$dir/$t/spins_t.csv" ]; then
      # solo se salta si se corrio con el protocolo actual (t_on = 5 (2π/0.005) para todo Ω)
      if grep -q "t_on = t_relax = 6283.18" "$OUT/$dir/$t/params.txt"; then
        echo "  ya existe: $dir/$t"; continue
      fi
      echo "  protocolo viejo (t_on distinto): se borra $dir/$t y se corre de nuevo"
      rm -rf "$OUT/$dir/$t"
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
    mkdir -p "$OUT/logs/$dir"
    log="$OUT/logs/$dir/$t.log"
    echo "  lanzando $dir/$t  ($(date +%H:%M))"
    ( GSO=$g JSD=$j THETA=$th OMEGA=$om OPENBLAS_NUM_THREADS=$BLAS \
        "$JULIA" run_config.jl "$AX" "$fpos" > "$log" 2>&1 \
      && echo "  terminado: $dir/$t" || echo "  FALLO: $dir/$t (ver $log)" ) &
    n_lanzadas=$((n_lanzadas + 1))
  done
done
wait
echo "corridas listas ($n_lanzadas lanzadas)"

[ "${NOANIM:-0}" = 1 ] && exit 0
for k in "${idx[@]}"; do
  read -r g j th om <<< "${SETS[$((k-1))]}"
  python3 animation_configs.py "$(tag "$g" "$j" "$th" "$om")" --frame-skip 10 \
    || echo "  animaciones incompletas para el conjunto $k"
done
