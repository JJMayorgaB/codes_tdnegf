#!/usr/bin/env bash
# 4 conjuntos de parametros x {z,y}_precession x {setup1,setup2}, en paralelo;
# luego una animacion por conjunto con sus 4 corridas.
# Uso (en tmux):  cd ~/codes_tdnegf/two_oscillators_setup && bash run_all.sh
set -e
cd "$(dirname "$0")"

#      GSO  THETA OMEGA
SETS=("0.1 10 0.005"     # base (tdnegf_single_spin.jl)
      "0.5 10 0.005"     # lam05: Markoviano, mas no reciprocidad
      "0.1 30 0.005"     # th30:  Markoviano, mas no reciprocidad
      "0.5 30 0.02")     # combo: no Markoviano

tag() { echo "gso${1/./p}_jsd0p2_th${2}deg_Om${3/./p}_buf4_adyn0p0"; }

for s in "${SETS[@]}"; do
  read g th om <<< "$s"
  for d in z_precession y_precession; do
    mkdir -p $d/output/logs
    for st in setup1 setup2; do
      (cd $d && GSO=$g THETA=$th OMEGA=$om OPENBLAS_NUM_THREADS=8 \
         julia $st.jl > output/logs/${st}_$(tag $g $th $om).log 2>&1) &
    done
  done
done
wait

ls */output/setup*_*/spins_t.csv | wc -l | grep -qx $(( ${#SETS[@]} * 4 )) || \
  { echo "alguna corrida fallo: revisar */output/logs/"; exit 1; }

for s in "${SETS[@]}"; do
  read g th om <<< "$s"
  t=$(tag $g $th $om)
  python3 animation.py z_precession/output/setup1_$t z_precession/output/setup2_$t \
                       y_precession/output/setup1_$t y_precession/output/setup2_$t \
                       --frame-skip 10 --out animations/$t.mp4
done
