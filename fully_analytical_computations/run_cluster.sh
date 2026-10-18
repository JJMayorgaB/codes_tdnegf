#!/usr/bin/env bash
# run_cluster.sh -- calcula TODOS los datos (sin figuras) de los 8 conjuntos de
# parametros en paralelo, uno por proceso de Julia:
#
#   1. armonicos de G, solo el par (1,1)            -> kernel_elements_1-1.csv
#   2. χ̌(0), i∂χ̌(0) de los 100 pares + LLG         -> kernel_chi0_sites1-10.csv, llg_elements_*.csv
#   3. kernel en ω (ventanas γ̃ y Ω), 100 pares      -> kernel_chi_m{i}n{j}.csv
#   4. kernel en τ, 100 pares                        -> kernel_tau_m{i}n{j}.csv
#
# Todo queda en output/<tag>/data/ y el registro de cada conjunto en logs/<tag>.log.
# Las figuras se hacen despues en el computador local con plot_kernel_elements.py y
# plot_llg_elements.py (--tag <tag>), a partir de los CSV traidos con scp.
#
# Uso (desde fully_analytical_computations):
#   bash run_cluster.sh                 # 8 conjuntos en paralelo, 16 hilos cada uno
#   THREADS=12 SITES=10 bash run_cluster.sh
#   bash run_cluster.sh 1 6 7           # solo esos conjuntos
set -u
cd "$(dirname "$0")"

THREADS=${THREADS:-16}
SITES=${SITES:-10}
JULIA=${JULIA:-julia}
mkdir -p logs

# tag | γ_SO | γ = sqrt(1-γ_SO²) (γ̃ = 1) | J_sd | θ (grados) | Ω
SETS=(
  "setup2_gso0p1_jsd0p2_th10deg_Om0p005   0.1                0.99498743710662   0.2 10 0.005"
  "setup2_gso0p1_jsd0p2_th30deg_Om0p005   0.1                0.99498743710662   0.2 30 0.005"
  "setup2_gso0p5_jsd0p2_th10deg_Om0p005   0.5                0.8660254037844386 0.2 10 0.005"
  "setup2_gso0p1_jsd0p2_th10deg_Om0p02    0.1                0.99498743710662   0.2 10 0.02"
  "setup2_gso0p5_jsd0p2_th30deg_Om0p02    0.5                0.8660254037844386 0.2 30 0.02"
  "setup2_gso0p2588_jsd0p2_th10deg_Om0p005 0.2588190451025208 0.9659258262890683 0.2 10 0.005"
  "setup2_gso0p1951_jsd0p2_th10deg_Om0p005 0.1950903220161283 0.9807852804032304 0.2 10 0.005"
  "setup2_gso0p1_jsd0p5_th10deg_Om0p005   0.1                0.99498743710662   0.5 10 0.005"
)

run_set () {
  local tag=$1 gso=$2 gam=$3 jsd=$4 th=$5 om=$6
  local log="logs/${tag}.log"
  local P=(--params none --tag "$tag" --gso "$gso" --gamma "$gam" --jsd "$jsd" --theta "$th" --Omega "$om")
  (
    echo "=== $tag   (inicio $(date))"
    "$JULIA" -t "$THREADS" kernel_elements.jl "${P[@]}" --pairs "1,1"                   || exit 1
    "$JULIA" -t "$THREADS" kernel_elements.jl "${P[@]}" --chi0 --sites "$SITES"         || exit 1
    "$JULIA" llg_elements.jl "output/${tag}/data/kernel_chi0_sites1-${SITES}.csv"        || exit 1
    "$JULIA" -t "$THREADS" kernel_elements.jl "${P[@]}" --kernel --sites "$SITES"       || exit 1
    "$JULIA" -t "$THREADS" kernel_elements.jl "${P[@]}" --tau --sites "$SITES"          || exit 1
    echo "=== $tag   (fin $(date))"
  ) > "$log" 2>&1
  local rc=$?
  echo "  $tag: $( [ $rc -eq 0 ] && echo terminado || echo FALLO ) (ver $log)"
}

# conjuntos pedidos (por defecto los 8)
if [ $# -gt 0 ]; then idx=("$@"); else idx=(1 2 3 4 5 6 7 8); fi

echo "conjuntos: ${idx[*]}   hilos por conjunto: $THREADS   sitios: 1..$SITES"
for k in "${idx[@]}"; do
  read -r tag gso gam jsd th om <<< "${SETS[$((k-1))]}"
  echo "  lanzando $k: $tag"
  run_set "$tag" "$gso" "$gam" "$jsd" "$th" "$om" &
done
wait
echo "listo. Datos en output/<tag>/data/  (traer con scp -r ... output/<tag>/data)"
