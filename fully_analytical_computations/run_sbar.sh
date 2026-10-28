#!/usr/bin/env bash
# run_sbar.sh -- densidad de espin s̄_m (marco rotante, t = 0) en los sitios 1..SITES
# del lead para los 8 conjuntos de parametros (los mismos de run_cluster.sh).
# Escribe output/<tag>/data/sbar_sites1-<SITES>.csv junto a llg_elements_sites1-10.csv.
# No recalcula el kernel: solo integra en ω los armonicos lesser de G (minutos en total).
#
# Uso (desde fully_analytical_computations; sirve en el computador local o en el cluster):
#   bash run_sbar.sh              # los 8 conjuntos, 4 a la vez, 4 hilos cada uno
#   bash run_sbar.sh 1 6 7        # solo esos conjuntos
#   THREADS=8 PAR=2 bash run_sbar.sh
set -u
cd "$(dirname "$0")"

THREADS=${THREADS:-4}
PAR=${PAR:-4}
SITES=${SITES:-10}
JULIA=${JULIA:-julia}
mkdir -p logs

# tag | γ_SO | γ = sqrt(1-γ_SO²) (γ̃ = 1) | J_sd | θ (grados) | Ω   (igual que run_cluster.sh)
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

if [ $# -gt 0 ]; then idx=("$@"); else idx=(1 2 3 4 5 6 7 8); fi

echo "s̄: conjuntos ${idx[*]}   $PAR a la vez, $THREADS hilos cada uno, sitios 1..$SITES"
for k in "${idx[@]}"; do
  read -r tag gso gam jsd th om <<< "${SETS[$((k-1))]}"
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  (
    "$JULIA" -t "$THREADS" kernel_elements.jl --params none --tag "$tag" --gso "$gso" --gamma "$gam" \
        --jsd "$jsd" --theta "$th" --Omega "$om" --sbar --sites "$SITES" > "logs/sbar_${tag}.log" 2>&1 \
      && echo "  $tag: listo -> output/$tag/data/sbar_sites1-$SITES.csv" \
      || echo "  $tag: FALLO (ver logs/sbar_${tag}.log)"
  ) &
done
wait
