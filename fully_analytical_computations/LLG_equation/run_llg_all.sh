#!/usr/bin/env bash
# run_llg_all.sh -- llg_two_moments.jl para todos los conjuntos con sbar calculado
# (../output/setup2_*/data/sbar_sites1-10.csv) y las configuraciones DFF F1-2, F3-6, F5-9.
# Las corridas que ya tienen spins_t_llg.csv se saltan (llg_two_moments.jl lo revisa).
#
# Uso (desde LLG_equation):
#   bash run_llg_all.sh                        # todo, 4 a la vez
#   bash run_llg_all.sh setup2_gso0p1_jsd0p2_th10deg_Om0p005      # solo ese conjunto
#   PAR=8 EXTRA="--overwrite" bash run_llg_all.sh                 # recalcular todo
set -u
cd "$(dirname "$0")"

PAR=${PAR:-4}
JULIA=${JULIA:-julia}
EXTRA=${EXTRA:-}
SITES=("1,2" "3,6" "5,9")
mkdir -p logs

if [ $# -gt 0 ]; then tags=("$@"); else
  tags=(); for d in ../output/setup2_*/data/sbar_sites1-10.csv; do tags+=("$(basename "$(dirname "$(dirname "$d")")")"); done
fi

for t in "${tags[@]}"; do
  for s in "${SITES[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    lab="F${s/,/-}"
    ( "$JULIA" llg_two_moments.jl --tag "$t" --sites "$s" $EXTRA > "logs/${t}_${lab}.log" 2>&1 \
        && echo "  $t $lab: listo" || echo "  $t $lab: FALLO (ver logs/${t}_${lab}.log)" ) &
  done
done
wait
