#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=10

LOG=oscillators_tdnegf/output/r_scan.log
julia --project=. --threads=12 oscillators_tdnegf/oscillators.jl 2>&1 | tee "$LOG"

# Copia el log dentro de la subcarpeta etiquetada de esta corrida (la línea
# final "Listo en ... Salidas en <ruta>" la imprime oscillators.jl).
OUT_RUN=$(grep -oP '(?<=Salidas en ).*' "$LOG" | tail -n1)
if [ -n "$OUT_RUN" ] && [ -d "$OUT_RUN" ]; then
    cp "$LOG" "$OUT_RUN/r_scan.log"
    echo "Log copiado a $OUT_RUN/r_scan.log"
fi
