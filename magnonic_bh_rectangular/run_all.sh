#!/usr/bin/env bash
# Corrida completa del setup de constricción rectangular (forma de H).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
SETUP="$(basename "$SCRIPT_DIR")"

THREADS="${THREADS:-4}"

LOG_DIR="$SCRIPT_DIR/output/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$PROJ_DIR"

echo "Setup:    $SETUP"
echo "Proyecto: $PROJ_DIR"
echo "Hilos:    $THREADS"

TOTAL0=$SECONDS

julia --project=. --threads="$THREADS" "$SETUP/relax_state.jl" \
    2>&1 | tee "$LOG_DIR/${STAMP}_relax.log"

julia --project=. --threads="$THREADS" "$SETUP/constriction_dynamics.jl" \
    2>&1 | tee "$LOG_DIR/${STAMP}_dynamics.log"

echo ""
printf "TODO LISTO en %d s (%.1f h)\n" "$((SECONDS - TOTAL0))" \
       "$(echo "scale=2; ($SECONDS - $TOTAL0)/3600" | bc -l)"
echo "Salidas: $SCRIPT_DIR/output"
