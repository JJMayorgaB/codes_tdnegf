#!/bin/bash
set -e
cd "$(dirname "$0")/.."

RUN_TAG=${1:?uso: bash run_animations.sh <run_tag> [frame_skip] [t_min]}
SKIP=${2:-20}
TMIN=${3:-}

ARGS=()
[ -n "$TMIN" ] && ARGS+=(--t-min "$TMIN")

for r in 0.1 0.25 0.5 1.0 1.5 2.0; do
    for k in pos neg; do
        python3 oscillators_tdnegf/animate_oscillators.py \
            --run-tag "$RUN_TAG" --r "$r" --k "$k" --frame-skip "$SKIP" "${ARGS[@]}"
    done
done
