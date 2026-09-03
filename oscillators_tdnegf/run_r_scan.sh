#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=10
julia --project=. --threads=12 oscillators_tdnegf/oscillators.jl 2>&1 | tee oscillators_tdnegf/output/r_scan.log
