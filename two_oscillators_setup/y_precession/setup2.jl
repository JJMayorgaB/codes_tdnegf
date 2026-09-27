#!/usr/bin/env julia
# setup2  [buf] D F F [buf]   -- driver precesando alrededor de y (eje del Rashba σ_y)
# Uso:  OPENBLAS_NUM_THREADS=16 julia setup2.jl
include(joinpath(@__DIR__, "..", "common.jl"))
run_setup(:setup2, :y; outroot = joinpath(@__DIR__, "output"))
