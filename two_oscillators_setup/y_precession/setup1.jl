#!/usr/bin/env julia
# setup1  [buf] F D F [buf]   -- driver precesando alrededor de y (eje del Rashba σ_y)
# Uso:  OPENBLAS_NUM_THREADS=16 julia setup1.jl
include(joinpath(@__DIR__, "..", "common.jl"))
run_setup(:setup1, :y; outroot = joinpath(@__DIR__, "output"))
