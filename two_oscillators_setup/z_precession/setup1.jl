#!/usr/bin/env julia
# setup1  [buf] F D F [buf]   -- driver precesando alrededor de z
# Uso:  OPENBLAS_NUM_THREADS=16 julia setup1.jl
include(joinpath(@__DIR__, "..", "common.jl"))
run_setup(:setup1, :z; outroot = joinpath(@__DIR__, "output"))
