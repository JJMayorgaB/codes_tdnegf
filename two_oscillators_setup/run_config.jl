#!/usr/bin/env julia
# Una corrida con posiciones arbitrarias de los F (distancias con signo al driver).
# Salida en <eje>_precession/output/<disposicion>/<separacion>/<param_tag>/.
# Uso:
#   GSO=0.1 JSD=0.2 THETA=10 OMEGA=0.005 OPENBLAS_NUM_THREADS=8 julia run_config.jl y 3,6
#   julia run_config.jl y -4,4
include(joinpath(@__DIR__, "common.jl"))
length(ARGS) == 2 || error("uso: julia run_config.jl <eje: y|z> <FPOS, p. ej. 3,6 o -1,1>")
axis = Symbol(ARGS[1])
run_config(parse_fpos(ARGS[2]), axis; outroot = joinpath(@__DIR__, "$(axis)_precession", "output"))
