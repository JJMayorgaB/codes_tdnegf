#!/usr/bin/env julia
# Reescribe los CSV de una o varias corridas desde su fields.jld2 (sin correr la dinamica).
# Uso:  julia regen_csv.jl z_precession/output/setup1_<tag> [otras corridas ...]
include(joinpath(@__DIR__, "common.jl"))
for d in ARGS
    regen_csvs(abspath(d))
end
