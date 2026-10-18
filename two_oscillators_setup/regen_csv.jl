#!/usr/bin/env julia
# Reescribe los CSV de una o varias corridas desde su fields.jld2 (sin correr la dinamica).
# Uso:  julia regen_csv.jl y_precession/output/DFF/F3-6/<tag> [otras corridas ...]
include(joinpath(@__DIR__, "common.jl"))
for d in ARGS
    regen_csvs(abspath(d))
end
