#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Printf
using Plots, LaTeXStrings

pgfplotsx()

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

# Sitios a seguir, en coordenadas 0-based
const XS = [1, 3, 5, 7, 9, 11, 13]
const YS = [3, 4]

# Instante en que arranca la etapa de dinámica y sube el bias
const T_BIAS = 100.0

const RUNS = [
    ("FM_sym",   L"\mathrm{FM}\ J_x{=}J_y"),
    ("FM_asym",  L"\mathrm{FM}\ J_x{\neq}J_y"),
    ("AFM_sym",  L"\mathrm{AFM}\ J_x{=}J_y"),
    ("AFM_asym", L"\mathrm{AFM}\ J_x{\neq}J_y"),
]

const COMPS = [(1, "sx", L"s^x"), (2, "sy", L"s^y"), (3, "sz", L"s^z")]

const AX = Dict(:axis => Dict("tick style" => "{line width=1.2pt, color=black}"))
base(; kw...) = (framestyle = :box, grid = false, dpi = 300,
                 background_color_legend = :transparent,
                 foreground_color_legend = :transparent,
                 extra_kwargs = AX, kw...)

fields_path(r)  = joinpath(OUT, "fields_$(r).jld2")
rfields_path(r) = joinpath(OUT, "fields_relax_$(r).jld2")

# Índice lineal a partir de coordenadas 0-based: l = x*Ny + y + 1
site_index0(x0, y0) = x0*Ny + y0 + 1

"Relajación y dinámica concatenadas en el tiempo, como una sola corrida."
function load_spins(r)
    a = isfile(rfields_path(r)) ? load(rfields_path(r)) : nothing
    b = isfile(fields_path(r))  ? load(fields_path(r))  : nothing
    a === nothing && b === nothing && return nothing
    a === nothing && return (b["t"], b["s_i"], b["keep"])
    b === nothing && return (a["t"], a["s_i"], a["keep"])
    return (vcat(a["t"], b["t"]), cat(a["s_i"], b["s_i"]; dims = 3), b["keep"])
end

const AVAIL = [r for (r, _) in RUNS
               if isfile(fields_path(r)) || isfile(rfields_path(r))]
isempty(AVAIL) && error("No hay fields_*.jld2 en $OUT")
println("Corridas encontradas: ", join(AVAIL, ", "))

function fig_component(r, lab, comp, cname, csym)
    d = load_spins(r);  d === nothing && return
    t, s, keep = d

    cols = cgrad(:viridis, length(XS); categorical = true)

    p = plot(; xlabel = L"t\ (\hbar/\gamma)", ylabel = csym,
             title = lab, titlefontsize = 8,
             legend = :outerright, legendfontsize = 5,
             size = (560, 300), base()...)

    # marca del encendido del bias
    vline!(p, [T_BIAS]; lc = :gray, ls = :dot, lw = 0.8, label = "")

    for (i, x0) in enumerate(XS), (j, y0) in enumerate(YS)
        keep[x0 + 1, y0 + 1] || continue
        l  = site_index0(x0, y0)
        ls = j == 1 ? :solid : :dash
        # solo la curva de y=3 entra en la leyenda; la de y=4 comparte color
        lb = j == 1 ? LaTeXString("\$x=$(x0)\$") : ""
        plot!(p, t, s[l, comp, :]; lc = cols[i], ls = ls, lw = 0.9, label = lb)
    end

    savefig(p, joinpath(OUT, "fig_spins_$(cname)_$(r).png"))
    savefig(p, joinpath(OUT, "fig_spins_$(cname)_$(r).svg"))
    println("  fig_spins_$(cname)_$(r)")
end

function main()
    println("\nGenerando figuras en $OUT")
    println("Sitios: x = ", join(XS, ", "), "   y = ", join(YS, ", "),
            "   (continua y=$(YS[1]), discontinua y=$(YS[2]))\n")
    for (r, lab) in RUNS
        r in AVAIL || continue
        for (comp, cname, csym) in COMPS
            fig_component(r, lab, comp, cname, csym)
        end
    end
    println("\nListo.")
end

main()
