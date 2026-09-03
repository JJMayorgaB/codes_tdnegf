#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Printf, FFTW
using Plots, LaTeXStrings

pgfplotsx()

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

const XS = [1, 3, 5, 7, 9, 11, 13]
const YS = [3, 4]

# Inicio de la ventana: el bias se enciende en t=100.
const T_START = 100.0

const Ω_MAX = 0.5

const RUNS = [
    ("FM_sym",   L"\mathrm{FM}\ J_x{=}J_y"),
    ("FM_asym",  L"\mathrm{FM}\ J_x{\neq}J_y"),
    ("AFM_sym",  L"\mathrm{AFM}\ J_x{=}J_y"),
    ("AFM_asym", L"\mathrm{AFM}\ J_x{\neq}J_y"),
]

const COMPS = [(1, "sx", raw"s^x"), (2, "sy", raw"s^y")]

const AX = Dict(:axis => Dict("tick style" => "{line width=1.2pt, color=black}"))
base(; kw...) = (framestyle = :box, grid = false, dpi = 300,
                 background_color_legend = :transparent,
                 foreground_color_legend = :transparent,
                 extra_kwargs = AX, kw...)

fields_path(r)  = joinpath(OUT, "fields_$(r).jld2")
rfields_path(r) = joinpath(OUT, "fields_relax_$(r).jld2")

site_index0(x0, y0) = x0*Ny + y0 + 1

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


function spectrum(t::AbstractVector, x::AbstractVector)
    n  = length(x)
    Δt = t[2] - t[1]
    X  = rfft(x)
    ω  = 2π .* rfftfreq(n, 1/Δt)
    A  = abs.(X) ./ n
    A[2:end] .*= 2                  # espectro de un solo lado, salvo ω=0
    return ω, A
end

function fig_component(r, lab, comp, cname, csym)
    d = load_spins(r);  d === nothing && return
    t, s, keep = d

    sel = findall(≥(T_START), t)
    length(sel) < 8 && (println("  (ventana demasiado corta en $r)"); return)
    tw = t[sel]

    cols = cgrad(:viridis, length(XS); categorical = true)

    p = plot(; xlabel = L"\omega\ (\gamma/\hbar)",
             ylabel = LaTeXString("\$|" * csym * "(\\omega)|\$"),
             title = lab, titlefontsize = 8,
             xlims = (0, Ω_MAX),
             legend = :outerright, legendfontsize = 5,
             size = (560, 300), base()...)

    for (i, x0) in enumerate(XS), (j, y0) in enumerate(YS)
        keep[x0 + 1, y0 + 1] || continue
        l = site_index0(x0, y0)
        ω, A = spectrum(tw, s[l, comp, sel])
        m = ω .<= Ω_MAX
        ls = j == 1 ? :solid : :dash
        lb = j == 1 ? LaTeXString("\$x=$(x0)\$") : ""
        plot!(p, ω[m], A[m]; lc = cols[i], ls = ls, lw = 0.9, label = lb)
    end

    savefig(p, joinpath(OUT, "fig_fourier_$(cname)_$(r).png"))
    savefig(p, joinpath(OUT, "fig_fourier_$(cname)_$(r).svg"))
    println("  fig_fourier_$(cname)_$(r)")
end

function main()
    println("\nGenerando espectros en $OUT")
    @printf("Ventana: t ≥ %.0f    ω ≤ %.2f    (continua y=%d, discontinua y=%d)\n\n",
            T_START, Ω_MAX, YS[1], YS[2])
    for (r, lab) in RUNS
        r in AVAIL || continue
        for (comp, cname, csym) in COMPS
            fig_component(r, lab, comp, cname, csym)
        end
    end
    println("\nListo.")
end

main()
