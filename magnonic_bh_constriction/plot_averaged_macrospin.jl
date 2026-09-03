#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Printf, FFTW
using Plots, LaTeXStrings

pgfplotsx()

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

const ZONES = [(0:4,  "input"),
               (5:9,  "constriction"),
               (10:14, "output")]

const T_START = 100.0
const Ω_MAX   = 0.5

const RUNS = [
    ("FM_sym",   L"\mathrm{FM}\ J_x{=}J_y"),
    ("FM_asym",  L"\mathrm{FM}\ J_x{\neq}J_y"),
    ("AFM_sym",  L"\mathrm{AFM}\ J_x{=}J_y"),
    ("AFM_asym", L"\mathrm{AFM}\ J_x{\neq}J_y"),
]

const ZCOL = [:black, :red, :blue]

const AX = Dict(:axis => Dict("tick style" => "{line width=1.2pt, color=black}"))
base(; kw...) = (framestyle = :box, grid = false, dpi = 300,
                 background_color_legend = :transparent,
                 foreground_color_legend = :transparent,
                 extra_kwargs = AX, kw...)

raw"""
Título que combina la etiqueta de la corrida con una anotación, en un solo
bloque matemático.

`lab` ya trae sus delimitadores de modo matemático, así que concatenar
directamente lo cerraría antes de tiempo y LaTeX vería el `\mathrm` fuera de él.
Hay que despojarlo de los delimitadores y volver a envolver el conjunto.

Nota: esta docstring es `raw` porque contiene barras invertidas; en una cadena
normal Julia las leería como secuencias de escape.
"""
titled(lab, note::AbstractString) =
    LaTeXString("\$" * strip(String(lab), '\$') * "\\quad(" * note * ")\$")

fields_path(r)  = joinpath(OUT, "fields_$(r).jld2")
rfields_path(r) = joinpath(OUT, "fields_relax_$(r).jld2")

site_index0(x0, y0) = x0*Ny + y0 + 1
is_afm(r) = startswith(r, "AFM")

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


function psd(t::AbstractVector, x::AbstractVector)
    n  = length(x)
    Δt = t[2] - t[1]
    X  = rfft(x)
    ω  = 2π .* rfftfreq(n, 1/Δt)
    S  = (Δt/n) .* abs2.(X)
    S[2:end] .*= 2
    return ω, S
end

"""
Promedio sobre los sitios de la zona de la PSD de cada sitio, componente por
componente. Devuelve (ω, [S̄_x, S̄_y], N_sitios).
"""
function zone_psd(t, s, keep, xs, sel)
    tw = t[sel]
    ω  = Float64[]
    Sx = Float64[];  Sy = Float64[]
    n  = 0
    for x0 in xs, y0 in 0:Ny-1
        keep[x0 + 1, y0 + 1] || continue
        l = site_index0(x0, y0)
        ωc, sx = psd(tw, s[l, 1, sel])
        _,  sy = psd(tw, s[l, 2, sel])
        if n == 0
            ω = ωc;  Sx = sx;  Sy = sy      # primer sitio: se toma tal cual
        else
            Sx .+= sx;  Sy .+= sy
        end
        n += 1
    end
    if n > 0
        Sx ./= n;  Sy ./= n
    end
    return ω, [Sx, Sy], n
end

function order_parameter(s, keep, xs, staggered::Bool)
    nt = size(s, 3)
    O  = zeros(3, nt);  n = 0
    for x0 in xs, y0 in 0:Ny-1
        keep[x0 + 1, y0 + 1] || continue
        n += 1
        l   = site_index0(x0, y0)
        sgn = staggered ? (-1)^(x0 + y0) : 1
        @inbounds for c in 1:3, it in 1:nt
            O[c, it] += sgn * s[l, c, it]
        end
    end
    n > 0 && (O ./= n)
    return O, n
end

function fig_run(r, lab)
    d = load_spins(r);  d === nothing && return
    t, s, keep = d
    sel = findall(≥(T_START), t)
    length(sel) < 8 && (println("  (ventana demasiado corta en $r)"); return)
    tw = t[sel]

    # PSD promediada sitio a sitio, una componente por panel.
    pps = [plot(; xlabel = c == 2 ? L"\omega\ (\gamma/\hbar)" : "",
                ylabel = LaTeXString("\$\\bar{S}_{$(cn)}(\\omega)\$"),
                title = c == 1 ? lab : "", titlefontsize = 8,
                xlims = (0, Ω_MAX),
                legend = c == 1 ? :best : false, legendfontsize = 5, base()...)
           for (c, cn) in enumerate(("x", "y"))]

    # PSD del parámetro de orden de la zona (suma coherente), en figura aparte.
    pks = Dict(tag => [plot(; xlabel = c == 2 ? L"\omega\ (\gamma/\hbar)" : "",
                ylabel = LaTeXString("\$S_{$(cn)}^{$(tag)}(\\omega)\$"),
                title = c == 1 ? titled(lab, tag) : "",
                titlefontsize = 8, xlims = (0, Ω_MAX),
                legend = c == 1 ? :best : false, legendfontsize = 5, base()...)
               for (c, cn) in enumerate(("x", "y"))]
               for tag in ("M", "N"))

    for (z, (xs, zname)) in enumerate(ZONES)
        ω, S̄, n = zone_psd(t, s, keep, xs, sel)
        n == 0 && continue
        m = ω .<= Ω_MAX
        for c in 1:2
            plot!(pps[c], ω[m], S̄[c][m]; lc = ZCOL[z], lw = 1.0,
                  label = "$(zname) (N=$(n))")
        end

        for (tag, stg) in (("M", false), ("N", true))
            O, _ = order_parameter(s, keep, xs, stg)
            for c in 1:2
                _, Scoh = psd(tw, O[c, sel])
                plot!(pks[tag][c], ω[m], Scoh[m];
                      lc = ZCOL[z], lw = 1.0, label = "$(zname) (N=$(n))")
            end
        end
    end

    pp = plot(pps...; layout = (2, 1), size = (520, 520), link = :x)
    savefig(pp, joinpath(OUT, "fig_avgpsd_$(r).png"))
    savefig(pp, joinpath(OUT, "fig_avgpsd_$(r).svg"))
    for tag in ("M", "N")
        pk = plot(pks[tag]...; layout = (2, 1), size = (520, 520), link = :x)
        savefig(pk, joinpath(OUT, "fig_totalpsd_$(tag)_$(r).png"))
        savefig(pk, joinpath(OUT, "fig_totalpsd_$(tag)_$(r).svg"))
    end
    println("  fig_avgpsd_$(r)   fig_totalpsd_{M,N}_$(r)")
end

function main()
    println("\nGenerando figuras en $OUT")
    @printf("Zonas: %s   ventana t ≥ %.0f   ω ≤ %.2f\n\n",
            join((string(first(z[1]), "–", last(z[1])) for z in ZONES), ", "),
            T_START, Ω_MAX)
    for (r, lab) in RUNS
        r in AVAIL || continue
        fig_run(r, lab)
    end
    println("\nListo.")
end

main()
