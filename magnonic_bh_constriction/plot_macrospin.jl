#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Printf, FFTW
using Plots, LaTeXStrings

pgfplotsx()

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

# Zonas en x, coordenadas 0-based
const ZONES = [(0:4,  "input"),
               (5:9,  "constriction"),
               (10:14, "output")]

# Ventana de análisis: solo la etapa con el bias encendido.
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

# pgfplotsx escribe cada punto como una coordenada en el .tex, así que una serie
# de 6000 pasos por varias curvas desborda la memoria de tokens de LuaTeX. Las
# figuras miden ~520 px de ancho, de modo que por encima de MAX_PTS puntos por
# curva no se gana resolución visible. Solo afecta al dibujo: los espectros se
# calculan siempre con la serie completa.
const MAX_PTS = 2000
thin(n::Int) = max(1, cld(n, MAX_PTS))

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

function order_parameter(s, keep, xs, staggered::Bool)
    nt = size(s, 3)
    O  = zeros(3, nt)
    n  = 0
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
Genera las figuras de una corrida para un parámetro de orden dado.

`stag = false` da la magnetización, `true` el vector de Néel. Se calculan los dos
en todas las corridas, no uno según la fase.

La razón: la regla "FM → magnetización, AFM → Néel" describe el orden estático,
pero no la dinámica. En un antiferromagneto el magnón es una precesión del vector
de Néel que induce una magnetización neta transversal; ambas son variables
conjugadas y la señal magnónica vive en las dos. Empíricamente, en estas corridas
el canal limpio del AFM resulta ser la magnetización transversal, no el Néel.
"""
function fig_order(r, lab, stag::Bool)
    d = load_spins(r);  d === nothing && return
    t, s, keep = d
    sel = findall(≥(T_START), t)
    length(sel) < 8 && (println("  (ventana demasiado corta en $r)"); return)
    tw  = t[sel]
    sym = stag ? "N" : "M"
    tag = stag ? "N" : "M"

    # El parámetro de orden de cada zona se calcula una sola vez y se reutiliza
    # en los tres paneles y en los espectros.
    zdata = [(zname, order_parameter(s, keep, xs, stag)...)
             for (xs, zname) in ZONES]

    panels = []
    for (c, cname) in enumerate(("x", "y", "z"))
        # Eje y centrado en cero, con margen de 0.1 sobre el máximo absoluto
        # entre las tres zonas, para que los paneles sean comparables entre sí.
        amax = 0.0
        for (_, O, n) in zdata
            n == 0 && continue
            amax = max(amax, maximum(abs, view(O, c, sel)))
        end
        lim = amax + 0.1

        p = plot(; ylabel = LaTeXString("\$$(sym)^$(cname)\$"),
                 xlabel = c == 3 ? L"t\ (\hbar/\gamma)" : "",
                 title = c == 1 ? titled(lab, tag) : "",
                 titlefontsize = 8,
                 ylims = (-lim, lim),
                 legend = c == 1 ? :best : false, legendfontsize = 5,
                 base()...)
        st = thin(length(sel))
        for (z, (zname, O, n)) in enumerate(zdata)
            n == 0 && continue
            plot!(p, tw[1:st:end], O[c, sel][1:st:end]; lc = ZCOL[z], lw = 0.9,
                  label = "$(zname) (N=$(n))")
        end
        push!(panels, p)
    end
    p = plot(panels...; layout = (3, 1), size = (520, 620), link = :x)
    savefig(p, joinpath(OUT, "fig_macrospin_$(tag)_t_$(r).png"))
    savefig(p, joinpath(OUT, "fig_macrospin_$(tag)_t_$(r).svg"))
    println("  fig_macrospin_$(tag)_t_$(r)")

    # Espectros de cada componente por separado, no de la combinación
    # transversal. Manteniéndolas separadas se puede ver si la precesión es
    # circular —misma amplitud en x e y— o elíptica, información que se pierde
    # al sumar en cuadratura.
    pas = Any[];  pps = Any[]
    for (c, cname) in enumerate(("x", "y"))
        push!(pas, plot(; xlabel = c == 2 ? L"\omega\ (\gamma/\hbar)" : "",
              ylabel = LaTeXString("\$|\\tilde{$(sym)}^$(cname)(\\omega)|\$"),
              title = c == 1 ? titled(lab, tag) : "",
              titlefontsize = 8, xlims = (0, Ω_MAX),
              legend = c == 1 ? :best : false, legendfontsize = 5, base()...))
        push!(pps, plot(; xlabel = c == 2 ? L"\omega\ (\gamma/\hbar)" : "",
              ylabel = LaTeXString("\$S_{$(cname)}^{$(sym)}(\\omega)\$"),
              title = c == 1 ? titled(lab, tag) : "",
              titlefontsize = 8, xlims = (0, Ω_MAX),
              legend = c == 1 ? :best : false, legendfontsize = 5, base()...))
    end

    nT = length(sel);  Δt = tw[2] - tw[1]
    for (z, (zname, O, n)) in enumerate(zdata)
        n == 0 && continue
        for c in 1:2
            ω, S = psd(tw, O[c, sel])
            m = ω .<= Ω_MAX
            A = sqrt.(S .* (2/(nT*Δt)))
            plot!(pas[c], ω[m], A[m]; lc = ZCOL[z], lw = 1.0, label = zname)
            plot!(pps[c], ω[m], S[m]; lc = ZCOL[z], lw = 1.0, label = zname)
        end
    end

    pa = plot(pas...; layout = (2, 1), size = (520, 520), link = :x)
    pp = plot(pps...; layout = (2, 1), size = (520, 520), link = :x)

    savefig(pa, joinpath(OUT, "fig_macrospin_$(tag)_fft_$(r).png"))
    savefig(pa, joinpath(OUT, "fig_macrospin_$(tag)_fft_$(r).svg"))
    savefig(pp, joinpath(OUT, "fig_macrospin_$(tag)_psd_$(r).png"))
    savefig(pp, joinpath(OUT, "fig_macrospin_$(tag)_psd_$(r).svg"))
    println("  fig_macrospin_$(tag)_fft_$(r)   fig_macrospin_$(tag)_psd_$(r)")
end

function main()
    println("\nGenerando figuras en $OUT")
    @printf("Zonas: %s   ventana t ≥ %.0f   ω ≤ %.2f\n\n",
            join((string(first(z[1]), "–", last(z[1])) for z in ZONES), ", "),
            T_START, Ω_MAX)
    for (r, lab) in RUNS
        r in AVAIL || continue
        for stag in (false, true)          # magnetización y vector de Néel
            fig_order(r, lab, stag)
        end
    end
    println("\nListo.")
end

main()
