#!/usr/bin/env julia
#
# Parámetro de orden por zona: serie temporal y espectro.
#
# El dispositivo se divide en tres zonas de 5 columnas: entrada (x = 0…4),
# constricción (x = 5…9) y salida (x = 10…14). En cada zona se construye el
# parámetro de orden que corresponde al sistema,
#
#     FM :   M_i(t) = (1/N_i) Σ_{j∈i} s_j(t)
#     AFM:   N_i(t) = (1/N_i) Σ_{j∈i} (-1)^{x_j+y_j} s_j(t)
#
# normalizado por el número de sitios activos N_i de la zona, que difiere entre
# zonas y entre geometrías. Sin esa normalización la zona del cuello saldría
# sistemáticamente más pequeña por razones geométricas, no físicas.
#
# Esta es la suma COHERENTE: solo sobreviven las componentes que oscilan en fase
# a lo largo de la zona. Compárese con plot_averaged_macrospin.jl, que hace la
# suma incoherente.
#
#   julia --project=. magnonic_bh_constriction/plot_macrospin.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Printf, FFTW
using Plots, LaTeXStrings

pgfplotsx()

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

# Zonas en x, coordenadas 0-based
const ZONES = [(0:4,  "entrada"),
               (5:9,  "constricción"),
               (10:14, "salida")]

# Ventana de análisis: solo la etapa con el bias encendido.
const T_START = 100.0
const Ω_MAX   = 1.0

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

# ═══════════════════════════════════════════════════════════════════════════════
"""
    order_parameter(s, keep, xs, staggered)

Parámetro de orden de la zona `xs`, como matriz (3, nt), normalizado por el
número de sitios activos. `staggered = true` da el vector de Néel; `false`, la
magnetización. Devuelve también el conteo de sitios.
"""
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

"""
    psd(t, x)

Densidad espectral de potencia de un solo lado,  S(ω) = 2Δt|X_k|²/N,  que es la
versión discreta de  S(ω) = |x̃(ω)|²/T  con  x̃(ω) = ∫₀ᵀ x(t)e^{-iωt}dt.

El 1/T es lo que hace que la cantidad converja al alargar la ventana en vez de
crecer con ella, y es también lo que permite comparar corridas de duración
distinta —las de t=400 con las de t=700—. El área bajo S/2π es la potencia media
de la señal.

La señal entra cruda: no se resta la media ni se aplica ventana.
"""
function psd(t::AbstractVector, x::AbstractVector)
    n  = length(x)
    Δt = t[2] - t[1]
    X  = rfft(x)
    ω  = 2π .* rfftfreq(n, 1/Δt)
    S  = (Δt/n) .* abs2.(X)
    S[2:end] .*= 2                  # un solo lado, salvo ω=0
    return ω, S
end

"PSD de la parte transversal: S_x + S_y."
function psd_perp(t, O)
    ω, Sx = psd(t, O[1, :])
    _, Sy = psd(t, O[2, :])
    return ω, Sx .+ Sy
end

# ═══════════════════════════════════════════════════════════════════════════════
function fig_run(r, lab)
    d = load_spins(r);  d === nothing && return
    t, s, keep = d
    sel = findall(≥(T_START), t)
    length(sel) < 8 && (println("  (ventana demasiado corta en $r)"); return)
    tw  = t[sel]
    stag = is_afm(r)
    sym  = stag ? "N" : "M"

    # ── serie temporal: una fila por componente, las tres zonas superpuestas ──
    panels = []
    for (c, cname) in enumerate(("x", "y", "z"))
        p = plot(; ylabel = LaTeXString("\$$(sym)^$(cname)\$"),
                 xlabel = c == 3 ? L"t\ (\hbar/\gamma)" : "",
                 title = c == 1 ? lab : "", titlefontsize = 8,
                 legend = c == 1 ? :best : false, legendfontsize = 5,
                 base()...)
        for (z, (xs, zname)) in enumerate(ZONES)
            O, n = order_parameter(s, keep, xs, stag)
            n == 0 && continue
            plot!(p, tw, O[c, sel]; lc = ZCOL[z], lw = 0.9,
                  label = "$(zname) (N=$(n))")
        end
        push!(panels, p)
    end
    p = plot(panels...; layout = (3, 1), size = (520, 620), link = :x)
    savefig(p, joinpath(OUT, "fig_macrospin_t_$(r).png"))
    savefig(p, joinpath(OUT, "fig_macrospin_t_$(r).svg"))
    println("  fig_macrospin_t_$(r)")

    # ── espectro: amplitud y potencia ────────────────────────────────────────
    pa = plot(; xlabel = L"\omega\ (\gamma/\hbar)",
              ylabel = LaTeXString("\$|\\tilde{$(sym)}_\\perp(\\omega)|\$"),
              title = lab, titlefontsize = 8, xlims = (0, Ω_MAX),
              legend = :best, legendfontsize = 5, size = (520, 300), base()...)
    pp = plot(; xlabel = L"\omega\ (\gamma/\hbar)",
              ylabel = LaTeXString("\$S_\\perp^{$(sym)}(\\omega)\$"),
              title = lab, titlefontsize = 8, xlims = (0, Ω_MAX),
              legend = :best, legendfontsize = 5, size = (520, 300), base()...)

    for (z, (xs, zname)) in enumerate(ZONES)
        O, n = order_parameter(s, keep, xs, stag)
        n == 0 && continue
        ω, S = psd_perp(tw, O[:, sel])
        m = ω .<= Ω_MAX
        # amplitud del transverso, en las mismas unidades que la señal
        nT = length(sel);  Δt = tw[2] - tw[1]
        A  = sqrt.(S .* (2/(nT*Δt)))
        plot!(pa, ω[m], A[m]; lc = ZCOL[z], lw = 1.0, label = zname)
        plot!(pp, ω[m], S[m]; lc = ZCOL[z], lw = 1.0, label = zname)
    end

    savefig(pa, joinpath(OUT, "fig_macrospin_fft_$(r).png"))
    savefig(pa, joinpath(OUT, "fig_macrospin_fft_$(r).svg"))
    savefig(pp, joinpath(OUT, "fig_macrospin_psd_$(r).png"))
    savefig(pp, joinpath(OUT, "fig_macrospin_psd_$(r).svg"))
    println("  fig_macrospin_fft_$(r)   fig_macrospin_psd_$(r)")
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
