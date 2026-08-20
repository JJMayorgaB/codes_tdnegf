#!/usr/bin/env julia
#
# Espectro promediado sitio a sitio, por zona.
#
# Para cada sitio activo se calcula la densidad espectral de potencia de su
# componente transversal y luego se promedia sobre los sitios de la zona:
#
#     S̄_i(ω) = (1/N_i) Σ_{j∈i} [ S_j^x(ω) + S_j^y(ω) ]
#
# Es la suma INCOHERENTE: al promediar potencias se pierde la fase, así que
# contribuyen todos los modos, estén o no en fase entre sí. Por eso aquí no hace
# falta distinguir FM de AFM — el orden de Néel no cancela nada, porque el signo
# desaparece al tomar el módulo al cuadrado.
#
# Contrastar con plot_macrospin.jl, que hace la suma coherente. El cociente entre
# ambos, frecuencia a frecuencia, mide la coherencia espacial dentro de la zona:
#
#     coherencia(ω) = S_coherente(ω) / S̄_incoherente(ω)
#
# vale ~1 si todos los sitios oscilan en fase y tiende a 1/N_i si sus fases son
# aleatorias.
#
#   julia --project=. magnonic_bh_constriction/plot_averaged_macrospin.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Printf, FFTW
using Plots, LaTeXStrings

pgfplotsx()

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

const ZONES = [(0:4,  "entrada"),
               (5:9,  "constricción"),
               (10:14, "salida")]

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
"Densidad espectral de potencia de un solo lado, S(ω) = 2Δt|X_k|²/N."
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
    zone_psd(t, s, keep, xs, sel)

Promedio sobre los sitios de la zona de la PSD transversal de cada sitio.
Devuelve (ω, S̄, N_sitios).
"""
function zone_psd(t, s, keep, xs, sel)
    tw = t[sel]
    ω  = nothing
    S  = nothing
    n  = 0
    for x0 in xs, y0 in 0:Ny-1
        keep[x0 + 1, y0 + 1] || continue
        l = site_index0(x0, y0)
        ωx, Sx = psd(tw, s[l, 1, sel])
        _,  Sy = psd(tw, s[l, 2, sel])
        if S === nothing
            ω = ωx;  S = Sx .+ Sy
        else
            S .+= Sx .+ Sy
        end
        n += 1
    end
    n > 0 && (S ./= n)
    return ω, S, n
end

"Parámetro de orden de la zona, para el cociente de coherencia."
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

# ═══════════════════════════════════════════════════════════════════════════════
function fig_run(r, lab)
    d = load_spins(r);  d === nothing && return
    t, s, keep = d
    sel = findall(≥(T_START), t)
    length(sel) < 8 && (println("  (ventana demasiado corta en $r)"); return)
    tw   = t[sel]
    stag = is_afm(r)

    pp = plot(; xlabel = L"\omega\ (\gamma/\hbar)",
              ylabel = L"\bar{S}_\perp(\omega)",
              title = lab, titlefontsize = 8, xlims = (0, Ω_MAX),
              legend = :best, legendfontsize = 5, size = (520, 300), base()...)

    pc = plot(; xlabel = L"\omega\ (\gamma/\hbar)",
              ylabel = L"S_\perp^{\mathrm{coh}}/\bar{S}_\perp",
              title = lab, titlefontsize = 8, xlims = (0, Ω_MAX),
              legend = :best, legendfontsize = 5, size = (520, 300), base()...)

    for (z, (xs, zname)) in enumerate(ZONES)
        ω, S̄, n = zone_psd(t, s, keep, xs, sel)
        n == 0 && continue
        m = ω .<= Ω_MAX
        plot!(pp, ω[m], S̄[m]; lc = ZCOL[z], lw = 1.0, label = "$(zname) (N=$(n))")

        # cociente de coherencia contra la suma coherente de la misma zona
        O, _ = order_parameter(s, keep, xs, stag)
        _, Sx = psd(tw, O[1, sel])
        _, Sy = psd(tw, O[2, sel])
        Scoh = Sx .+ Sy
        ratio = Scoh ./ max.(S̄, eps())
        plot!(pc, ω[m], ratio[m]; lc = ZCOL[z], lw = 1.0, label = zname)
    end

    savefig(pp, joinpath(OUT, "fig_avgpsd_$(r).png"))
    savefig(pp, joinpath(OUT, "fig_avgpsd_$(r).svg"))
    savefig(pc, joinpath(OUT, "fig_coherence_$(r).png"))
    savefig(pc, joinpath(OUT, "fig_coherence_$(r).svg"))
    println("  fig_avgpsd_$(r)   fig_coherence_$(r)")
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
