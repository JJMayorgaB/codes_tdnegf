#!/usr/bin/env julia
#
# Texturas de espín con Makie: vista cenital 2D, flechas con punta real.
# Complementa a plot_constriction.jl, que hace el resto de observables.
#
# Requiere:  julia --project=. -e 'using Pkg; Pkg.add("CairoMakie")'

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2, Printf, LaTeXStrings
using CairoMakie
using CairoMakie: Makie

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

const RUNS = ["FM_sym", "FM_asym", "AFM_sym", "AFM_asym"]
const SNAP_TIMES = [50.0, 105.0, 150.0, 200.0]

# umbral por debajo del cual s_⊥ es ruido de redondeo y no se dibuja
const ARROW_TOL = 1e-3
# longitud de flecha, en unidades de red, para el mayor s_⊥ del instante
const ARROW_LEN = 0.60

fields_path(r) = joinpath(OUT, "fields_$(r).jld2")

# ── índices de los instantes a mostrar ───────────────────────────────────────
function snap_indices(t)
    within = filter(τ -> t[1] <= τ <= t[end], SNAP_TIMES)
    targets = length(within) >= 2 ? within :
              [t[1] + f*(t[end] - t[1]) for f in (0.25, 0.5, 0.75, 1.0)]
    return [argmin(abs.(t .- τ)) for τ in targets]
end

# ── vector por sitio → matriz (Nx, Ny) con NaN en los huecos ────────────────
# El índice lineal es l = (x-1)*Ny + y. Makie espera mat[i,j] en (xs[i], ys[j]),
# o sea (Nx, Ny) — traspuesto respecto a lo que necesita Plots.jl.
function to_grid(v, keep)
    A = permutedims(reshape(collect(float.(v)), Ny, Nx))   # (Nx, Ny)
    for x in 1:Nx, y in 1:Ny
        keep[x, y] || (A[x, y] = NaN)
    end
    return A
end

# ── flechas, con respaldo si la API de Makie cambió de nombre ───────────────
# Makie renombró `arrows!` a `arrows2d!` en versiones recientes; los kwargs
# también cambiaron, así que si algo falla se dibuja a mano.
function draw_arrows!(ax, xs, ys, us, vs; color = :black)
    fn = isdefined(Makie, :arrows2d!) ? Makie.arrows2d! :
         isdefined(Makie, :arrows!)   ? Makie.arrows!   : nothing
    if fn !== nothing
        try
            fn(ax, xs, ys, us, vs; color = color)
            return
        catch
        end
    end
    for i in eachindex(xs)
        lines!(ax, [xs[i], xs[i] + us[i]], [ys[i], ys[i] + vs[i]];
               color = color, linewidth = 1.2)
    end
    scatter!(ax, xs .+ us, ys .+ vs; color = color, markersize = 4)
end

# ── figura por corrida ───────────────────────────────────────────────────────
function texture_figure(r)
    isfile(fields_path(r)) || (println("  (falta fields_$(r).jld2)"); return)
    d = load(fields_path(r))
    t, keep, s = d["t"], d["keep"], d["s_i"]
    idx = snap_indices(t)

    fig = Figure(size = (1020, 265))

    for (k, it) in enumerate(idx)
        # componente en el plano: magnitud máxima del instante
        smax = 0.0
        for x in 1:Nx, y in 1:Ny
            keep[x, y] || continue
            l = (x - 1)*Ny + y
            smax = max(smax, hypot(s[l, 1, it], s[l, 2, it]))
        end
        draw = smax > ARROW_TOL

        ttl = draw ?
            latexstring(@sprintf("t=%g,\\ \\times%.0f", t[it], ARROW_LEN/smax)) :
            latexstring(@sprintf("t=%g,\\ s_\\perp<10^{%d}", t[it],
                                 Int(ceil(log10(max(smax, 1e-300))))))

        ax = Axis(fig[1, k];
                  title = ttl, titlesize = 11,
                  xlabel = L"x", ylabel = k == 1 ? L"y" : "",
                  xlabelsize = 13, ylabelsize = 13,
                  aspect = DataAspect(),
                  xticks = 0:4:Nx-1, yticks = 0:2:Ny-1,
                  xgridvisible = false, ygridvisible = false)

        heatmap!(ax, 0:Nx-1, 0:Ny-1, to_grid(s[:, 3, it], keep);
                 colormap = :RdBu, colorrange = (-1, 1))

        if draw
            sc = ARROW_LEN/smax
            xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
            for x in 1:Nx, y in 1:Ny
                keep[x, y] || continue
                l = (x - 1)*Ny + y
                u, v = sc*s[l, 1, it], sc*s[l, 2, it]
                push!(xs, x - 1 - u/2); push!(ys, y - 1 - v/2)   # centrada
                push!(us, u);           push!(vs, v)
            end
            draw_arrows!(ax, xs, ys, us, vs; color = :black)
        end

        limits!(ax, -0.6, Nx - 0.4, -0.6, Ny - 0.4)
        k > 1 && hideydecorations!(ax; grid = false)
    end

    Colorbar(fig[1, length(idx) + 1]; colormap = :RdBu, limits = (-1, 1),
             label = L"s_z", labelsize = 14, width = 12)

    colgap!(fig.layout, 6)
    Label(fig[0, :], latexstring("\\vec{s}_i \\quad \\mathrm{" *
          replace(r, "_" => "\\_") * "}"); fontsize = 13)

    save(joinpath(OUT, "fig_texture_makie_$(r).png"), fig; px_per_unit = 3)
    save(joinpath(OUT, "fig_texture_makie_$(r).svg"), fig)
    println("  fig_texture_makie_$(r)")
end

# ═══════════════════════════════════════════════════════════════════════════════
function main()
    avail = filter(r -> isfile(fields_path(r)), RUNS)
    isempty(avail) && error("No hay fields_*.jld2 en $OUT")
    println("\nTexturas con Makie en $OUT\n")
    for r in avail
        texture_figure(r)
    end
    println("\nListo.")
end

main()
