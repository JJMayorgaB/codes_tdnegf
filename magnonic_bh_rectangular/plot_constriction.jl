#!/usr/bin/env julia
#
# Visualización de los observables de constriction_spin_dynamics.jl
#
#   1  fig_geometry               máscara de la constricción
#   2  fig_charge_current_<run>   I_L(t), I_R(t)                 — 1 fig/corrida
#   3  fig_spin_current_<run>     I^{s_x,y,z}(t)                 — 1 fig/corrida
#   4  fig_magnetization       M(t)                              — 1 panel/corrida
#   5  fig_neel                N(t)                              — 1 panel/corrida
#   6  fig_occupation_<run>    n_i(x,y) en 4 instantes
#   7  fig_dsigma_<run>        |σ - σ_eq|(x,y) en 4 instantes
#  7b  fig_torque_<run>        |s × σ|(x,y): torque sd completo
#   8  fig_texture_<run>       quiver (s_x,s_y) sobre s_z

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DelimitedFiles, JLD2, Printf, Statistics
using Plots, LaTeXStrings


const BACKEND = :pgf
BACKEND === :pgf ? pgfplotsx() : gr()

const OUT = joinpath(@__DIR__, "output")
const Nx, Ny = 15, 8

# corrida → (etiqueta, color, estilo)
const RUNS = [
    ("FM_sym",   L"\mathrm{FM}\ J_x{=}J_y",   :black, :solid),
    ("FM_asym",  L"\mathrm{FM}\ J_x{\neq}J_y", :red,   :solid),
    ("AFM_sym",  L"\mathrm{AFM}\ J_x{=}J_y",  :black, :dash),
    ("AFM_asym", L"\mathrm{AFM}\ J_x{\neq}J_y", :red,  :dash),
]

const AX = Dict(:axis => Dict("tick style" => "{line width=1.2pt, color=black}"))

base(; kw...) = (framestyle = :box, grid = false, dpi = 300,
                 background_color_legend = :transparent,
                 foreground_color_legend = :transparent,
                 extra_kwargs = AX, kw...)

# Carga de datos
trace_path(r)  = joinpath(OUT, "trace_$(r).csv")
fields_path(r) = joinpath(OUT, "fields_$(r).jld2")
rtrace_path(r)  = joinpath(OUT, "trace_relax_$(r).csv")
rfields_path(r) = joinpath(OUT, "fields_relax_$(r).jld2")

read_csv(p) = begin
    raw  = readdlm(p, ',')
    head = string.(raw[1, :])
    Dict(head[j] => Float64.(raw[2:end, j]) for j in eachindex(head))
end

# Relajación y dinámica se concatenan en el tiempo y se tratan como una sola
# corrida: las dos etapas escriben el mismo esquema, así que basta pegar.
function load_trace(r)
    a = isfile(rtrace_path(r)) ? read_csv(rtrace_path(r)) : nothing
    b = isfile(trace_path(r))  ? read_csv(trace_path(r))  : nothing
    a === nothing && return b
    b === nothing && return a
    Dict(k => vcat(a[k], b[k]) for k in keys(b) if haskey(a, k))
end

function load_fields(r)
    a = isfile(rfields_path(r)) ? load(rfields_path(r)) : nothing
    b = isfile(fields_path(r))  ? load(fields_path(r))  : nothing
    a === nothing && return b
    b === nothing && return a
    Dict("t"        => vcat(a["t"], b["t"]),
         "keep"     => b["keep"],
         "n_i"      => hcat(a["n_i"], b["n_i"]),                    # (Ns, nt)
         "sigma_i"  => cat(a["sigma_i"],  b["sigma_i"];  dims = 3), # (Ns, 3, nt)
         "sigma_eq" => cat(a["sigma_eq"], b["sigma_eq"]; dims = 3),
         "s_i"      => cat(a["s_i"],      b["s_i"];      dims = 3))
end

const AVAIL = [r for (r, _, _, _) in RUNS
               if isfile(trace_path(r)) || isfile(rtrace_path(r))]
isempty(AVAIL) && error("No hay CSV en $OUT — ¿ya corrió constriction_spin_dynamics.jl?")
println("Corridas encontradas: ", join(AVAIL, ", "))

# Utilidades espaciales
# El índice lineal es l = (x-1)*Ny + y  ⇒  reshape(v, Ny, Nx)[y, x]
"Vector de N_sites → matriz (Ny, Nx), con NaN en los sitios eliminados."
function to_grid(v::AbstractVector, keep)
    A = Matrix{Float64}(reshape(collect(v), Ny, Nx))
    for x in 1:Nx, y in 1:Ny
        keep[x, y] || (A[y, x] = NaN)
    end
    return A
end

"Índices de tiempo más cercanos a los valores pedidos."
nearest(t, targets) = [argmin(abs.(t .- τ)) for τ in targets]


# Instantáneas: N_SNAPS tiempos igualmente espaciados sobre TODO el rango
# disponible, relajación y dinámica juntas. Sin lista fija de tiempos, así que
# se adapta solo a la duración que tengan los datos.
const N_SNAPS = 8

snap_indices(t) = nearest(t, range(t[1], t[end]; length = N_SNAPS))

# Ocho paneles en fila serían ilegibles: se reparten en dos filas de cuatro.
# Con menos instantáneas se ajusta solo, para que el script siga sirviendo si
# alguna vez se baja N_SNAPS.
grid_shape(n) = n <= 4 ? (1, n) : (cld(n, 4), 4)
is_left(k, ncol)        = (k - 1) % ncol == 0
is_bottom(k, n, ncol)   = k > n - ncol

# 1 · Geometría
function fig_geometry()
    f = load_fields(AVAIL[1]);  f === nothing && return
    keep = f["keep"]
    A = [keep[x, y] ? 1.0 : NaN for y in 1:Ny, x in 1:Nx]

    p = heatmap(0:Nx-1, 0:Ny-1, A;
        c = cgrad([:steelblue, :steelblue]), clims = (0, 1), colorbar = false,
        xlabel = L"x", ylabel = L"y", aspect_ratio = :equal,
        xticks = 0:2:Nx-1, yticks = 0:1:Ny-1,
        xlims = (-0.5, Nx-0.5), ylims = (-0.5, Ny-0.5),
        size = (420, 250), base()...)

    savefig(p, joinpath(OUT, "fig_geometry.png"))
    savefig(p, joinpath(OUT, "fig_geometry.svg"))
    println("  fig_geometry")
end

# 2 · Corriente de carga — una figura por configuración, con I_L e I_R
function fig_charge_current(r, lab)
    d = load_trace(r);  d === nothing && return
    p = plot(; xlabel = L"t\ (\hbar/\gamma)", ylabel = L"I(t)\ (e\gamma/\hbar)",
             title = lab, titlefontsize = 8,
             legend = :topright, legendfontsize = 7, size = (420, 280), base()...)
    plot!(p, d["t"], d["I_L"]; lc = :black, ls = :solid, lw = 1.0, label = L"I_L")
    plot!(p, d["t"], d["I_R"]; lc = :red,   ls = :dash,  lw = 1.0, label = L"I_R")
    savefig(p, joinpath(OUT, "fig_charge_current_$(r).png"))
    savefig(p, joinpath(OUT, "fig_charge_current_$(r).svg"))
    println("  fig_charge_current_$(r)")
end


# 3 · Corrientes de espín — una figura por configuración, tres componentes
function fig_spin_current(r, lab)
    d = load_trace(r);  d === nothing && return
    comps = [("Isx_L", L"I^{s_x}"), ("Isy_L", L"I^{s_y}"), ("Isz_L", L"I^{s_z}")]

    panels = []
    for (k, (col, sym)) in enumerate(comps)
        p = plot(; ylabel = sym,
                 xlabel = k == 3 ? L"t\ (\hbar/\gamma)" : "",
                 title = k == 1 ? lab : "", titlefontsize = 8,
                 legend = false, base()...)
        plot!(p, d["t"], d[col]; lc = :black, lw = 1.0, label = "")
        push!(panels, p)
    end
    p = plot(panels...; layout = (3, 1), size = (420, 620), link = :x)
    savefig(p, joinpath(OUT, "fig_spin_current_$(r).png"))
    savefig(p, joinpath(OUT, "fig_spin_current_$(r).svg"))
    println("  fig_spin_current_$(r)   (electrodo izquierdo)")
end

# 4-5 · Parámetros de orden
function fig_order(prefix::String, sym, fname::String; tol::Float64 = 1e-8)
    amps = Float64[]
    for (r, _, _, _) in RUNS
        d = load_trace(r);  d === nothing && continue
        push!(amps, maximum(maximum(abs, d["$(prefix)_$(c)"]) for c in ("x","y","z")))
    end
    isempty(amps) && return
    leg_panel = something(findfirst(≥(tol), amps), 1)

    panels = []
    for (k, (r, lab, _, _)) in enumerate(RUNS)
        d = load_trace(r);  d === nothing && continue

        comps = [d["$(prefix)_$(c)"] for c in ("x", "y", "z")]
        amp   = maximum(maximum(abs, v) for v in comps)
        flat  = amp < tol

        p = plot(; title = lab, titlefontsize = 7,
                 ylabel = k in (1, 3) ? sym : "",
                 xlabel = k in (3, 4) ? L"t\ (\hbar/\gamma)" : "",
                 legend = k == leg_panel ? :best : false, legendfontsize = 5,
                 ylims = flat ? (-1.05, 1.05) : :auto,
                 yticks = flat ? (-1:0.5:1) : :auto,
                 base()...)

        for (v, comp, col, ls) in zip(comps, ("x", "y", "z"),
                                      (:red, :green, :blue),
                                      (:solid, :dash, :dot))
            plot!(p, d["t"], flat ? zero(v) : v;
                  lc = col, ls = ls, lw = 1.0, label = LaTeXString("\$$(comp)\$"))
        end

        if flat
            annotate!(p, (d["t"][1] + d["t"][end])/2, 0.45,
                      text(LaTeXString(@sprintf("\$\\equiv 0\\ (<10^{%d})\$",
                                                Int(floor(log10(max(amp, 1e-300)))))),
                           6, :gray))
        end
        push!(panels, p)
    end
    isempty(panels) && return
    p = plot(panels...; layout = (2, 2), size = (620, 420))
    savefig(p, joinpath(OUT, "$(fname).png"))
    savefig(p, joinpath(OUT, "$(fname).svg"))
    println("  $fname")
end

# 6-7 · Mapas espaciales
"Fila de mapas de calor en 4 instantes."
function spatial_row(r, field_fun, label, fname, cmap; symmetric = false)
    f = load_fields(r);  f === nothing && return
    t, keep = f["t"], f["keep"]
    idx = snap_indices(t)

    grids = [to_grid(field_fun(f, it), keep) for it in idx]
    finite = filter(isfinite, vcat(vec.(grids)...))
    isempty(finite) && return
    lims = symmetric ? (-maximum(abs, finite), maximum(abs, finite)) :
                       (minimum(finite), maximum(finite))

    n = length(idx)
    nrow, ncol = grid_shape(n)
    panels = []
    for (k, it) in enumerate(idx)
        p = heatmap(0:Nx-1, 0:Ny-1, grids[k];
            c = cmap, clims = lims, colorbar = (k == n),
            colorbar_title = k == n ? LaTeXString("\$" * label * "\$") : "",
            title = LaTeXString(@sprintf("\$t=%g\$", t[it])), titlefontsize = 7,
            xlabel = is_bottom(k, n, ncol) ? L"x" : "",
            ylabel = is_left(k, ncol) ? L"y" : "",
            aspect_ratio = :equal, xticks = 0:4:Nx-1,
            yticks = is_left(k, ncol) ? (0:2:Ny-1) :
                     (0:2:Ny-1, fill("", length(0:2:Ny-1))),
            xlims = (-0.5, Nx-0.5), ylims = (-0.5, Ny-0.5),
            left_margin  = (is_left(k, ncol) ? 1 : -3)Plots.mm,
            right_margin = (k % ncol == 0 ? 1 : -3)Plots.mm,
            base()...)
        push!(panels, p)
    end

    p = plot(panels...; layout = (nrow, ncol),
             size = (760, 185*nrow), link = :y)
    savefig(p, joinpath(OUT, "$(fname)_$(r).png"))
    savefig(p, joinpath(OUT, "$(fname)_$(r).svg"))
    println("  $(fname)_$(r)")
end

occupation(f, it) = f["n_i"][:, it]

function dsigma(f, it)
    σ, σeq = f["sigma_i"], f["sigma_eq"]
    [sqrt(sum((σ[l, c, it] - σeq[l, c, it])^2 for c in 1:3)) for l in axes(σ, 1)]
end


function torque(f, it)
    σ, s = f["sigma_i"], f["s_i"]
    map(axes(σ, 1)) do l
        σ1, σ2, σ3 = σ[l,1,it], σ[l,2,it], σ[l,3,it]
        s1, s2, s3 = s[l,1,it], s[l,2,it], s[l,3,it]
        sqrt((s2*σ3 - s3*σ2)^2 + (s3*σ1 - s1*σ3)^2 + (s1*σ2 - s2*σ1)^2)
    end
end

# 8 · Textura clásica (quiver sobre s_z)
function fig_texture(r; tol::Float64 = 1e-3)
    f = load_fields(r);  f === nothing && return
    t, keep, s = f["t"], f["keep"], f["s_i"]
    idx = snap_indices(t)

    n = length(idx)
    nrow, ncol = grid_shape(n)
    panels = []
    for (k, it) in enumerate(idx)
        sz = to_grid(s[:, 3, it], keep)

        sperp(l) = sqrt(s[l, 1, it]^2 + s[l, 2, it]^2)
        smax = maximum(sperp(l) for x in 1:Nx, y in 1:Ny
                       if keep[x, y] for l in ((x-1)*Ny + y,))
        draw = smax > tol

        ttl = draw ?
            @sprintf("\$t=%g,\\ \\times%.0f\$", t[it], 0.45/smax) :
            @sprintf("\$t=%g,\\ s_\\perp<10^{%d}\$", t[it],
                     Int(ceil(log10(max(smax, 1e-300)))))

        p = heatmap(0:Nx-1, 0:Ny-1, sz;
            c = :RdBu, clims = (-1, 1), colorbar = (k == n),
            colorbar_title = k == n ? L"s_z" : "",
            title = LaTeXString(ttl), titlefontsize = 6,
            xlabel = is_bottom(k, n, ncol) ? L"x" : "",
            ylabel = is_left(k, ncol) ? L"y" : "",
            aspect_ratio = :equal, xticks = 0:4:Nx-1,
            yticks = is_left(k, ncol) ? (0:2:Ny-1) :
                     (0:2:Ny-1, fill("", length(0:2:Ny-1))),
            xlims = (-0.5, Nx-0.5), ylims = (-0.5, Ny-0.5),
            left_margin  = (is_left(k, ncol) ? 1 : -3)Plots.mm,
            right_margin = (k % ncol == 0 ? 1 : -3)Plots.mm,
            base()...)

        if draw
            scale = 0.45/smax
            xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
            hx = Float64[]; hy = Float64[]
            for x in 1:Nx, y in 1:Ny
                keep[x, y] || continue
                l = (x - 1)*Ny + y
                u, v = scale*s[l, 1, it], scale*s[l, 2, it]
                # se dibuja centrada en el sitio: de -u/2 a +u/2
                push!(xs, x - 1 - u/2); push!(ys, y - 1 - v/2)
                push!(us, u);           push!(vs, v)
                push!(hx, x - 1 + u/2); push!(hy, y - 1 + v/2)
            end
            quiver!(p, xs, ys; quiver = (us, vs), lc = :black, lw = 0.7)
            # punta explícita: pgfplotsx no dibuja cabezas de flecha en quiver
            scatter!(p, hx, hy; m = :circle, ms = 1.1, mc = :black,
                     msc = :black, label = "")
        end
        push!(panels, p)
    end
    p = plot(panels...; layout = (nrow, ncol),
             size = (760, 185*nrow), link = :y)
    savefig(p, joinpath(OUT, "fig_texture_$(r).png"))
    savefig(p, joinpath(OUT, "fig_texture_$(r).svg"))
    println("  fig_texture_$(r)")
end


function main()
    println("\nGenerando figuras en $OUT\n")

    fig_geometry()
    fig_order("M",    L"M_j",    "fig_magnetization")
    fig_order("Neel", L"N_j",    "fig_neel")

    for (r, lab, _, _) in RUNS
        r in AVAIL || continue
        fig_charge_current(r, lab)
        fig_spin_current(r, lab)
    end

    for r in AVAIL
        spatial_row(r, occupation, raw"n_i", "fig_occupation", :viridis)
        spatial_row(r, dsigma, raw"|\vec{\sigma}_i - \vec{\sigma}_i^{\,\mathrm{eq}}|",
                    "fig_dsigma", :magma)
        spatial_row(r, torque,
                    raw"|\vec{s}_i\times\vec{\sigma}_i|",
                    "fig_torque", :inferno)
        fig_texture(r)
    end

    println("\nListo.")
end

main()
