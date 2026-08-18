#!/usr/bin/env julia
# Test de temperatura, solo Sunny. No usa TDNEGF.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Sunny
using Printf, Statistics, DelimitedFiles
using Plots

gr()   

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# Mismos parámetros y geometría que la constricción
const Nx, Ny = 15, 8
const a0     = 1.0
const Bx     = 1e-5
const Δt     = 0.1

const damping = 0.5       
const t_end   = 100.0
const N_seeds = 4            

const KT_LIST = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

const REMOVED_0BASED = vcat(
    [(x, 7) for x in 5:9], [(x, 6) for x in 6:8], [(7, 5)],
    [(x, 0) for x in 5:9], [(x, 1) for x in 6:8], [(7, 2)],
)

function build_mask()
    keep = trues(Nx, Ny)
    for (x0, y0) in REMOVED_0BASED
        keep[x0 + 1, y0 + 1] = false
    end
    return keep
end

const KEEP    = build_mask()
const REMOVED = [(x0 + 1, y0 + 1) for (x0, y0) in REMOVED_0BASED]
const N_ACT   = count(KEEP)

function build_system(J::Float64)
    latvecs = lattice_vectors(a0, a0*(1 + 1e-3), 4*a0, 90, 90, 90)
    cryst   = Crystal(latvecs, [[0.5, 0.5, 0.0]])
    moments = [1 => Moment(s = 1.0, g = 1.0)]

    sys = System(cryst, moments, :dipole; dims = (Nx, Ny, 1))
    set_field!(sys, [Bx, 0.0, 0.0])
    set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J, Bond(1, 1, [0, 1, 0]))

    sys = to_inhomogeneous(sys)
    remove_periodicity!(sys, (true, true, true))

    # aísla los sitios eliminados, igual que en la constricción
    for (x, y) in REMOVED
        for (dx, dy) in ((1,0), (-1,0), (0,1), (0,-1))
            xn, yn = x + dx, y + dy
            (1 <= xn <= Nx && 1 <= yn <= Ny) || continue
            set_exchange_at!(sys, 0.0, (x, y, 1, 1), (xn, yn, 1, 1);
                             offset = (dx, dy, 0))
        end
        set_field_at!(sys, [0.0, 0.0, 0.0], (x, y, 1, 1))
    end

    # textura inicial ferro: todos en +z
    for x in 1:Nx, y in 1:Ny
        sys.dipoles[x, y, 1, 1] = Sunny.SVector(0.0, 0.0, 1.0)
    end
    return sys
end

function magnetization(sys)
    m = zeros(3)
    for x in 1:Nx, y in 1:Ny
        KEEP[x, y] || continue
        v = sys.dipoles[x, y, 1, 1]
        m[1] += v[1]; m[2] += v[2]; m[3] += v[3]
    end
    return m ./ N_ACT
end


# Exchange nulo: el test aísla el efecto de la temperatura sobre el orden.
const J_EX = 0.0


function transverse_per_site(sys)
    s = 0.0
    for x in 1:Nx, y in 1:Ny
        KEEP[x, y] || continue
        v = sys.dipoles[x, y, 1, 1]
        s += sqrt(v[1]^2 + v[2]^2)
    end
    return s / N_ACT
end

function run_kT(kT, J)
    N_steps = Int(round(t_end / Δt))
    ts = collect(Δt:Δt:t_end)
    Mz = zeros(N_steps)
    Sp = zeros(N_steps)          # ⟨s_⊥⟩ por sitio

    for seed in 1:N_seeds
        sys = build_system(J)
        llg = Langevin(Δt; damping = damping, kT = kT)
        for i in 1:N_steps
            Sunny.step!(sys, llg)
            Mz[i] += magnetization(sys)[3]
            Sp[i] += transverse_per_site(sys)
        end
    end
    Mz ./= N_seeds
    Sp ./= N_seeds

    return ts, Mz, Sp, Mz[end], Sp[end]
end


function main()
    @printf("\nTest térmico (solo Sunny, sin exchange, sin anisotropía)\n")
    @printf("  sitios activos = %d,  α = %.2f,  t_end = %.0f,  realizaciones = %d\n",
            N_ACT, damping, t_end, N_seeds)
    @printf("\n  %-9s %-13s %-13s\n", "kT", "Mz(t=100)", "<s_perp>(t=100)")

    curves = []
    rows   = Vector{Vector{Float64}}()
    for kT in KT_LIST
        ts, Mz, Sp, mfin, sfin = run_kT(kT, J_EX)
        push!(curves, (kT, ts, Mz, Sp))
        push!(rows, [kT, mfin, sfin])
        @printf("  %-9.4f %-13.4f %-13.4f\n", kT, mfin, sfin)
        flush(stdout)
    end

    writedlm(joinpath(OUT, "spin_test_summary.csv"),
             vcat(["kT" "Mz_final" "s_perp_final"],
                  permutedims(hcat(rows...))), ",")

    ts  = curves[1][2]
    hdr = vcat("t", ["Mz_kT=$(c[1])" for c in curves],
                    ["sperp_kT=$(c[1])" for c in curves])
    writedlm(joinpath(OUT, "spin_test_traces.csv"),
             vcat(permutedims(hdr),
                  hcat(ts, hcat([c[3] for c in curves]...),
                           hcat([c[4] for c in curves]...))), ",")

    # Márgenes explícitos: con los de fábrica, gr() recorta las etiquetas de los
    # ejes al componer dos paneles.
    mar = (left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
           top_margin = 3Plots.mm, right_margin = 3Plots.mm)

    p1 = plot(; xlabel = "t", ylabel = "M_z", legend = :bottomleft,
              legendfontsize = 6, framestyle = :box, grid = false, dpi = 300,
              xlims = (0, t_end), ylims = (-0.05, 1.05), mar...)
    for (kT, ts, Mz, _) in curves
        plot!(p1, ts, Mz; lw = 1.2, label = "kT=$(kT)")
    end

    p2 = plot(; xlabel = "t", ylabel = "<s_perp> por sitio", legend = false,
              framestyle = :box, grid = false, dpi = 300,
              xlims = (0, t_end), ylims = (0, 1.0), mar...)
    for (kT, ts, _, Sp) in curves
        plot!(p2, ts, Sp; lw = 1.2, label = "kT=$(kT)")
    end

    p = plot(p1, p2; layout = (1, 2), size = (940, 360))
    savefig(p, joinpath(OUT, "fig_spin_test.png"))

    println("\n  → spin_test_summary.csv, spin_test_traces.csv, fig_spin_test.png")
end

main()
