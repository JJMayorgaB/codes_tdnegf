#!/usr/bin/env julia
# Figuras de rashba_conductance_verification a partir de los CSV del cluster.
#   1) conductancia: línea negra (Landauer) + puntos rojos (TDNEGF estacionario)
#   2) corriente de carga vs tiempo
#   3) corrientes de espín vs tiempo

using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra, DelimitedFiles, Printf
using Plots, LaTeXStrings
pgfplotsx()

const σ_0 = ComplexF64[1 0; 0 1]
const σ_x = ComplexF64[0 1; 1 0]
const σ_y = ComplexF64[0 -im; im 0]

const Ny, Nσ = 3, 2
const γ, γso, γc = 1.0, 1.0 + 0.0im, 1.0
const M   = Ny*Nσ
const OUT = joinpath(@__DIR__, "output")

const E_F_traces = [0.0, 1.5]

# ── copia literal de TDNEGF/src/hamiltonians.jl : build_blocks ───────────────
function build_blocks(; Ny::Int, γ::Float64, γso::ComplexF64)
    I_y = Matrix{ComplexF64}(I, Ny, Ny)
    Ty  = diagm(-1 => ones(Ny-1))
    T0  = kron(Ty, -γ*σ_0 - 1im*γso*σ_x)
    H0  = T0 + T0'
    T   = kron(I_y, -γ*σ_0 + γso*1im*σ_y)
    return H0, T
end

const H_ii,    H_ip1_i = build_blocks(Ny=Ny, γ=γ, γso=γso)
const H0_lead, H1_lead = build_blocks(Ny=Ny, γ=γ, γso=0.0+0.0im)
const H_i_ip1 = adjoint(H_ip1_i)
const V_c     = ComplexF64.(-γc * Matrix(I, M, M))

function lead_G(E, eta, steps, H_1, H_0)
    Id = Matrix{ComplexF64}(I, M, M)
    A = ComplexF64.(H_1); B = adjoint(A)
    C = (E + eta*im)*Id - H_0
    D = (E + eta*im)*Id - H_0
    for _ in 1:steps
        Dinv = inv(D)
        A_new = A*Dinv*A;  B_new = B*Dinv*B
        C_new = C - A*Dinv*B
        D_new = D - A*Dinv*B - B*Dinv*A
        A, B, C, D = A_new, B_new, C_new, D_new
        norm(A) + norm(B) < 1e-12 && break
    end
    return inv(C)
end

function transmission_rec(E::Float64; eta::Float64 = 1e-6)
    g  = lead_G(E, eta, 100, H1_lead, H0_lead)
    Σ  = V_c * g * adjoint(V_c)
    Γ  = im*(Σ - adjoint(Σ))
    EI = (E + eta*im) * Matrix{ComplexF64}(I, M, M)
    G_11 = inv(EI - H_ii - Σ)
    G_22 = inv(EI - H_ii - H_ip1_i * G_11 * H_i_ip1)
    G_12 = G_11 * H_i_ip1 * G_22
    G_33 = inv(EI - H_ii - Σ - H_ip1_i * G_22 * H_i_ip1)
    G_13 = G_12 * H_i_ip1 * G_33
    return real(tr(Γ * G_13 * Γ * adjoint(G_13)))
end

const AXOPTS = Dict(:axis => Dict("tick style" => "{line width=1.5pt, color=black}"))

# ═══ Figura 1: conductancia ═══════════════════════════════════════════════════
dense_csv = joinpath(OUT, "rashba_conductance_dense.csv")
if isfile(dense_csv)
    d = readdlm(dense_csv, ',', skipstart = 1)
    E_dense, G_dense = Float64.(d[:,1]), Float64.(d[:,2])
else
    E_dense = collect(range(-4.0, 4.0; length = 600))
    G_dense = [transmission_rec(E)/2 for E in E_dense]
end

pts_new = joinpath(OUT, "rashba_conductance_points.csv")
pts_old = joinpath(OUT, "rashba_conductance_data.csv")
if isfile(pts_new)
    p = readdlm(pts_new, ',', skipstart = 1)
    E_F, G_rec, G_tdn = Float64.(p[:,1]), Float64.(p[:,2]), Float64.(p[:,3])
else
    p = readdlm(pts_old, ',', skipstart = 1)
    E_F, G_rec = Float64.(p[:,1]), Float64.(p[:,2])
    G_tdn = Float64.(p[:,3]) ./ 2
end

err = 100 .* abs.(G_tdn .- G_rec) ./ max.(G_rec, 1e-12)
@printf("error relativo:  medio %.3f %%   máx %.3f %% en E_F = %+.2f\n",
        sum(err)/length(err), maximum(err), E_F[argmax(err)])

plot(E_dense, G_dense;
     lc = :black, lw = 1.2, ls = :solid, label = L"\mathrm{Landauer}",
     xlabel = L"E_F/\gamma", ylabel = L"G\ [2e^2/h]",
     framestyle = :box, legend = :topright,
     size = (350, 240), dpi = 300,
     xlims = (-4, 4), ylims = (0, 2.0),
     xticks = -4:2:4, yticks = 0:0.5:2,
     grid = false,
     background_color_legend = :transparent,
     foreground_color_legend = :transparent,
     extra_kwargs = AXOPTS)

plot!(E_F, G_tdn; lw = 0, m = :circle, ms = 3, mc = :red, msc = :red,
      label = L"\mathrm{TDNEGF}")

savefig(joinpath(OUT, "rashba_conductance_verification.svg"))
savefig(joinpath(OUT, "rashba_conductance_verification.png"))

# ═══ Figuras 2 y 3: trazas temporales ════════════════════════════════════════
tracefile(E) = joinpath(OUT, "rashba_trace_EF$(replace(@sprintf("%+.2f", E), "." => "p")).csv")
available = [E for E in E_F_traces if isfile(tracefile(E))]

if isempty(available)
    println("\n(sin CSV de trazas — corre rashba_conductance_verification.jl actualizado)")
else
    traces = Dict(E => readdlm(tracefile(E), ',', skipstart = 1) for E in available)
    cols   = [:red, :blue]

    # ── Figura 2: corriente de carga ─────────────────────────────────────────
    plt2 = plot(; xlabel = L"t\ [\hbar/\gamma]", ylabel = L"I(t)\ [e\gamma/\hbar]",
                framestyle = :box, legend = :bottomright,
                size = (350, 240), dpi = 300, grid = false,
                background_color_legend = :transparent,
                foreground_color_legend = :transparent,
                extra_kwargs = AXOPTS)
    for (k, E) in enumerate(available)
        d = traces[E]
        plot!(plt2, Float64.(d[:,1]), Float64.(d[:,2]);
              lc = cols[k], lw = 1.0, ls = :solid,
              label = L"E_F = %$(round(E, digits=2))")
    end
    savefig(plt2, joinpath(OUT, "rashba_charge_current.svg"))
    savefig(plt2, joinpath(OUT, "rashba_charge_current.png"))

    # ── Figura 3: corrientes de espín (lead izquierdo, E_F_traces[1]) ────────
    E1 = available[1]
    d  = traces[E1]
    t  = Float64.(d[:,1])
    plt3 = plot(t, Float64.(d[:,4]);
                lc = :red, lw = 1.0, label = L"I^{s_x}",
                xlabel = L"t\ [\hbar/\gamma]",
                ylabel = L"I^{s_j}(t)\ [\gamma/2]",
                framestyle = :box, legend = :best,
                size = (350, 240), dpi = 300, grid = false,
                background_color_legend = :transparent,
                foreground_color_legend = :transparent,
                extra_kwargs = AXOPTS)
    plot!(plt3, t, Float64.(d[:,5]); lc = :black, lw = 1.0, label = L"I^{s_y}")
    plot!(plt3, t, Float64.(d[:,6]); lc = :blue,  lw = 1.0, label = L"I^{s_z}")

    savefig(plt3, joinpath(OUT, "rashba_spin_current.svg"))
    savefig(plt3, joinpath(OUT, "rashba_spin_current.png"))

    @printf("\ntrazas en E_F = %s   (figura de espín: E_F = %+.2f)\n",
            string(available), E1)
end

println("\nfiguras en $OUT")
