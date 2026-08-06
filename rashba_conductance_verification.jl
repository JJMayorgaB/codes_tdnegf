#!/usr/bin/env julia
# Verificación TDNEGF con Rashba: device γso≠0 + leads limpios.
# Referencia = método recursivo (decimación + slices) sobre EL MISMO H_ab.

using Pkg
Pkg.activate(@__DIR__)

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using Statistics
using Printf
using DelimitedFiles

# Sin graficación: este script solo calcula y escribe CSV.
# Las figuras se hacen aparte con rashba_plot_from_csv.jl

println("Threads disponibles: ", Threads.nthreads())

# ── parámetros ────────────────────────────────────────────────────────────────
const Nx    = 3
const Ny    = 3
const Nσ    = 2
const N_orb = 1
const N_λ1  = 49
const N_λ2  = 30
const β     = 40.0
const γ     = 1.0
const γso   = 1.0 + 0.0im          # t_SO/t_O = 1  ⇒  L_SO ≈ π sitios
const γc    = 1.0                  # bond de interfaz, limpio
const δV    = 0.01
const t_max = 500.0

const M   = Ny*Nσ                  # 6 = tamaño de rebanada
const Ns  = Nx*Ny*Nσ*N_orb         # 18
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

E_F_vals = collect(range(-3.0, 3.0; length = 32))

# E_F donde además se guarda la traza temporal completa (carga + espín)
const E_F_traces = [0.0, 1.5]        # 3 canales abiertos | 2 canales
const N_trace_pts = 500              # puntos guardados en [0, t_max]

# ── bloques: extraídos de build_H_ab para garantizar el mismo H ──────────────
H_ab    = build_H_ab(; Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, γ=γ, γso=γso)
H_clean = build_H_ab(; Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, γ=γ, γso=0.0+0.0im)

const H_ii    = Matrix(H_ab[1:M, 1:M])            # rebanada del device (Rashba)
const H_ip1_i = Matrix(H_ab[M+1:2M, 1:M])         # hopping en x (Rashba)
const H_i_ip1 = adjoint(H_ip1_i)

const H0_lead = Matrix(H_clean[1:M, 1:M])         # rebanada del lead (limpia)
const H1_lead = Matrix(H_clean[M+1:2M, 1:M])      # hopping del lead (limpio)
const V_c     = ComplexF64.(-γc * Matrix(I, M, M))  # V ∝ 𝟙 ⇒ Σ = γc² g

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 1: referencia estática (decimación + recursión por rebanadas)
# ═══════════════════════════════════════════════════════════════════════════════

function lead_G(E, eta, steps, H_1, H_0)
    Id = Matrix{ComplexF64}(I, M, M)
    A = ComplexF64.(H_1)
    B = adjoint(A)
    C = (E + eta*im)*Id - H_0
    D = (E + eta*im)*Id - H_0
    for _ in 1:steps
        Dinv = inv(D)
        A_new = A * Dinv * A
        B_new = B * Dinv * B
        C_new = C - A * Dinv * B
        D_new = D - A * Dinv * B - B * Dinv * A
        A, B, C, D = A_new, B_new, C_new, D_new
        norm(A) + norm(B) < 1e-12 && break
    end
    return inv(C)
end

function transmission_rec(E::Float64; eta::Float64 = 1e-6)
    g  = lead_G(E, eta, 100, H1_lead, H0_lead)     # H1 hermítico ⇒ g_L = g_R
    Σ  = V_c * g * adjoint(V_c)
    ΓL = im*(Σ - adjoint(Σ));  ΓR = ΓL

    EI = (E + eta*im) * Matrix{ComplexF64}(I, M, M)
    G_11_f = inv(EI - H_ii - Σ)
    G_22_f = inv(EI - H_ii - H_ip1_i * G_11_f * H_i_ip1)
    G_12_f = G_11_f * H_i_ip1 * G_22_f
    G_33_f = inv(EI - H_ii - Σ - H_ip1_i * G_22_f * H_i_ip1)
    G_13   = G_12_f * H_i_ip1 * G_33_f

    return real(tr(ΓL * G_13 * ΓR * adjoint(G_13)))
end

# G en unidades de 2e²/h:  G = (1/2δV) ∫ T(E)[f_L - f_R] dE
function G_ref_2e2h(E_F::Float64; n_pts::Int = 4000)
    μL = E_F + δV/2;  μR = E_F - δV/2
    grid = range(-5.0, 5.0; length = n_pts);  dε = step(grid)
    acc = 0.0
    for ε in grid
        T  = transmission_rec(Float64(ε))
        fL = abs(β*(ε-μL)) > 500 ? (ε < μL ? 1.0 : 0.0) : 1.0/(1.0+exp(β*(ε-μL)))
        fR = abs(β*(ε-μR)) > 500 ? (ε < μR ? 1.0 : 0.0) : 1.0/(1.0+exp(β*(ε-μR)))
        acc += T*(fL - fR)*dε
    end
    return acc / (2δV)
end

println("\n=== Parte 1: referencia recursiva ===")
E_dense = collect(range(-4.0, 4.0; length = 400))
T_dense = [transmission_rec(E) for E in E_dense]
G_ref   = [G_ref_2e2h(E_F) for E_F in E_F_vals]
@printf("T máx en el barrido denso: %.4f  (sin Rashba serían 6)\n", maximum(T_dense))

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 2: TDNEGF dinámico
# ═══════════════════════════════════════════════════════════════════════════════
println("\nConstruyendo objetos compartidos...")

Rλ, zλ = load_poles_square(N_λ1, N_λ2)
ξ_anL  = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=1,  y_coup=1:Ny)
ξ_anR  = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=Nx, y_coup=1:Ny)
p_ref  = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
                           Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
const Nc = p_ref.Nc
println("  Nc=$Nc  Ns=$(p_ref.Ns)  |ξ_L - ξ_R| = $(norm(ξ_anL - ξ_anR))")

"Propaga un punto y devuelve las observables en los tiempos `t_save`."
function run_tdnegf(E_F::Float64, t_save::Vector{Float64})
    μ_L = E_F + δV/2
    μ_R = E_F - δV/2

    χ_L  = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_L)
    χ_R  = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_R)
    ΣL_L = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_L)
    ΣG_L = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_L)
    ΣL_R = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_R)
    ΣG_R = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=μ_R)

    blocks = [SelfEnergyBlock(:left,  Nc, N_λ1, N_λ2, ΣL_L, ΣG_L, χ_L, ξ_anL),
              SelfEnergyBlock(:right, Nc, N_λ1, N_λ2, ΣL_R, ΣG_R, χ_R, ξ_anR)]

    p_model = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
                                Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
    p_model.H_ab  .= H_ab          # ← device CON Rashba
    p_model.H0_ab .= H_ab

    pb = ExperimentalBlockRHSParams(copy(H_ab), blocks,
                                    ComplexF64[+δV/2, -δV/2], p_model)

    Ns_sq = p_model.Ns^2
    ρ0 = ρ_eq(E_F, β, H_ab, 30, Nx, Ny, Nσ, N_orb)
    u0 = zeros(ComplexF64, Ns_sq + pb.aux_layout.total_size)
    u0[1:Ns_sq] .= vec(ρ0)

    sol = solve(ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_max), pb), Vern7();
                reltol=1e-8, abstol=1e-10,
                dense=false, save_everystep=false, saveat=t_save)

    obs = ObservablesTDNEGF(p_model; N_tmax=length(sol.t), N_leads=2)
    obs.t .= sol.t
    for (it, ut) in enumerate(sol.u)
        obs.idx = it
        obs_Ixα!(pointer_blocks(ut, pb.dims_ρ_ab, pb.aux_layout), pb, obs)
    end
    return obs
end

# obs.Iα = 2·trΠ y obs.Iαx = 2·s  (observables.jl) ⇒ el ½ corrige ese factor.
"Conductancia estacionaria en unidades de 2e²/h (π·I/δV pasa de e²/h a 2e²/h)."
function steady_G(E_F::Float64)
    obs = run_tdnegf(E_F, collect(range(0.8*t_max, t_max; length = 20)))
    I_L = mean(obs.Iα[1, :])
    I_R = mean(obs.Iα[2, :])
    return 0.5 * π*I_L/δV, abs(I_L + I_R)
end

println("\n=== Parte 2: TDNEGF (paralelo, $(Threads.nthreads()) hilos) ===")
N_pts = length(E_F_vals)
G_tdn = Vector{Float64}(undef, N_pts)
dI    = Vector{Float64}(undef, N_pts)

@time Threads.@threads for i in 1:N_pts
    G_tdn[i], dI[i] = steady_G(E_F_vals[i])
end

# ── trazas temporales: corriente de carga y de espín ─────────────────────────
println("\n=== Parte 2b: trazas temporales en E_F = $(E_F_traces) ===")
t_trace = collect(range(0.0, t_max; length = N_trace_pts))

@time Threads.@threads for E_F in E_F_traces
    obs = run_tdnegf(E_F, t_trace)
    tag = replace(@sprintf("%+.2f", E_F), "." => "p")

    # ½ por el factor del observable; el signo de R se voltea para que ambas
    # corrientes se midan en el mismo sentido (izquierda → derecha).
    I_L  =  0.5 .* obs.Iα[1, :]
    I_R  = -0.5 .* obs.Iα[2, :]
    Is_L =  0.5 .* obs.Iαx[1, :, :]      # (3, N_t) : componentes x, y, z
    Is_R = -0.5 .* obs.Iαx[2, :, :]

    writedlm(joinpath(OUT, "rashba_trace_EF$(tag).csv"),
             vcat(["t" "I_L" "I_R" "Isx_L" "Isy_L" "Isz_L" "Isx_R" "Isy_R" "Isz_R"],
                  hcat(obs.t, I_L, I_R,
                       Is_L[1,:], Is_L[2,:], Is_L[3,:],
                       Is_R[1,:], Is_R[2,:], Is_R[3,:])), ",")

    @printf("  E_F=%+.2f  I_L(∞)=%.5e  |I_L+I_R|=%.2e  |Is|max=(%.2e, %.2e, %.2e)\n",
            E_F, I_L[end], abs(I_L[end] - I_R[end]),
            maximum(abs, Is_L[1,:]), maximum(abs, Is_L[2,:]), maximum(abs, Is_L[3,:]))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 3: comparación
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "="^74)
println("G en unidades de 2e²/h   (TDNEGF ya corregido por el ×2 de obs.Iα)")
println("="^74)
@printf("%-8s  %-12s  %-12s  %-10s  %-12s\n",
        "E_F", "G_recursivo", "G_TDNEGF", "err(%)", "|I_L+I_R|")
println("-"^74)
for i in 1:N_pts
    err = abs(G_ref[i]) > 1e-8 ? 100*abs(G_tdn[i]-G_ref[i])/abs(G_ref[i]) : NaN
    @printf("%-8.2f  %-12.6f  %-12.6f  %-10.4f  %-12.2e\n",
            E_F_vals[i], G_ref[i], G_tdn[i], err, dI[i])
end

msk = abs.(G_ref) .> 1e-6
@printf("\nerror relativo medio: %.4f %%   |   máx: %.4f %%\n",
        100*mean(abs.(G_tdn[msk].-G_ref[msk])./G_ref[msk]),
        100*maximum(abs.(G_tdn[msk].-G_ref[msk])./G_ref[msk]))

# ═══════════════════════════════════════════════════════════════════════════════
# Parte 4: salida a CSV (las figuras se hacen con rashba_plot_from_csv.jl)
# ═══════════════════════════════════════════════════════════════════════════════
writedlm(joinpath(OUT, "rashba_conductance_points.csv"),
         vcat(["E_F" "G_recursivo" "G_TDNEGF" "dI"],
              hcat(E_F_vals, G_ref, G_tdn, dI)), ",")

writedlm(joinpath(OUT, "rashba_conductance_dense.csv"),
         vcat(["E" "G_recursivo"], hcat(E_dense, T_dense ./ 2)), ",")

println("\nCSV escritos en $OUT:")
println("  rashba_conductance_points.csv   ($(N_pts) puntos: recursivo vs TDNEGF)")
println("  rashba_conductance_dense.csv    ($(length(E_dense)) puntos: curva de referencia)")
for E_F in E_F_traces
    println("  rashba_trace_EF$(replace(@sprintf("%+.2f", E_F), "." => "p")).csv")
end
