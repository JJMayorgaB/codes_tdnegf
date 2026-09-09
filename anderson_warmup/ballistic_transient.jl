#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TDNEGF
using DifferentialEquations
using LinearAlgebra
using LinearAlgebra: BLAS
using Statistics
using Printf
using DelimitedFiles

# Parametros
const Nx    = 5            # wire extendido: leads en columnas 1 y Nx (distintas)
const Ny    = 2            # 2 canales transversales
const Nσ    = 2            # x2 por espin  ->  4 canales en total
const N_orb = 1
const N_λ1  = 49
const N_λ2  = 30
const β     = 40.0
const γ     = 1.0
const γso   = 0.0 + 0.0im
const δV    = 0.01         # bias pequeno: μ_L,μ_R = ±0.005, dentro del plateau |E|<1
const E_F   = 0.0          # centro de la banda, plateau T=4
const t_max = 600.0
const dt_save = 0.1

# Conversion a unidades fisicas (γ = 1 eV)
const γ_eV     = 1.0
const ħ_eVfs   = 0.6582119569
const T_TO_FS  = ħ_eVfs / γ_eV
const I_TO_2EG = π

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# Referencia estatica: T(E) de Landauer
function transmission_std(E::Float64; η = 1e-8)
    H   = build_H_ab(; Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb, γ = γ, γso = γso)
    dim = size(H, 1)
    Id  = Matrix{ComplexF64}(LinearAlgebra.I, dim, dim)
    ΣL  = TDNEGF.ΣL_tot(complex(E, η); γ = γ, γc = γ, Nx = Nx, Ny = Ny, Nσ = Nσ)
    ΣR  = TDNEGF.ΣR_tot(complex(E, η); γ = γ, γc = γ, Nx = Nx, Ny = Ny, Nσ = Nσ)
    Gr  = inv((E + 1im*η)*Id - H - ΣL - ΣR)
    ΓL  = 1im*(ΣL - ΣL')
    ΓR  = 1im*(ΣR - ΣR')
    return real(tr(ΓL * Gr * ΓR * Gr'))
end

# Corrida transitoria
function run_transient()
    μ_L = E_F + δV/2
    μ_R = E_F - δV/2

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    H_ab   = build_H_ab(; Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb, γ = γ, γso = γso)
    ξ_anL  = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = 1,  y_coup = 1:Ny)
    ξ_anR  = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = Nx, y_coup = 1:Ny)

    p_model = ModelParamsTDNEGF(Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb,
                                 Nα = 2, N_λ1 = N_λ1, N_λ2 = N_λ2)
    Nc = p_model.Nc
    Ns = p_model.Ns

    χ_L  = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = μ_L)
    χ_R  = build_χ_nλ(zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = μ_R)
    Σᴸ_L = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = μ_L)
    Σᴳ_L = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = μ_L)
    Σᴸ_R = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = μ_R)
    Σᴳ_R = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = μ_R)

    left_block  = SelfEnergyBlock(:left,  Nc, N_λ1, N_λ2, Σᴸ_L, Σᴳ_L, χ_L, ξ_anL)
    right_block = SelfEnergyBlock(:right, Nc, N_λ1, N_λ2, Σᴸ_R, Σᴳ_R, χ_R, ξ_anR)

    p_model.H_ab  .= H_ab
    p_model.H0_ab .= H_ab
    Δ_blocks = ComplexF64[+δV/2, -δV/2]
    p_blocks = ExperimentalBlockRHSParams(copy(H_ab), [left_block, right_block],
                                          Δ_blocks, p_model)

    ρ0 = ρ_eq(E_F, β, H_ab, N_λ2, Nx, Ny, Nσ, N_orb)
    u0 = zeros(ComplexF64, Ns^2 + p_blocks.aux_layout.total_size)
    u0[1:Ns^2] .= vec(ρ0)

    # Se avanza el integrador paso a paso y se extraen las corrientes al vuelo.
    # Guardar la solucion completa con saveat costaria ~200 kB por punto
    # (rho + auxiliares de los leads) y reventaria la memoria.
    nt = length(0.0:dt_save:t_max)
    @printf("Resolviendo hasta t=%.0f  (%d puntos, Nc=%d, Ns=%d)\n",
            t_max, nt, Nc, Ns)
    flush(stdout)

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_max), p_blocks)
    intg = init(prob, Vern7(); dt = dt_save, reltol = 1e-8, abstol = 1e-10,
                dense = false, save_everystep = false, adaptive = true)

    obs = ObservablesTDNEGF(p_model; N_tmax = nt, N_leads = 2)

    started = time()
    for i in 1:nt
        i > 1 && DifferentialEquations.step!(intg, dt_save, true)
        obs.idx = i
        obs.t[i] = intg.t
        dv = pointer_blocks(intg.u, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        obs_Ixα!(dv, p_blocks, obs)
        if i % 1000 == 0
            @printf("  t=%7.1f/%.0f   I_R=% .4e   elapsed=%.0fs\n",
                    intg.t, t_max, obs.Iα[2, i], time() - started)
            flush(stdout)
        end
    end
    @printf("ODE resuelta en %.1f s\n", time() - started)

    # Convencion de normalizacion de obs.Iα (la misma que usa oscillators.jl):
    #   I_L = +0.5·Iα[1],  I_R = -0.5·Iα[2]
    # Verificado contra Landauer: sin el 0.5 el plateau sale en 8e²/h en vez de 4e²/h.
    return collect(obs.t), 0.5 .* collect(obs.Iα[1, :]), -0.5 .* collect(obs.Iα[2, :])
end

function main()
    # Ns=20 -> matrices chicas: BLAS multihilo no aporta y solo consume memoria.
    BLAS.set_num_threads(1)

    @printf("Alambre balistico: Nx=%d  Ny=%d  Nσ=%d  γ=%.2f  γso=0  β=%.1f\n",
            Nx, Ny, Nσ, γ, β)
    @printf("E_F=%.3f  δV=%.4f  (μ_L=%+.4f, μ_R=%+.4f)\n",
            E_F, δV, E_F + δV/2, E_F - δV/2)

    T_theory = transmission_std(E_F)
    @printf("\nLandauer estatico:  T(E_F=%.2f) = %.6f   ->  G = %.6f e²/h\n",
            E_F, T_theory, T_theory)
    flush(stdout)

    t, I_L, I_R = run_transient()

    # Estado estacionario: promedio sobre el ultimo 20%
    i0     = max(1, round(Int, 0.8 * length(t)))
    I_R_ss = mean(I_R[i0:end])
    I_L_ss = mean(I_L[i0:end])

    # G en unidades de e²/h:  G_var = I/(2π δV) y el plateau T vale 2π·G_var
    T_implied = 2π * abs(I_R_ss) / δV

    println("\n" * "="^64)
    @printf("I_L(ss) = %+.8e     I_R(ss) = %+.8e   [unidades naturales]\n", I_L_ss, I_R_ss)
    @printf("|I_L-I_R| = %.2e   (conservacion de corriente)\n", abs(I_L_ss - I_R_ss))
    @printf("T implicado por el transitorio = %.6f\n", T_implied)
    @printf("T exacto de Landauer           = %.6f\n", T_theory)
    @printf("razon  T_implied/T_theory      = %.6f   <-- debe dar 1.000\n",
            T_implied / T_theory)
    println("="^64)
    @printf("\nPlateau en unidades pedidas: I_R(∞) = %.6f  [2eγ/h]\n",
            I_TO_2EG * abs(I_R_ss))
    @printf("Duracion en fs: t_max = %.1f fs   (γ = %.1f eV)\n", T_TO_FS * t_max, γ_eV)

    # Salida
    hdr  = ["t_sim" "t_fs" "I_L_sim" "I_R_sim" "I_R_2egh"]
    data = hcat(t, T_TO_FS .* t, I_L, I_R, I_TO_2EG .* abs.(I_R))
    csv  = joinpath(OUT, "ballistic_transient.csv")
    writedlm(csv, vcat(hdr, data), ",")

    open(joinpath(OUT, "ballistic_transient_meta.txt"), "w") do io
        println(io, "Nx = ", Nx)
        println(io, "Ny = ", Ny)
        println(io, "Nsigma = ", Nσ)
        println(io, "N_orb = ", N_orb)
        println(io, "gamma = ", γ, "   (tomado como ", γ_eV, " eV)")
        println(io, "gamma_so = 0")
        println(io, "beta = ", β)
        println(io, "E_F = ", E_F)
        println(io, "deltaV = ", δV)
        println(io, "t_max = ", t_max)
        println(io, "dt_save = ", dt_save)
        println(io, "")
        println(io, "T_theory = ", T_theory)
        println(io, "T_implied = ", T_implied)
        println(io, "I_R_ss_sim = ", I_R_ss)
        println(io, "I_R_ss_2egh = ", I_TO_2EG * abs(I_R_ss))
        println(io, "")
        println(io, "t_fs = t_sim * ", T_TO_FS)
        println(io, "I_2egh = pi * I_sim")
    end

    println("\nGuardado: ", csv)
    return nothing
end

main()
