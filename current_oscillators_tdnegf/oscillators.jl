#!/usr/bin/env julia
#=
  oscillators.jl  --  PREPARACION DEL STEADY STATE  (variante SIN g1)

  Igual que oscillators_tdnegf/oscillators.jl pero sin los 5 espines de la onda
  viajera. Aqui la cadena es simetrica alrededor del driver:

      g2 (10 libres)  |  g3 (driver)  |  g4 (10 libres)

  La excitacion ya no va a venir de una onda cinematica impuesta sobre espines
  clasicos, sino de INYECTAR UNA CORRIENTE POLARIZADA por los leads via TDNEGF.
  Por eso g1 desaparece por completo: no hay t_on_g1, ni k, ni barrido en r.

  Este script solo prepara el estado estacionario con el driver g3 encendido y
  guarda un CHECKPOINT COMPLETO (ρ_ab + auxiliares de los leads + espines), que
  es lo unico con lo que se puede reanudar exactamente la dinamica despues.

  Protocolo temporal:
    t ∈ [0, t_relax)        relajacion con damping fuerte, driver apagado
    t = t_on_g3 = t_relax    arranca g3 (precesion uniforme), damping debil
    t ∈ [t_on_g3, t_final]   el sistema se asienta
    t = t_final              se escribe el checkpoint

  Salidas en  output/steady_state_<param_tag>/
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TDNEGF
using DifferentialEquations
using Sunny
using LinearAlgebra
using LinearAlgebra: BLAS
using StaticArrays
using Printf
using DelimitedFiles
using JLD2

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# Geometría
const N_SPINS   = 21
const Nx, Ny    = 2 * N_SPINS + 1, 1        # 43 sitios electrónicos
const Nσ, N_orb = 2, 1

# sitio electrónico (1-based) del espín m: 2,4,...,42
@inline elec_site(m::Int) = 2 * m

# Sin g1: la cadena queda simetrica alrededor del driver (10 | 1 | 10).
# Se conservan los nombres g2/g3/g4 para que la correspondencia con el proyecto
# de oscillators_tdnegf sea directa (el driver sigue siendo g3 en ambos).
const GROUPS = (
    g2 = 1:10,     # libre (LLG)
    g3 = 11:11,    # precesión uniforme, arranca en t_on_g3
    g4 = 12:21,    # libre (LLG)
)
const DRIVEN = collect(GROUPS.g3)
const FREE   = vcat(collect(GROUPS.g2), collect(GROUPS.g4))

# Parametros físicos
const γso   = 0.1
const γ     = sqrt(1.0 - γso^2)
const γ_eff = sqrt(γ^2 + γso^2)

const E_F = 0.0                    # sin bias en la preparacion
const β   = 40.0
const N_λ1, N_λ2 = 49, 30
const j_sd = 0.5

const Δt = 0.1

const damping_relax = 1.0
const damping_dyn   = 0.05
const kT            = 0.0

# driving
const θ_max   = deg2rad(10.0)
const Ω       = 0.01
const T_drive = 2π / Ω             # 628.32
const t_rise  = 630.0              # ~1 periodo, encendido adiabatico de g3
const t_relax = 10000.0             # relajacion de leads/electrones
const t_on_g3 = 10000.0             # el driver arranca donde termina la relajacion
const t_final = 20000.0            # + 7500 de driver ≈ 11.9 periodos

# Etiqueta de parametros. Lleva el numero de espines porque esta variante y la
# de oscillators_tdnegf comparten los mismos parametros fisicos y solo difieren
# en la geometria: sin el n<N_SPINS> las etiquetas serian identicas.
@inline fmtnum(x::Real) = replace(string(round(Float64(x); digits = 4)), "." => "p", "-" => "m")
param_tag() = "n$(N_SPINS)_gso$(fmtnum(γso))_jsd$(fmtnum(j_sd))_th$(round(Int, rad2deg(θ_max)))deg_Om$(fmtnum(Ω))"

const OUT_RUN = joinpath(OUT, "steady_state_" * param_tag()); mkpath(OUT_RUN)
const CKPT = joinpath(OUT_RUN, "checkpoint_t$(round(Int, t_final)).jld2")

# Driving cinemático del driver: precesion uniforme en un cono de angulo θ_max
@inline smooth_switch(τ, ti) = τ < 0 ? 0.0 : (τ < ti ? sin((π / 2) * τ / ti)^2 : 1.0)

@inline function pumped_spin(t::Float64, t_on::Float64)
    θ = θ_max * smooth_switch(t - t_on, t_rise)
    φ = -Ω * (t - t_on)
    return SVector{3,Float64}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
end

function force_driven!(sys, t::Float64)
    for m in GROUPS.g3
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, t_on_g3)
    end
    return nothing
end

# Sistema de espines (Sunny)
function init_spins()
    latvecs   = lattice_vectors(1.0, 1.0 * (1 + 1e-3), 4.0, 90, 90, 90)
    positions = [[0.5, 0.5, 0.0]]
    cryst     = Crystal(latvecs, positions)
    moments   = [1 => Moment(s = 1.0, g = 1.0)]
    sys = System(cryst, moments, :dipole; dims = (N_SPINS, 1, 1))
    for m in 1:N_SPINS
        sys.dipoles[m, 1, 1, 1] = Sunny.SVector(0.0, 0.0, 1.0)
    end
    return sys
end

function full_dipoles(sys)
    S = Matrix{SVector{3,Float64}}(undef, Nx, Ny)
    fill!(S, SVector{3,Float64}(0.0, 0.0, 0.0))
    for m in 1:N_SPINS
        S[elec_site(m), 1] = SVector{3,Float64}(sys.dipoles[m, 1, 1, 1])
    end
    return S
end

function update_H_s_free!(sys, σx_i_now)
    for m in FREE
        site = elec_site(m)
        Sunny.set_field_at!(sys, -j_sd .* σx_i_now[site, :], (m, 1, 1, 1))
    end
    for m in DRIVEN
        Sunny.set_field_at!(sys, [0.0, 0.0, 0.0], (m, 1, 1, 1))
    end
    return nothing
end

# Metadatos compartidos por todas las salidas
prep_params() = (γ = γ, γso = γso, γ_eff = γ_eff, j_sd = j_sd, θmax = θ_max, Ω = Ω,
                 E_F = E_F, β = β, N_λ1 = N_λ1, N_λ2 = N_λ2, Δt = Δt,
                 t_on_g3 = t_on_g3, t_rise = t_rise,
                 t_relax = t_relax, t_final = t_final,
                 damping_relax = damping_relax, damping_dyn = damping_dyn, kT = kT)

geometry() = (N_SPINS = N_SPINS, Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb)

# Checkpoint: lo unico con lo que se puede reanudar la dinamica
function save_checkpoint(intg, sys)
    S_final = Array{Float64}(undef, 3, N_SPINS)
    for m in 1:N_SPINS, c in 1:3
        S_final[c, m] = sys.dipoles[m, 1, 1, 1][c]
    end
    jldsave(CKPT;
            u = collect(intg.u), t = intg.t, dipoles = S_final,
            len_u = length(intg.u),
            geometry = geometry(), params = prep_params(),
            groups = (g2 = collect(GROUPS.g2), g3 = collect(GROUPS.g3),
                      g4 = collect(GROUPS.g4)))
    @printf("Checkpoint escrito: %s  (t=%.1f, |u|=%d, %.2f MB)\n",
            basename(CKPT), intg.t, length(intg.u), 16 * length(intg.u) / 1e6)
    return nothing
end

# Observables de la preparacion
function save_outputs(obs, S_hist)
    t  = obs.t
    Nt = length(t)

    jldsave(joinpath(OUT_RUN, "prep_fields.jld2");
            t = t, s_i = S_hist,
            sigma_i = obs.σx_i, sigma_eq = obs.σx_i_eq, n_i = obs.n_i,
            I_alpha = obs.Iα, I_alpha_x = obs.Iαx,
            groups = (g2 = collect(GROUPS.g2), g3 = collect(GROUPS.g3),
                      g4 = collect(GROUPS.g4)),
            driven = DRIVEN, free = FREE,
            elec_sites = [elec_site(m) for m in 1:N_SPINS],
            geometry = geometry(), params = prep_params())

    header = ["t", "I_L", "I_R", "Isx_L", "Isy_L", "Isz_L", "Isx_R", "Isy_R", "Isz_R"]
    for m in 1:N_SPINS
        site = elec_site(m)
        append!(header, ["S$(m)_x_site$(site)", "S$(m)_y_site$(site)", "S$(m)_z_site$(site)"])
    end

    data = Matrix{Any}(undef, Nt, length(header))
    for i in 1:Nt
        col = 1
        data[i, col] = t[i]; col += 1
        data[i, col] =  0.5 * obs.Iα[1, i]; col += 1
        data[i, col] = -0.5 * obs.Iα[2, i]; col += 1
        data[i, col] =  0.5 * obs.Iαx[1, 1, i]; col += 1
        data[i, col] =  0.5 * obs.Iαx[1, 2, i]; col += 1
        data[i, col] =  0.5 * obs.Iαx[1, 3, i]; col += 1
        data[i, col] = -0.5 * obs.Iαx[2, 1, i]; col += 1
        data[i, col] = -0.5 * obs.Iαx[2, 2, i]; col += 1
        data[i, col] = -0.5 * obs.Iαx[2, 3, i]; col += 1
        for m in 1:N_SPINS
            data[i, col] = S_hist[1, m, i]; col += 1
            data[i, col] = S_hist[2, m, i]; col += 1
            data[i, col] = S_hist[3, m, i]; col += 1
        end
    end
    writedlm(joinpath(OUT_RUN, "prep_trace.csv"), vcat(permutedims(header), data), ",")
    println("Observables: prep_fields.jld2  prep_trace.csv")
    return nothing
end

function write_params_label()
    open(joinpath(OUT_RUN, "params.txt"), "w") do io
        println(io, "PREPARACION DEL STEADY STATE  (variante SIN g1)")
        println(io, "param_tag = ", param_tag())
        println(io, "checkpoint = ", basename(CKPT))
        println(io, "")
        println(io, "N_SPINS = ", N_SPINS, "   Nx = ", Nx, "   Ny = ", Ny,
                    "   Nσ = ", Nσ, "   N_orb = ", N_orb)
        println(io, "g2 = ", GROUPS.g2, "  (libre)")
        println(io, "g3 = ", GROUPS.g3, "  (driver, precesion uniforme)")
        println(io, "g4 = ", GROUPS.g4, "  (libre)")
        println(io, "")
        println(io, "γ       = ", γ)
        println(io, "γso     = ", γso)
        println(io, "γ_eff   = ", γ_eff)
        println(io, "j_sd    = ", j_sd)
        println(io, "θ_max   = ", rad2deg(θ_max), " deg (", θ_max, " rad)")
        println(io, "Ω       = ", Ω, "   (periodo T = ", T_drive, ")")
        println(io, "")
        println(io, "E_F     = ", E_F)
        println(io, "β       = ", β)
        println(io, "N_λ1    = ", N_λ1)
        println(io, "N_λ2    = ", N_λ2)
        println(io, "Δt      = ", Δt)
        println(io, "")
        println(io, "damping_relax = ", damping_relax)
        println(io, "damping_dyn   = ", damping_dyn)
        println(io, "kT            = ", kT)
        println(io, "")
        println(io, "t_relax = ", t_relax)
        println(io, "t_on_g3 = ", t_on_g3)
        println(io, "t_rise  = ", t_rise)
        println(io, "t_final = ", t_final,
                    "   (", round((t_final - t_on_g3)/T_drive, digits = 2), " periodos de driver)")
    end
    println("params.txt escrito")
    return nothing
end

function run_prep()
    p_model = ModelParamsTDNEGF(Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb,
                                 Nα = 2, N_λ1 = N_λ1, N_λ2 = N_λ2)
    H0 = build_H_ab(; Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb,
                     γ = γ, γso = complex(γso, 0.0))

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    Σᴸ = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)
    Σᴳ = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)
    χ  = build_χ_nλ(zλ,      Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)

    ξ_L = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = 1,  y_coup = 1:Ny)
    ξ_R = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = Nx, y_coup = 1:Ny)

    blocks = [SelfEnergyBlock(:left,  p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_L),
              SelfEnergyBlock(:right, p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_R)]

    p_model.H0_ab .= H0
    p_model.H_ab  .= H0
    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, ComplexF64[0.0, 0.0], p_model)

    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)
    @printf("Vector de estado: |u| = %d  (%.2f MB por checkpoint)\n",
            length(u0), 16 * length(u0) / 1e6)

    sys = init_spins()
    site_ranges = [get_sub(i, p_model.N_loc) for i in 1:p_model.N_sites]

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_final), p_blocks)
    intg = init(prob, Vern7(); dt = Δt, save_everystep = false, adaptive = true, dense = false)

    llg_relax = Langevin(Δt; damping = damping_relax, kT = kT)
    llg_dyn   = Langevin(Δt; damping = damping_dyn,   kT = kT)

    N_steps = Int(round(t_final / Δt))
    obs = ObservablesTDNEGF(p_model; N_tmax = N_steps, N_leads = 2)
    S_hist = Array{Float64}(undef, 3, N_SPINS, N_steps)

    started = time()
    for i in 1:N_steps
        obs.idx = i
        llg = intg.t < t_relax ? llg_relax : llg_dyn

        DifferentialEquations.step!(intg, Δt, true)
        Sunny.step!(sys, llg)
        force_driven!(sys, intg.t)

        dv = pointer_blocks(intg.u, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        ρ  = ρ_eq(E_F, β, p_model.H_ab, N_λ2, Nx, Ny, Nσ, N_orb)

        obs.t[i] = intg.t
        obs_n_i!(dv, p_model, obs)
        obs_σ_i!(dv, p_model, obs)
        obs_Ixα!(dv, p_blocks, obs)
        obs_σ_i_eq!(ρ, p_model, obs)

        for m in 1:N_SPINS, c in 1:3
            S_hist[c, m, i] = sys.dipoles[m, 1, 1, 1][c]
        end

        update_H_s_free!(sys, obs.σx_i[:, :, i])
        update_H_e!(p_model, site_ranges, full_dipoles(sys), j_sd)

        if i % 1000 == 0
            etapa = intg.t < t_on_g3 ? "relajacion" : "driver g3"
            @printf("  t=%7.1f/%.0f  [%s]  I_L=% .3e  I_R=% .3e  elapsed=%.0fs\n",
                    intg.t, t_final, etapa, 0.5 * obs.Iα[1, i], -0.5 * obs.Iα[2, i],
                    time() - started)
            flush(stdout)
        end
    end
    @printf("\nDinamica lista en %.1f s\n", time() - started)

    save_checkpoint(intg, sys)
    save_outputs(obs, S_hist)
    return nothing
end

function main()
    BLAS.set_num_threads(Sys.CPU_THREADS)

    println("="^70)
    println("PREPARACION DEL STEADY STATE   (sin g1: 10 libres | driver | 10 libres)")
    println("="^70)
    @printf("Cadena Rashba: Nx=%d (=2·%d+1)  γ=%.4f  γso=%.4f  γ_eff=%.4f\n",
            Nx, N_SPINS, γ, γso, γ_eff)
    @printf("Jsd=%.3f  θmax=%.2f°  Ω=%.4f  (periodo T=%.1f)\n",
            j_sd, rad2deg(θ_max), Ω, T_drive)
    @printf("g2(libre)=%s  g3(driver)=%s  g4(libre)=%s\n",
            GROUPS.g2, GROUPS.g3, GROUPS.g4)
    @printf("t_relax=%.0f  t_on_g3=%.0f  t_rise=%.0f  t_final=%.0f  (%.1f periodos de driver)\n",
            t_relax, t_on_g3, t_rise, t_final, (t_final - t_on_g3) / T_drive)
    @printf("Salidas en %s\n", OUT_RUN)
    println("="^70)
    flush(stdout)

    write_params_label()
    run_prep()

    @printf("\nListo. Checkpoint para la inyeccion de corriente:\n  %s\n", CKPT)
end

main()
