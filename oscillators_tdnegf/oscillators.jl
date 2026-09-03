#!/usr/bin/env julia
#
# Cadena 1D con acoplamiento Rashba (TDNEGF) + 16 espines clásicos acoplados
# vía Jsd, sin exchange directo entre ellos.
#
#   H_e = Σ_i,σσ'  ε0 c†_iσ c_iσ  - t(c†_{i+1}σ c_iσ + h.c.)
#         - iλ( c†_iσ [σy]_σσ' c_{i+1}σ' - c†_{i+1}σ [σy]_σσ' c_iσ' )
#
# se construye con build_H_ab(γ=t, γso=λ) — verificado contra
# TDNEGF/src/hamiltonians.jl: para Ny=1 da exactamente H_{i,i+1}=-tσ0-iλσy,
# H_{i+1,i}=-tσ0+iλσy, ε0=0.
#
# 33 sitios electrónicos (2·16+1). Los 16 espines viven en los sitios pares
# 2,4,...,32 (uno de por medio libre). Grupos:
#   grupo1 = espines 1-5    (sitios 2..10)   onda viajera, arranca en t=300
#   grupo2 = espines 6-10   (sitios 12..20)  libre (LLG)
#   grupo3 = espín 11       (sitio 22)       precesión uniforme, t=100
#   grupo4 = espines 12-16  (sitios 24..32)  libre (LLG)
#
# Grupo1 y grupo3 son cinemáticos: M(t) se sustituye directo en el
# acoplamiento sd, nunca entran a la integración LLG (igual que en
# Project_nonreciprocal_magnons/scripts/pump_two_emitters.jl). Antes de su
# t_on, θ=0 ⇒ M=(0,0,1), igual que el resto — no hace falta tratarlos aparte.
#
# Arquitectura del solver: la misma que magnonic_bh_* — DOS integradores en
# paralelo avanzados juntos cada Δt: DifferentialEquations.step! para el
# bloque electrónico (TDNEGF) y Sunny.step! (Langevin) para los espines
# clásicos. Los espines del sistema Sunny (16 sitios, sin exchange ni
# anisotropía) evolucionan libremente; justo después de cada Sunny.step! se
# sobreescriben a la fuerza los de grupo1/grupo3 con su valor cinemático,
# igual que ya se hace con los sitios "removidos" en la constricción, solo
# que aquí el valor forzado depende de t.
#
# Sin bias: los leads son baños térmicos puros (μ_L=μ_R=E_F=0). Sin ruido de
# Langevin en los espines (kT=0): la única disipación entra por los leads.
#
#   julia --project=. oscillators_tdnegf/oscillators.jl

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

# ═══════════════════════════════════════════════════════════════════════════
# Geometría
# ═══════════════════════════════════════════════════════════════════════════
const N_SPINS   = 16
const Nx, Ny    = 2 * N_SPINS + 1, 1        # 33 sitios electrónicos
const Nσ, N_orb = 2, 1

# sitio electrónico (1-based) del espín m: 2,4,...,32
@inline elec_site(m::Int) = 2 * m

const GROUPS = (
    g1 = 1:5,      # onda viajera,        arranca en t_on_g1
    g2 = 6:10,     # libre (LLG)
    g3 = 11:11,    # precesión uniforme,  arranca en t_on_g3
    g4 = 12:16,    # libre (LLG)
)
const DRIVEN = vcat(collect(GROUPS.g1), collect(GROUPS.g3))
const FREE   = vcat(collect(GROUPS.g2), collect(GROUPS.g4))

# ═══════════════════════════════════════════════════════════════════════════
# Parámetros físicos
# ═══════════════════════════════════════════════════════════════════════════
const γ   = 1.0 / sqrt(2)          # hopping t
const γso = 1.0 / sqrt(2)          # Rashba λ  (γ=λ ⇒ γ_eff=√(t²+λ²)=1)
const γ_eff = sqrt(γ^2 + γso^2)

const E_F = 0.0                    # sin bias: μ_L=μ_R=E_F
const β   = 40.0
const N_λ1, N_λ2 = 49, 20
const j_sd = 1.0                   # mismo Jsd en los 16 sitios

const Δt = 0.1

const damping_relax = 0.5
const damping_dyn   = 0.007
const kT            = 0.0          # sin ruido de Langevin: disipan los leads

# --- driving cinemático ---
const θ_max   = deg2rad(15.0)
const Ω       = 0.5
const k_wave  = Ω / γ_eff          # resonancia con v_F=2γ_eff ⇒ k=Ω/γ_eff=0.5
const t_rise  = 10.0
const t_on_g3 = 100.0
const t_on_g1 = 300.0
const t_relax = 100.0
const t_final = 700.0              # ajustar si hace falta ver más tiempo

# ═══════════════════════════════════════════════════════════════════════════
# Driving cinemático: M(t) = [sinθ(t)cos(kx-Ω(t-t_on)), sinθ(t)sin(...), cosθ(t)]
# ═══════════════════════════════════════════════════════════════════════════
@inline smooth_switch(τ, ti) = τ < 0 ? 0.0 : (τ < ti ? sin((π / 2) * τ / ti)^2 : 1.0)

@inline function pumped_spin(t::Float64, x::Float64, k::Float64, t_on::Float64)
    θ = θ_max * smooth_switch(t - t_on, t_rise)
    φ = k * x - Ω * (t - t_on)
    return SVector{3,Float64}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
end

# (x, k, t_on) de cada espín impulsado. x = índice del espín (1..16); para
# grupo3, k=0 así que x no importa.
@inline function drive_of(m::Int)
    m in GROUPS.g3 && return (0.0, 0.0, t_on_g3)
    m in GROUPS.g1 && return (Float64(m), k_wave, t_on_g1)
    error("drive_of llamado con un espín libre")
end

# ═══════════════════════════════════════════════════════════════════════════
# Sistema de espines (Sunny): 16 sitios, sin exchange ni anisotropía —
# "entre los espines no debe haber interacciones, la única es Jsd".
# ═══════════════════════════════════════════════════════════════════════════
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

"""
Matriz (Nx×Ny) de espines para alimentar `update_H_e!`: cero en los sitios
electrónicos impares (sin espín asociado), el valor actual de Sunny en los
pares. `update_H_e!` de la librería recorre TODOS los sitios electrónicos,
así que los impares deben entrar como vector nulo (acoplamiento sd = 0 ahí).
"""
function full_dipoles(sys)
    S = Matrix{SVector{3,Float64}}(undef, Nx, Ny)
    fill!(S, SVector{3,Float64}(0.0, 0.0, 0.0))
    for m in 1:N_SPINS
        S[elec_site(m), 1] = SVector{3,Float64}(sys.dipoles[m, 1, 1, 1])
    end
    return S
end

"Sobreescribe a la fuerza grupo1/grupo3 con su valor cinemático en el tiempo t."
function force_driven!(sys, t::Float64)
    for m in DRIVEN
        x, k, t_on = drive_of(m)
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, x, k, t_on)
    end
    return nothing
end

"""
Campo efectivo -Jsd·σ_local sobre los espines libres (grupo2, grupo4), leído
de `σx_i_now` (matriz N_sitios_electrónicos×3 en el paso actual). Los
impulsados no reciben campo (se sobreescriben aparte en `force_driven!`).
"""
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

# ═══════════════════════════════════════════════════════════════════════════
# Salida
# ═══════════════════════════════════════════════════════════════════════════
function save_outputs(obs, S_hist)
    t  = obs.t
    Nt = length(t)

    jldsave(joinpath(OUT, "oscillators_fields.jld2");
            t = t, s_i = S_hist,
            sigma_i = obs.σx_i, sigma_eq = obs.σx_i_eq, n_i = obs.n_i,
            I_alpha = obs.Iα, I_alpha_x = obs.Iαx,
            groups = (g1 = collect(GROUPS.g1), g2 = collect(GROUPS.g2),
                      g3 = collect(GROUPS.g3), g4 = collect(GROUPS.g4)),
            driven = DRIVEN, free = FREE,
            elec_sites = [elec_site(m) for m in 1:N_SPINS],
            params = (γ = γ, γso = γso, j_sd = j_sd, θmax = θ_max, Ω = Ω,
                      k = k_wave, t_on_g3 = t_on_g3, t_on_g1 = t_on_g1,
                      t_rise = t_rise, t_relax = t_relax,
                      damping_relax = damping_relax, damping_dyn = damping_dyn,
                      kT = kT, β = β, N_λ1 = N_λ1, N_λ2 = N_λ2))

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
    writedlm(joinpath(OUT, "oscillators_trace.csv"), vcat(permutedims(header), data), ",")
    @printf("  → oscillators_fields.jld2   oscillators_trace.csv\n")
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
function main()
    BLAS.set_num_threads(8)
    Rλ, zλ = load_poles_square(N_λ1, N_λ2)

    @printf("Cadena Rashba: Nx=%d (33=2·16+1)  γ=t=%.4f  γso=λ=%.4f  γ_eff=%.4f\n",
            Nx, γ, γso, γ_eff)
    @printf("Jsd=%.3f  θmax=%.2f°  Ω=%.3f  k=%.3f (=Ω/γ_eff)\n",
            j_sd, rad2deg(θ_max), Ω, k_wave)
    @printf("grupo1(onda)=%s  grupo2(libre)=%s  grupo3(driver)=%s  grupo4(libre)=%s\n",
            GROUPS.g1, GROUPS.g2, GROUPS.g3, GROUPS.g4)
    @printf("t_on_g3=%.0f  t_on_g1=%.0f  t_rise=%.0f  t_relax=%.0f  t_final=%.0f\n",
            t_on_g3, t_on_g1, t_rise, t_relax, t_final)
    flush(stdout)

    p_model = ModelParamsTDNEGF(Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb,
                                 Nα = 2, N_λ1 = N_λ1, N_λ2 = N_λ2)
    H0 = build_H_ab(; Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb,
                     γ = γ, γso = complex(γso, 0.0))

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

        if i % 200 == 0
            @printf("  t=%6.1f/%.0f   I_L=% .4e   I_R=% .4e   elapsed=%.0fs\n",
                    intg.t, t_final, 0.5 * obs.Iα[1, i], -0.5 * obs.Iα[2, i],
                    time() - started)
            flush(stdout)
        end
    end

    save_outputs(obs, S_hist)
    @printf("\nListo en %.1f s. Salidas en %s\n", time() - started, OUT)
end

main()
