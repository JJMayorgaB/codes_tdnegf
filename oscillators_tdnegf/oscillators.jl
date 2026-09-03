#!/usr/bin/env julia

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

# Etiqueta de la corrida: identifica los parámetros físicos en el nombre de
# la carpeta, para no mezclar salidas de distintas configuraciones.
@inline fmtnum(x::Real) = replace(string(round(Float64(x); digits = 4)), "." => "p", "-" => "m")
run_tag() = "gso$(fmtnum(γso))_jsd$(fmtnum(j_sd))_th$(round(Int, rad2deg(θ_max)))deg_Om$(fmtnum(Ω))"

# Geometría
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

# Parmetros físicos 
const γso   = 1.0/sqrt(2.0)
const γ     = sqrt(1.0 - γso^2)     
const γ_eff = sqrt(γ^2 + γso^2)

const E_F = 0.0                    # sin bias
const β   = 40.0
const N_λ1, N_λ2 = 49, 20
const j_sd = 0.25                   

const Δt = 0.1

const damping_relax = 0.75
const damping_dyn   = 0.05
const kT            = 0.0         

#driving
const θ_max   = deg2rad(10.0)
const Ω       = 0.05
const k_mag   = Ω / γ_eff
const t_rise  = 125.0
const t_on_g3 = 500.0
const t_on_g1 = 2000.0
const t_relax = 500.0
const t_final = 6500.0

const R_VALUES = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)

const OUT_RUN = joinpath(OUT, run_tag()); mkpath(OUT_RUN)

const RUNS = Tuple(
    (name = "r$(replace(string(r), "." => "p"))_k$(s > 0 ? "pos" : "neg")",
     k = s * r * Ω / γ_eff)
    for r in R_VALUES for s in (+1, -1)
)


# Driving cinemático: M(t) 
@inline smooth_switch(τ, ti) = τ < 0 ? 0.0 : (τ < ti ? sin((π / 2) * τ / ti)^2 : 1.0)

@inline function pumped_spin(t::Float64, x::Float64, k::Float64, t_on::Float64)
    θ = θ_max * smooth_switch(t - t_on, t_rise)
    φ = k * x - Ω * (t - t_on)
    return SVector{3,Float64}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
end


function force_driven!(sys, t::Float64, k::Float64)
    for m in GROUPS.g3
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, 0.0, 0.0, t_on_g3)
    end
    for m in GROUPS.g1
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, Float64(m), k, t_on_g1)
    end
    return nothing
end


# Sistema de espines (Sunny): 16 sitios
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

#Salida
function save_outputs(name::String, k::Float64, obs, S_hist)
    t  = obs.t
    Nt = length(t)

    jldsave(joinpath(OUT, "oscillators_fields_$(name).jld2");
            t = t, s_i = S_hist,
            sigma_i = obs.σx_i, sigma_eq = obs.σx_i_eq, n_i = obs.n_i,
            I_alpha = obs.Iα, I_alpha_x = obs.Iαx,
            groups = (g1 = collect(GROUPS.g1), g2 = collect(GROUPS.g2),
                      g3 = collect(GROUPS.g3), g4 = collect(GROUPS.g4)),
            driven = DRIVEN, free = FREE,
            elec_sites = [elec_site(m) for m in 1:N_SPINS],
            params = (γ = γ, γso = γso, j_sd = j_sd, θmax = θ_max, Ω = Ω,
                      k = k, t_on_g3 = t_on_g3, t_on_g1 = t_on_g1,
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
    writedlm(joinpath(OUT, "oscillators_trace_$(name).csv"), vcat(permutedims(header), data), ",")
    @printf("  [%s] → oscillators_fields_%s.jld2   oscillators_trace_%s.csv\n", name, name, name)
    return nothing
end

function run_case(cfg, Rλ, zλ)
    name, k = cfg.name, cfg.k
    @printf("[%s]  k=%+.4f  (Ω=%.3f, γ_eff=%.4f)\n", name, k, Ω, γ_eff)
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
        force_driven!(sys, intg.t, k)

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
            @printf("  [%s] t=%6.1f/%.0f   I_L=% .4e   I_R=% .4e   elapsed=%.0fs\n",
                    name, intg.t, t_final, 0.5 * obs.Iα[1, i], -0.5 * obs.Iα[2, i],
                    time() - started)
            flush(stdout)
        end
    end

    save_outputs(name, k, obs, S_hist)
    @printf("[%s] listo en %.1f s\n", name, time() - started)
    return nothing
end


function main()
    sel = isempty(ARGS) ? RUNS : filter(c -> c.name in ARGS, RUNS)
    isempty(sel) && error("Ninguna corrida coincide con $(ARGS). Opciones: " *
                          join((c.name for c in RUNS), ", "))
    BLAS.set_num_threads(max(1, Sys.CPU_THREADS ÷ length(sel)))

    @printf("Cadena Rashba: Nx=%d (33=2·16+1)  γ=t=%.4f  γso=λ=%.4f  γ_eff=%.4f\n",
            Nx, γ, γso, γ_eff)
    @printf("Jsd=%.3f  θmax=%.2f°  Ω=%.3f  r=k/Ω ∈ %s\n",
            j_sd, rad2deg(θ_max), Ω, R_VALUES)
    @printf("grupo1(onda)=%s  grupo2(libre)=%s  grupo3(driver)=%s  grupo4(libre)=%s\n",
            GROUPS.g1, GROUPS.g2, GROUPS.g3, GROUPS.g4)
    @printf("t_on_g3=%.0f  t_on_g1=%.0f  t_rise=%.0f  t_relax=%.0f  t_final=%.0f\n",
            t_on_g3, t_on_g1, t_rise, t_relax, t_final)
    @printf("Corridas: %s\nHilos de Julia: %d (se usan %d)\n",
            join((c.name for c in sel), ", "), Threads.nthreads(), length(sel))
    Threads.nthreads() < length(sel) &&
        @printf("AVISO: hay %d corridas y solo %d hilos de Julia. Relanza con --threads=%d\n",
                length(sel), Threads.nthreads(), length(sel))
    flush(stdout)

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)

    started = time()
    Threads.@threads for i in eachindex(sel)
        run_case(sel[i], Rλ, zλ)
    end
    @printf("\nListo en %.1f s. Salidas en %s\n", time() - started, OUT)
end

main()
