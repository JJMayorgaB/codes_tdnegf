#!/usr/bin/env julia
#=
  oscillators.jl  --  DOS MODOS

    prep    (por defecto)  arranca en frio, lleva la cadena a su estado
                           estacionario con solo el driver g3 encendido, y
                           guarda un CHECKPOINT COMPLETO al final.

    resume                 carga ese checkpoint y continua la dinamica desde
                           t_ckpt encendiendo g1 (la onda viajera), una rama
                           por cada (r, signo de k).

  Uso:
    julia --project=. oscillators_tdnegf/oscillators.jl
    julia --project=. oscillators_tdnegf/oscillators.jl resume
    julia --project=. oscillators_tdnegf/oscillators.jl resume r1p0_kpos r1p0_kneg

  El checkpoint guarda el vector de estado del integrador (ρ_ab + auxiliares de
  los leads) y la configuracion de espines. Los observables por si solos NO
  alcanzan para reanudar: la memoria de los leads no se puede reconstruir.

  Salidas:
    prep     output/steady_state_<param_tag>/
    resume   output/pumping_<param_tag>/
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

# Presupuesto de hilos de BLAS para ESTE proceso. Sin la variable de entorno se
# toman todos los cores, que es lo correcto en una maquina dedicada; en un
# cluster compartido hay que acotarlo para no acaparar:
#   OPENBLAS_NUM_THREADS=32 julia --project=. ...
const N_BLAS = parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", string(Sys.CPU_THREADS)))

# Geometría
const N_SPINS   = 26
const Nx, Ny    = 2 * N_SPINS + 1, 1        
const Nσ, N_orb = 2, 1

# sitio electrónico del espín m: 2,4,...,52
@inline elec_site(m::Int) = 2 * m

const GROUPS = (
    g1 = 1:5,      # onda viajer
    g2 = 6:15,     # libre (LLG)
    g3 = 16:16,    # precesión uniforme,  arranca en t_on_g3
    g4 = 17:26,    # libre (LLG)
)
const DRIVEN = vcat(collect(GROUPS.g1), collect(GROUPS.g3))
const FREE   = vcat(collect(GROUPS.g2), collect(GROUPS.g4))

# Parametros físicos
const γso   = 0.1
const γ     = sqrt(1.0 - γso^2)
const γ_eff = sqrt(γ^2 + γso^2)

const E_F = 0.0                    # sin bias
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
const T_drive = 2π / Ω             
const t_rise  = 630.0
const t_leads = 630.0               # encendido suave del acople a los leads, desde t=0
const t_relax = 10000.0             # relajacion de leads/electrones
const t_on_g3 = 10000.0             # el driver arranca donde termina la relajacion
const t_final = 20000.0            # fin de la preparacion 
const t_pump  = 20000.0            # duracion del pump

const R_VALUES = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)

# Etiqueta de parametros fisicos
@inline fmtnum(x::Real) = replace(string(round(Float64(x); digits = 4)), "." => "p", "-" => "m")
param_tag() = "gso$(fmtnum(γso))_jsd$(fmtnum(j_sd))_th$(round(Int, rad2deg(θ_max)))deg_Om$(fmtnum(Ω))"

const OUT_PREP = joinpath(OUT, "steady_state_" * param_tag())
const OUT_PUMP = joinpath(OUT, "pumping_" * param_tag())
const CKPT     = joinpath(OUT_PREP, "checkpoint_t$(round(Int, t_final)).jld2")

const RUNS = Tuple(
    (name = "r$(replace(string(r), "." => "p"))_k$(s > 0 ? "pos" : "neg")",
     k = s * r * Ω / γ_eff)
    for r in R_VALUES for s in (+1, -1)
)

# Configuracion de una corrida 
struct RunCfg
    name::String
    k::Float64
    t_start::Float64
    t_stop::Float64
    t_on_g1::Float64      # = t_stop en prep (nunca prende); = t_start en resume
    outdir::String
    save_ckpt::Bool
end

is_prep(cfg) = cfg.name == "prep"
trace_file(cfg)  = is_prep(cfg) ? "prep_trace.csv"   : "oscillators_trace_$(cfg.name).csv"
fields_file(cfg) = is_prep(cfg) ? "prep_fields.jld2" : "oscillators_fields_$(cfg.name).jld2"

# Driving cinemático: M(t)
@inline smooth_switch(τ, ti) = τ < 0 ? 0.0 : (τ < ti ? sin((π / 2) * τ / ti)^2 : 1.0)

@inline function pumped_spin(t::Float64, x::Float64, k::Float64, t_on::Float64)
    θ = θ_max * smooth_switch(t - t_on, t_rise)
    φ = k * x - Ω * (t - t_on)
    return SVector{3,Float64}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
end

function force_driven!(sys, t::Float64, k::Float64, t_on_g1::Float64)
    for m in GROUPS.g3
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, 0.0, 0.0, t_on_g3)
    end
    for m in GROUPS.g1
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, Float64(m), k, t_on_g1)
    end
    return nothing
end

# Encendido suave del acople a los leads
#
#   ξ_α(t) = f(t) · ξ_α,   f: 0 -> 1 en t_leads (misma rampa sin² del driver)
#
# La hibridacion entra como Γ_α ∝ ξ_α², asi que Γ crece como f². Conectar los
# leads de golpe en t=0 proyecta el estado inicial sobre todos los niveles del
# dispositivo a la vez; los que estan debilmente acoplados quedan resonando sin
# poder disipar al continuo. Con la rampa el acople crece despacio frente a la
# escala de nivel del dispositivo y esas resonancias no se excitan.
#
# La EOM sigue siendo exacta con ξ dependiente del tiempo: en esta jerarquia ξ
# solo aparece a tiempos iguales -- el termino de borde t̄=t que alimenta a Ψ y
# Ω, y el factor externo de Π = Ψ ξᵀ. La memoria la cargan Ψ y Ω, no ξ, asi que
# no aparecen terminos ∂_t ξ.
@inline lead_switch(t::Float64) = smooth_switch(t, t_leads)

function set_lead_coupling!(blocks, ξ_L0, ξ_R0, f::Float64)
    blocks[1].ξ_an .= f .* ξ_L0
    blocks[2].ξ_an .= f .* ξ_R0
    return nothing
end

# Sistema de espines (Sunny)
function init_spins(dipoles0 = nothing)
    latvecs   = lattice_vectors(1.0, 1.0 * (1 + 1e-3), 4.0, 90, 90, 90)
    positions = [[0.5, 0.5, 0.0]]
    cryst     = Crystal(latvecs, positions)
    moments   = [1 => Moment(s = 1.0, g = 1.0)]
    sys = System(cryst, moments, :dipole; dims = (N_SPINS, 1, 1))
    for m in 1:N_SPINS
        v = dipoles0 === nothing ? Sunny.SVector(0.0, 0.0, 1.0) :
            Sunny.SVector(dipoles0[1, m], dipoles0[2, m], dipoles0[3, m])
        sys.dipoles[m, 1, 1, 1] = v
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

# Metadatos
geometry() = (N_SPINS = N_SPINS, Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb)

run_params(cfg) = (γ = γ, γso = γso, γ_eff = γ_eff, j_sd = j_sd, θmax = θ_max, Ω = Ω,
                   k = cfg.k, E_F = E_F, β = β, N_λ1 = N_λ1, N_λ2 = N_λ2, Δt = Δt,
                   t_on_g3 = t_on_g3, t_on_g1 = cfg.t_on_g1, t_rise = t_rise,
                   t_leads = t_leads,
                   t_relax = t_relax, t_start = cfg.t_start, t_stop = cfg.t_stop,
                   damping_relax = damping_relax, damping_dyn = damping_dyn, kT = kT)

# Checkpoint
function save_checkpoint(cfg, intg, sys)
    S_final = Array{Float64}(undef, 3, N_SPINS)
    for m in 1:N_SPINS, c in 1:3
        S_final[c, m] = sys.dipoles[m, 1, 1, 1][c]
    end
    jldsave(CKPT;
            u = collect(intg.u), t = intg.t, dipoles = S_final,
            len_u = length(intg.u),
            geometry = geometry(), params = run_params(cfg),
            groups = (g1 = collect(GROUPS.g1), g2 = collect(GROUPS.g2),
                      g3 = collect(GROUPS.g3), g4 = collect(GROUPS.g4)))
    @printf("Checkpoint escrito: %s  (t=%.1f, |u|=%d, %.2f MB)\n",
            basename(CKPT), intg.t, length(intg.u), 16 * length(intg.u) / 1e6)
    return nothing
end

function load_checkpoint()
    isfile(CKPT) || error("No existe el checkpoint:\n  $CKPT\nCorre primero el modo prep.")
    ck = jldopen(CKPT, "r") do f
        (u = f["u"], t = f["t"], dipoles = f["dipoles"],
         len_u = f["len_u"], geometry = f["geometry"], params = f["params"])
    end

    # Validacion: cargar un checkpoint de otra configuracion daria basura silenciosa
    g, p = ck.geometry, ck.params
    for (nom, esperado, guardado) in (("N_SPINS", N_SPINS, g.N_SPINS), ("Nx", Nx, g.Nx),
                                      ("Ny", Ny, g.Ny), ("Nσ", Nσ, g.Nσ),
                                      ("N_orb", N_orb, g.N_orb),
                                      ("γso", γso, p.γso), ("j_sd", j_sd, p.j_sd),
                                      ("Ω", Ω, p.Ω), ("β", β, p.β),
                                      ("N_λ1", N_λ1, p.N_λ1), ("N_λ2", N_λ2, p.N_λ2))
        esperado == guardado || error("El checkpoint no coincide con este script: " *
                                      "$nom = $guardado en el checkpoint, $esperado aqui.")
    end
    @printf("Checkpoint cargado: t=%.1f  |u|=%d  (%s)\n",
            ck.t, ck.len_u, basename(CKPT))
    return ck
end

# Observables
function save_outputs(cfg, obs, S_hist)
    t  = obs.t
    Nt = length(t)

    jldsave(joinpath(cfg.outdir, fields_file(cfg));
            t = t, s_i = S_hist,
            sigma_i = obs.σx_i, sigma_eq = obs.σx_i_eq, n_i = obs.n_i,
            I_alpha = obs.Iα, I_alpha_x = obs.Iαx,
            groups = (g1 = collect(GROUPS.g1), g2 = collect(GROUPS.g2),
                      g3 = collect(GROUPS.g3), g4 = collect(GROUPS.g4)),
            driven = DRIVEN, free = FREE,
            elec_sites = [elec_site(m) for m in 1:N_SPINS],
            geometry = geometry(), params = run_params(cfg))

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
    writedlm(joinpath(cfg.outdir, trace_file(cfg)), vcat(permutedims(header), data), ",")
    @printf("  [%s] -> %s  %s\n", cfg.name, fields_file(cfg), trace_file(cfg))
    return nothing
end

function write_params_label(outdir, modo, sel; t_on_g1 = nothing)
    open(joinpath(outdir, "params.txt"), "w") do io
        println(io, modo == :prep ? "PREPARACION DEL STEADY STATE  (g1 apagado)" :
                                    "PUMPING: ramas con g1 encendido")
        println(io, "param_tag = ", param_tag())
        if modo == :prep
            println(io, "checkpoint = ", basename(CKPT))
        else
            println(io, "checkpoint de origen = ", CKPT)
            # t_on_g1 lo leen los scripts de python para las lineas de hito y la
            # ventana de la FFT; en prep no se escribe porque g1 nunca se enciende.
            println(io, "t_on_g1 = ", t_on_g1)
            println(io, "t_pump = ", t_pump, "   (", round(t_pump/T_drive, digits = 2), " periodos)")
            println(io, "corridas = ", join((c.name for c in sel), ", "))
            println(io, "R_VALUES = ", R_VALUES)
        end
        println(io, "")
        println(io, "N_SPINS = ", N_SPINS, "   Nx = ", Nx, "   Ny = ", Ny,
                    "   Nσ = ", Nσ, "   N_orb = ", N_orb)
        println(io, "g1 = ", GROUPS.g1, modo == :prep ? "  (onda viajera, APAGADA)" :
                                                        "  (onda viajera, ENCENDIDA)")
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
        println(io, "t_leads = ", t_leads, "   (encendido suave de los leads, desde t=0)")
        println(io, "t_relax = ", t_relax)
        println(io, "t_on_g3 = ", t_on_g3)
        println(io, "t_rise  = ", t_rise)
        println(io, "t_final = ", t_final)
    end
    println("params.txt escrito en ", outdir)
    return nothing
end

# Nucleo compartido por los dos modos
function run_case(cfg::RunCfg, Rλ, zλ, u0_init, dipoles0)
    @printf("[%s]  t: %.0f -> %.0f   k=%+.5f   t_on_g1=%.0f\n",
            cfg.name, cfg.t_start, cfg.t_stop, cfg.k, cfg.t_on_g1)
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

    # SelfEnergyBlock guarda la referencia al arreglo, no una copia
    # (blocks[1].ξ_an === ξ_L), asi que hay que conservar aparte los valores sin
    # escalar. Se fija el acople ANTES de init: el integrador evalua el RHS en
    # t_start y debe ver ξ(t_start). En resume t_start ≫ t_leads -> f=1, o sea
    # el acople completo con el que se guardo el checkpoint.
    ξ_L0, ξ_R0 = copy(ξ_L), copy(ξ_R)
    set_lead_coupling!(blocks, ξ_L0, ξ_R0, lead_switch(cfg.t_start))

    p_model.H0_ab .= H0
    p_model.H_ab  .= H0
    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, ComplexF64[0.0, 0.0], p_model)

    len_u = p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size
    u0 = if u0_init === nothing
        zeros(ComplexF64, len_u)
    else
        length(u0_init) == len_u ||
            error("El checkpoint tiene |u|=$(length(u0_init)) y aqui se esperan $len_u.")
        copy(u0_init)                     # cada rama necesita su propia copia
    end

    sys = init_spins(dipoles0)
    site_ranges = [get_sub(i, p_model.N_loc) for i in 1:p_model.N_sites]

    # Al reanudar, H_ab debe corresponder a los espines del checkpoint desde el
    # primer paso; si no, la primera evaluacion del RHS usaria H sin acoplar.
    dipoles0 === nothing || update_H_e!(p_model, site_ranges, full_dipoles(sys), j_sd)

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (cfg.t_start, cfg.t_stop), p_blocks)
    intg = init(prob, Vern7(); dt = Δt, save_everystep = false, adaptive = true, dense = false)

    llg_relax = Langevin(Δt; damping = damping_relax, kT = kT)
    llg_dyn   = Langevin(Δt; damping = damping_dyn,   kT = kT)

    N_steps = Int(round((cfg.t_stop - cfg.t_start) / Δt))
    obs = ObservablesTDNEGF(p_model; N_tmax = N_steps, N_leads = 2)
    S_hist = Array{Float64}(undef, 3, N_SPINS, N_steps)

    started = time()
    for i in 1:N_steps
        obs.idx = i
        llg = intg.t < t_relax ? llg_relax : llg_dyn

        DifferentialEquations.step!(intg, Δt, true)
        # ξ(t_i): exacto para los observables de este paso y valor congelado con
        # el que arranca el paso siguiente, igual que se trata H_ab.
        set_lead_coupling!(blocks, ξ_L0, ξ_R0, lead_switch(intg.t))
        Sunny.step!(sys, llg)
        force_driven!(sys, intg.t, cfg.k, cfg.t_on_g1)

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
            etapa = intg.t < t_leads ? "encendiendo leads" :
                    (intg.t < t_on_g3 ? "relajacion" :
                    (intg.t < cfg.t_on_g1 ? "driver g3" : "g1+g3"))
            @printf("  [%s] t=%7.1f/%.0f  [%s]  I_L=% .3e  I_R=% .3e  elapsed=%.0fs\n",
                    cfg.name, intg.t, cfg.t_stop, etapa,
                    0.5 * obs.Iα[1, i], -0.5 * obs.Iα[2, i], time() - started)
            flush(stdout)
        end
    end
    @printf("[%s] dinamica lista en %.1f s\n", cfg.name, time() - started)

    cfg.save_ckpt && save_checkpoint(cfg, intg, sys)
    save_outputs(cfg, obs, S_hist)
    return nothing
end

function run_prep()
    mkpath(OUT_PREP)
    println("="^70)
    println("MODO prep -- PREPARACION DEL STEADY STATE  (g1 apagado)")
    println("="^70)
    @printf("Cadena Rashba: Nx=%d (=2·%d+1)  γ=%.4f  γso=%.4f  γ_eff=%.4f\n",
            Nx, N_SPINS, γ, γso, γ_eff)
    @printf("Jsd=%.3f  θmax=%.2f°  Ω=%.4f  (periodo T=%.1f)\n",
            j_sd, rad2deg(θ_max), Ω, T_drive)
    @printf("g1(onda,OFF)=%s  g2(libre)=%s  g3(driver)=%s  g4(libre)=%s\n",
            GROUPS.g1, GROUPS.g2, GROUPS.g3, GROUPS.g4)
    @printf("t_leads=%.0f (encendido suave de leads)  t_relax=%.0f  t_on_g3=%.0f  t_rise=%.0f  t_final=%.0f  (%.1f periodos de driver)\n",
            t_leads, t_relax, t_on_g3, t_rise, t_final, (t_final - t_on_g3) / T_drive)
    @printf("Salidas en %s\n", OUT_PREP)
    println("="^70)
    flush(stdout)

    BLAS.set_num_threads(N_BLAS)
    write_params_label(OUT_PREP, :prep, ())

    # t_on_g1 = t_stop garantiza por construccion que g1 nunca se enciende
    cfg = RunCfg("prep", 0.0, 0.0, t_final, t_final, OUT_PREP, true)
    run_case(cfg, load_poles_square(N_λ1, N_λ2)..., nothing, nothing)

    @printf("\nListo. Checkpoint para el modo resume:\n  %s\n", CKPT)
    return nothing
end

function run_resume(names)
    sel = isempty(names) ? RUNS : filter(c -> c.name in names, RUNS)
    isempty(sel) && error("Ninguna corrida coincide con $(names). Opciones: " *
                          join((c.name for c in RUNS), ", "))
    mkpath(OUT_PUMP)

    ck = load_checkpoint()
    t0 = ck.t

    println("="^70)
    println("MODO resume -- PUMPING desde el steady state (g1 encendido)")
    println("="^70)
    @printf("Checkpoint: t=%.1f   ->   ramas hasta t=%.1f  (%.1f periodos)\n",
            t0, t0 + t_pump, t_pump / T_drive)
    @printf("Corridas (%d): %s\n", length(sel), join((c.name for c in sel), ", "))
    @printf("Hilos de Julia: %d\n", Threads.nthreads())
    Threads.nthreads() < length(sel) &&
        @printf("AVISO: %d corridas y solo %d hilos. Relanza con --threads=%d\n",
                length(sel), Threads.nthreads(), length(sel))
    @printf("Salidas en %s\n", OUT_PUMP)
    println("="^70)
    flush(stdout)

    BLAS.set_num_threads(max(1, N_BLAS ÷ length(sel)))
    write_params_label(OUT_PUMP, :resume, sel; t_on_g1 = t0)

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)

    started = time()
    Threads.@threads for i in eachindex(sel)
        c = sel[i]
        # g1 arranca justo en el checkpoint: el eje temporal continua sin salto,
        # asi la fase de precesion de g3 empalma con la preparacion.
        cfg = RunCfg(c.name, c.k, t0, t0 + t_pump, t0, OUT_PUMP, false)
        run_case(cfg, Rλ, zλ, ck.u, ck.dipoles)
    end
    @printf("\nListo en %.1f s. Salidas en %s\n", time() - started, OUT_PUMP)
    return nothing
end

function main()
    modo = isempty(ARGS) ? "prep" : lowercase(ARGS[1])
    if modo == "prep"
        length(ARGS) > 1 && error("El modo prep no acepta argumentos extra: $(ARGS[2:end])")
        run_prep()
    elseif modo == "resume"
        run_resume(ARGS[2:end])
    else
        error("Modo desconocido: '$(ARGS[1])'. Usa 'prep' (por defecto) o 'resume [nombres...]'.")
    end
end

main()
