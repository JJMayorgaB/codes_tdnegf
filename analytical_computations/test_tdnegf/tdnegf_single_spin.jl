#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

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
const N_BLAS = parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", string(Sys.CPU_THREADS)))

# geometria
const N_SPINS   = 1
const N_BUF     = 21
const Nx, Ny    = 2 * N_SPINS + 1 + 2 * N_BUF, 1     
const Nσ, N_orb = 2, 1

@inline elec_site(m::Int) = N_BUF + 2 * m            
const SITE_C = elec_site(1)

# Solo hay un espin y es el driver. No hay espines libres.
const GROUPS = (g2 = 1:0, g3 = 1:1, g4 = 1:0)
const DRIVEN = collect(GROUPS.g3)
const FREE   = Int[]


const N_LEAD_OUT = min(N_BUF - 1, 20)
lead_site(α::Symbol, n::Int) = α === :R ? SITE_C + 1 + n : SITE_C - 1 - n

#parametros 
const γso   = 0.1         
const γ     = 1.0         
const E_F   = 0.0          
const β     = 40.0
const N_λ1, N_λ2 = 49, 30
const j_sd  = 0.2
const Δt    = 0.1

const damping_relax = 1.0
const damping_dyn   = 0.05
const kT            = 0.0

# driving
const PREC_SIGN = +1.0              
const θ_max   = deg2rad(10.0)
const Ω       = 0.005
const T_drive = 2π / Ω             
const t_leads = 630.0               # encendido suave del acople a los leads
const t_on_g3 = 2000.0              # arranca el driver
const t_rise  = 630.0               # rampa del angulo del cono
const t_relax = t_on_g3
const t_final = 10000.0            

# Salida: TODA la dinamica, transitorio incluido, submuestreada cada OUT_STRIDE
const OUT_STRIDE = 5      # Δt=0.1 -> Δt_out = 0.5

@inline fmtnum(x::Real) = replace(string(round(Float64(x); digits = 4)), "." => "p", "-" => "m")
param_tag() = "single_gso$(fmtnum(γso))_jsd$(fmtnum(j_sd))_th$(round(Int, rad2deg(θ_max)))deg_Om$(fmtnum(Ω))_buf$(N_BUF)"
const OUT_RUN = joinpath(OUT, param_tag()); mkpath(OUT_RUN)

# driving
@inline smooth_switch(τ, ti) = τ < 0 ? 0.0 : (τ < ti ? sin((π / 2) * τ / ti)^2 : 1.0)

@inline function pumped_spin(t::Float64, t_on::Float64)
    θ = θ_max * smooth_switch(t - t_on, t_rise)
    φ = PREC_SIGN * Ω * (t - t_on)
    return SVector{3,Float64}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
end

function force_driven!(sys, t::Float64)
    for m in GROUPS.g3
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, t_on_g3)
    end
    return nothing
end

@inline lead_switch(t::Float64) = smooth_switch(t, t_leads)

function set_lead_coupling!(blocks, ξ_L0, ξ_R0, f::Float64)
    blocks[1].ξ_an .= f .* ξ_L0
    blocks[2].ξ_an .= f .* ξ_R0
    return nothing
end

# espin (Sunny)
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
    for m in FREE                                  # vacio: no hay espines libres
        site = elec_site(m)
        Sunny.set_field_at!(sys, -j_sd .* σx_i_now[site, :], (m, 1, 1, 1))
    end
    for m in DRIVEN
        Sunny.set_field_at!(sys, [0.0, 0.0, 0.0], (m, 1, 1, 1))
    end
    return nothing
end

prep_params() = (γ = γ, γso = γso, j_sd = j_sd, θmax = θ_max, Ω = Ω,
                 E_F = E_F, β = β, N_λ1 = N_λ1, N_λ2 = N_λ2, Δt = Δt,
                 prec_sign = PREC_SIGN, t_on_g3 = t_on_g3, t_rise = t_rise,
                 t_leads = t_leads, t_final = t_final)

geometry() = (N_SPINS = N_SPINS, N_BUF = N_BUF, Nx = Nx, Ny = Ny,
              Nσ = Nσ, N_orb = N_orb, site_C = SITE_C, N_lead_out = N_LEAD_OUT)

#salida formato analitico
function write_lead_csv(obs)
    t_all = obs.t
    Nt_all = length(t_all)

    idxs = 1:OUT_STRIDE:Nt_all          # TODA la dinamica, tiempo crudo

    hdr = ["t", "lead", "site", "n_up", "n_dn", "Re_rho_updn", "Im_rho_updn",
           "sx", "sy", "sz", "n_tot"]
    nrow = length(idxs) * 2 * (N_LEAD_OUT + 1)
    data = Matrix{Any}(undef, nrow, length(hdr))

    r = 0
    for α in (:L, :R), n in 0:N_LEAD_OUT
        site = lead_site(α, n)
        for i in idxs
            sx = obs.σx_i[site, 1, i]
            sy = obs.σx_i[site, 2, i]
            sz = obs.σx_i[site, 3, i]
            nt = obs.n_i[site, i]
            r += 1
            data[r, :] = Any[t_all[i], String(α), n,
                             0.5 * (nt + sz), 0.5 * (nt - sz),
                             0.5 * sx, -0.5 * sy,
                             sx, sy, sz, nt]
        end
    end
    path = joinpath(OUT_RUN, "tdnegf_rho_t.csv")
    writedlm(path, vcat(permutedims(hdr), data), ",")
    @printf("  -> %s\n     t=[%.1f, %.1f]  Δt_out=%.2f  %d puntos x %d sitios x 2 leads = %d filas\n",
            path, t_all[first(idxs)], t_all[last(idxs)], OUT_STRIDE * Δt,
            length(idxs), N_LEAD_OUT + 1, nrow)
    return nothing
end

"Igual que arriba pero para el SITIO MANEJADO, comparable con floquet_rho_t.csv."
function write_driven_csv(obs)
    t_all = obs.t
    idxs = 1:OUT_STRIDE:length(t_all)

    hdr = ["t", "n_up", "n_dn", "Re_rho_updn", "Im_rho_updn", "sx", "sy", "sz", "n_tot"]
    data = Matrix{Any}(undef, length(idxs), length(hdr))
    for (r, i) in enumerate(idxs)
        sx = obs.σx_i[SITE_C, 1, i]
        sy = obs.σx_i[SITE_C, 2, i]
        sz = obs.σx_i[SITE_C, 3, i]
        nt = obs.n_i[SITE_C, i]
        data[r, :] = Any[t_all[i], 0.5*(nt+sz), 0.5*(nt-sz),
                         0.5*sx, -0.5*sy, sx, sy, sz, nt]
    end
    path = joinpath(OUT_RUN, "tdnegf_driven_rho_t.csv")
    writedlm(path, vcat(permutedims(hdr), data), ",")
    println("  -> ", path)
    return nothing
end

function write_params_label()
    open(joinpath(OUT_RUN, "params.txt"), "w") do io
        println(io, "TDNEGF: UN SOLO ESPIN CON DRIVING IMPUESTO")
        println(io, "contraparte de floquet_gf.jl + inbedding_leads.jl")
        println(io, "param_tag = ", param_tag())
        println(io, "")
        println(io, "Nx = ", Nx, "   N_SPINS = ", N_SPINS, "   N_BUF = ", N_BUF)
        println(io, "sitio del espin (manejado) = ", SITE_C)
        println(io, "leads enganchados en los sitios 1 y ", Nx)
        println(io, "mapeo: sitio TDNEGF ", SITE_C+1, "+n  <-> lead R analitico n")
        println(io, "       sitio TDNEGF ", SITE_C-1, "-n  <-> lead L analitico n")
        println(io, "n = 0..", N_LEAD_OUT)
        println(io, "")
        println(io, "γ (= t analitico)     = ", γ)
        println(io, "γso (= λ analitico)   = ", γso)
        println(io, "j_sd                  = ", j_sd)
        println(io, "θ_max                 = ", rad2deg(θ_max), " deg")
        println(io, "Ω                     = ", Ω, "   (T = ", T_drive, ")")
        println(io, "E_F (= μ)             = ", E_F)
        println(io, "β                     = ", β)
        println(io, "N_λ1, N_λ2            = ", N_λ1, ", ", N_λ2,
                    "   (expansion en polos de los leads; el analitico usa Σʳ exacta)")
        println(io, "Δt                    = ", Δt)
        println(io, "PREC_SIGN             = ", PREC_SIGN,
                    "   (+1 = mismo sentido de precesion que el analitico)")
        println(io, "")
        println(io, "t_leads = ", t_leads, "   t_on_g3 = ", t_on_g3,
                    "   t_rise = ", t_rise, "   t_final = ", t_final)
        println(io, "periodos utiles tras la rampa = ",
                    round((t_final - t_on_g3 - t_rise) / T_drive, digits = 2))
        println(io, "salida = TODA la dinamica, stride ", OUT_STRIDE,
                    " (Δt_out = ", OUT_STRIDE * Δt, "), tiempo crudo sin trasladar")
    end
    println("params.txt escrito")
    return nothing
end

#corrida
function run_sim()
    p_model = ModelParamsTDNEGF(Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb, Nα = 2, N_λ1 = N_λ1, N_λ2 = N_λ2)
    H0 = build_H_ab(; Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb, γ = γ, γso = complex(γso, 0.0))

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    Σᴸ = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)
    Σᴳ = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)
    χ  = build_χ_nλ(zλ,      Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)

    ξ_L = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = 1,  y_coup = 1:Ny)
    ξ_R = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = Nx, y_coup = 1:Ny)

    blocks = [SelfEnergyBlock(:left,  p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_L),  SelfEnergyBlock(:right, p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_R)]

    ξ_L0, ξ_R0 = copy(ξ_L), copy(ξ_R)
    set_lead_coupling!(blocks, ξ_L0, ξ_R0, lead_switch(0.0))

    p_model.H0_ab .= H0
    p_model.H_ab  .= H0
    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, ComplexF64[0.0, 0.0], p_model)

    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)
    @printf("Vector de estado: |u| = %d  (%.2f MB)\n", length(u0), 16 * length(u0) / 1e6)

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
        set_lead_coupling!(blocks, ξ_L0, ξ_R0, lead_switch(intg.t))
        Sunny.step!(sys, llg)
        force_driven!(sys, intg.t)

        dv = pointer_blocks(intg.u, p_blocks.dims_ρ_ab, p_blocks.aux_layout)

        obs.t[i] = intg.t
        obs_n_i!(dv, p_model, obs)
        obs_σ_i!(dv, p_model, obs)
        obs_Ixα!(dv, p_blocks, obs)

        for m in 1:N_SPINS, c in 1:3
            S_hist[c, m, i] = sys.dipoles[m, 1, 1, 1][c]
        end

        update_H_s_free!(sys, obs.σx_i[:, :, i])
        update_H_e!(p_model, site_ranges, full_dipoles(sys), j_sd)

        if i % 1000 == 0
            etapa = intg.t < t_leads ? "encendiendo leads" :
                    (intg.t < t_on_g3 ? "relajacion" :
                     (intg.t < t_on_g3 + t_rise ? "rampa del cono" : "driver estacionario"))
            @printf("  t=%7.1f/%.0f  [%s]  σ_C=(% .3e,% .3e,% .3e)  elapsed=%.0fs\n",
                    intg.t, t_final, etapa,
                    obs.σx_i[SITE_C, 1, i], obs.σx_i[SITE_C, 2, i],
                    obs.σx_i[SITE_C, 3, i], time() - started)
            flush(stdout)
        end
    end
    @printf("\nDinamica lista en %.1f s\n", time() - started)

    jldsave(joinpath(OUT_RUN, "fields.jld2");
            t = obs.t, s_i = S_hist, sigma_i = obs.σx_i, n_i = obs.n_i,
            I_alpha = obs.Iα, I_alpha_x = obs.Iαx,
            geometry = geometry(), params = prep_params())
    println("Observables: fields.jld2")
    write_lead_csv(obs)
    write_driven_csv(obs)
    return nothing
end

function main()
    BLAS.set_num_threads(N_BLAS)
    println("="^72)
    println("TDNEGF -- UN SOLO ESPIN CON DRIVING IMPUESTO")
    println("contraparte numerica de floquet_gf.jl + inbedding_leads.jl")
    println("="^72)
    @printf("Nx=%d  espin en el sitio %d  N_BUF=%d  γ=%.4f  γso=%.4f\n",
            Nx, SITE_C, N_BUF, γ, γso)
    @printf("j_sd=%.3f  θmax=%.2f°  Ω=%.4f  (T=%.1f)  E_F=%.2f  β=%.1f\n",
            j_sd, rad2deg(θ_max), Ω, T_drive, E_F, β)
    @printf("sitios de lead observables: n=0..%d por lado\n", N_LEAD_OUT)
    @printf("  lead R -> sitios TDNEGF %d..%d\n", SITE_C+1, SITE_C+1+N_LEAD_OUT)
    @printf("  lead L -> sitios TDNEGF %d..%d\n", SITE_C-1, SITE_C-1-N_LEAD_OUT)
    @printf("t_leads=%.0f  t_on_g3=%.0f  t_rise=%.0f  t_final=%.0f  (%.1f periodos utiles)\n",
            t_leads, t_on_g3, t_rise, t_final,
            (t_final - t_on_g3 - t_rise) / T_drive)
    @printf("Salidas en %s\n", OUT_RUN)
    println("="^72)
    flush(stdout)

    write_params_label()
    run_sim()
    println("\nListo. Para animar, con el MISMO script del analitico:")
    println("  python ../animate_lead_spins.py \\")
    println("      --csv ",    joinpath(OUT_RUN, "tdnegf_rho_t.csv"), " \\")
    println("      --outdir ", OUT_RUN, " \\")
    println("      --Omega ",  Ω)
    println("  (--outdir hace falta: si no, sobrescribe la animacion analitica)")
end

main()
