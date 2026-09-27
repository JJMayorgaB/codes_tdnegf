#!/usr/bin/env julia
#=
common.jl -- nucleo compartido de los setups de dos osciladores + driver.

Cadena Rashba 1D de Nx = 2 N_BUF + 3 sitios electronicos, leads en los sitios 1
y Nx, y TRES LMMs en los sitios contiguos N_BUF+1, N_BUF+2, N_BUF+3 (d = 1):

    setup1 :  [buf] F  D  F [buf]     (driver en el centro, un libre a cada lado)
    setup2 :  [buf] D  F  F [buf]     (driver a la izquierda, dos libres a su derecha)

D = LMM con precesion prescrita (cono θ_max, frecuencia Ω) alrededor del eje
de precesion (:z o :y); F = LMM libre que evoluciona con la LLG de Sunny bajo el
campo -j_sd ⟨σ̂_i⟩(t). Sin damping intrinseco en la fase dinamica (damping_dyn = 0):
todo el amortiguamiento de los F sale de su acople con los electrones.

Parametros fisicos y protocolo: los de analytical_computations/test_tdnegf/
tdnegf_single_spin.jl (comparables con el Floquet analitico y el kernel).

Uso (desde cada carpeta de precesion):
    include(joinpath(@__DIR__, "..", "common.jl"))
    run_setup(:setup1, :z; outroot = joinpath(@__DIR__, "output"))
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))              # entorno de codes_tdnegf

using TDNEGF
using DifferentialEquations
using Sunny
using LinearAlgebra
using LinearAlgebra: BLAS
using StaticArrays
using Printf
using DelimitedFiles
using JLD2

const N_BLAS = parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", string(Sys.CPU_THREADS)))

# parametros fisicos (tdnegf_single_spin.jl)
const γso   = parse(Float64, get(ENV, "GSO", "0.1"))        # override: GSO=0.5 julia setup1.jl
const γ     = sqrt(1 - γso^2)        # |T̂| = sqrt(γ² + γso²) = 1: banda del lead [-2, 2]
const E_F   = 0.0
const β     = 40.0
const N_λ1, N_λ2 = 49, 30
const j_sd  = 0.2
const Δt    = 0.1
const Nσ, N_orb, Ny = 2, 1, 1

const damping_relax = 1.0            # solo para preparar el estado (t < t_relax)
const kT            = 0.0

# driving
const PREC_SIGN = +1.0
const θ_max   = deg2rad(parse(Float64, get(ENV, "THETA", "10")))   # grados
const Ω       = parse(Float64, get(ENV, "OMEGA", "0.005"))
const T_drive = 2π / Ω
const t_leads = 630.0                # encendido suave del acople a los leads
const t_on    = 5 * T_drive          # arranca el driver
const t_rise  = 630.0                # rampa del angulo del cono
const t_relax = t_on
const t_final = 20 * T_drive

# salida: CSV cada OUT_STRIDE pasos (Δt_out = 2.0, ~628 puntos por periodo)
const OUT_STRIDE = 20

# geometria de cada setup: rol de los LMMs m = 1, 2, 3 (de izquierda a derecha)
const LAYOUTS = Dict(:setup1 => (:F, :D, :F),
                     :setup2 => (:D, :F, :F))

struct Cfg
    setup::Symbol
    axis::Symbol                     # eje de precesion del driver (:z o :y)
    roles::NTuple{3,Symbol}
    N_BUF::Int
    Nx::Int
    damping_dyn::Float64
    outdir::String
end

spin_site(c::Cfg, m::Int) = c.N_BUF + m                     # sitio electronico del LMM m
driven_spins(c::Cfg) = [m for m in 1:3 if c.roles[m] === :D]
free_spins(c::Cfg)   = [m for m in 1:3 if c.roles[m] === :F]
site_role(c::Cfg, s::Int) = (m = s - c.N_BUF; 1 <= m <= 3 ? String(c.roles[m]) :
                                               (s <= c.N_BUF ? "bufL" : "bufR"))

@inline fmtnum(x::Real) = replace(string(round(Float64(x); digits = 4)), "." => "p", "-" => "m")
param_tag(c::Cfg) = "gso$(fmtnum(γso))_jsd$(fmtnum(j_sd))_th$(round(Int, rad2deg(θ_max)))deg" *
                    "_Om$(fmtnum(Ω))_buf$(c.N_BUF)_adyn$(fmtnum(c.damping_dyn))"

# eje de precesion y cono prescrito del driver
axis_vec(ax::Symbol) = ax === :z ? SVector(0.0, 0.0, 1.0) :
                       ax === :y ? SVector(0.0, 1.0, 0.0) : error("eje $ax no soportado")

@inline smooth_switch(τ, ti) = τ < 0 ? 0.0 : (τ < ti ? sin((π / 2) * τ / ti)^2 : 1.0)

"""
Cono de angulo θ(t) alrededor del eje, en sentido dextrogiro para PREC_SIGN = +1:
  :z -> (sinθ cosφ, sinθ sinφ, cosθ)      (x -> y)
  :y -> (sinθ sinφ, cosθ, sinθ cosφ)      (z -> x)
"""
@inline function pumped_spin(t::Float64, ax::Symbol)
    θ = θ_max * smooth_switch(t - t_on, t_rise)
    φ = PREC_SIGN * Ω * (t - t_on)
    s, c = sin(θ), cos(θ)
    return ax === :z ? SVector(s * cos(φ), s * sin(φ), c) : SVector(s * sin(φ), c, s * cos(φ))
end

function force_driven!(sys, c::Cfg, t::Float64)
    for m in driven_spins(c)
        sys.dipoles[m, 1, 1, 1] = pumped_spin(t, c.axis)
    end
    return nothing
end

@inline lead_switch(t::Float64) = smooth_switch(t, t_leads)

function set_lead_coupling!(blocks, ξ_L0, ξ_R0, f::Float64)
    blocks[1].ξ_an .= f .* ξ_L0
    blocks[2].ξ_an .= f .* ξ_R0
    return nothing
end

# espines (Sunny): 3 dipolos sin interacciones propias, todos iniciados sobre el eje
function init_spins(c::Cfg)
    latvecs   = lattice_vectors(1.0, 1.0 * (1 + 1e-3), 4.0, 90, 90, 90)
    cryst     = Crystal(latvecs, [[0.5, 0.5, 0.0]])
    moments   = [1 => Moment(s = 1.0, g = 1.0)]
    sys = System(cryst, moments, :dipole; dims = (3, 1, 1))
    e = axis_vec(c.axis)
    for m in 1:3
        sys.dipoles[m, 1, 1, 1] = Sunny.SVector(e[1], e[2], e[3])
    end
    return sys
end

function full_dipoles(sys, c::Cfg)
    S = Matrix{SVector{3,Float64}}(undef, c.Nx, Ny)
    fill!(S, SVector{3,Float64}(0.0, 0.0, 0.0))
    for m in 1:3
        S[spin_site(c, m), 1] = SVector{3,Float64}(sys.dipoles[m, 1, 1, 1])
    end
    return S
end

function update_H_s_free!(sys, c::Cfg, σx_i_now)
    for m in free_spins(c)
        Sunny.set_field_at!(sys, -j_sd .* σx_i_now[spin_site(c, m), :], (m, 1, 1, 1))
    end
    for m in driven_spins(c)
        Sunny.set_field_at!(sys, [0.0, 0.0, 0.0], (m, 1, 1, 1))
    end
    return nothing
end

prep_params(c::Cfg) = (γ = γ, γso = γso, j_sd = j_sd, θmax = θ_max, Ω = Ω, E_F = E_F, β = β,
                       N_λ1 = N_λ1, N_λ2 = N_λ2, Δt = Δt, prec_sign = PREC_SIGN,
                       t_on = t_on, t_rise = t_rise, t_leads = t_leads, t_relax = t_relax,
                       t_final = t_final, damping_relax = damping_relax,
                       damping_dyn = c.damping_dyn, kT = kT, out_stride = OUT_STRIDE)

geometry(c::Cfg) = (setup = String(c.setup), axis = String(c.axis), N_BUF = c.N_BUF, Nx = c.Nx,
                    Ny = Ny, Nσ = Nσ, N_orb = N_orb,
                    roles = [String(r) for r in c.roles],
                    spin_sites = [spin_site(c, m) for m in 1:3])

# salidas de texto
const ENT = ((1, 1), (1, 2), (2, 1), (2, 2))            # orden de espin (↑,↓): uu, ud, du, dd

function write_geometry_csv(c::Cfg)
    open(joinpath(c.outdir, "geometry.csv"), "w") do io
        println(io, "m,site,role")
        for m in 1:3
            println(io, m, ",", spin_site(c, m), ",", c.roles[m])
        end
    end
end

function write_params_txt(c::Cfg)
    open(joinpath(c.outdir, "params.txt"), "w") do io
        println(io, "DOS OSCILADORES + DRIVER   setup = ", c.setup, "   eje de precesion = ", c.axis)
        println(io, "param_tag = ", param_tag(c))
        println(io, "")
        println(io, "Nx = ", c.Nx, "   N_BUF = ", c.N_BUF, " (sitios desnudos a cada lado)")
        println(io, "leads en los sitios 1 y ", c.Nx)
        for m in 1:3
            println(io, "LMM m=", m, " -> sitio ", spin_site(c, m), "  (", c.roles[m], ")")
        end
        println(io, "")
        println(io, "γ = ", γ, "   γso = ", γso, "   j_sd = ", j_sd)
        println(io, "θ_max = ", rad2deg(θ_max), " deg   Ω = ", Ω, "   T = ", T_drive,
                    "   PREC_SIGN = ", PREC_SIGN)
        println(io, "E_F = ", E_F, "   β = ", β, "   N_λ1, N_λ2 = ", N_λ1, ", ", N_λ2, "   Δt = ", Δt)
        println(io, "damping_relax = ", damping_relax, "   damping_dyn = ", c.damping_dyn,
                    "   kT = ", kT)
        println(io, "t_leads = ", t_leads, "   t_on = t_relax = ", t_on, "   t_rise = ", t_rise,
                    "   t_final = ", t_final, "   (", round((t_final - t_on) / T_drive, digits = 2),
                    " periodos de driver)")
        println(io, "salida: CSV cada ", OUT_STRIDE, " pasos (Δt_out = ", OUT_STRIDE * Δt,
                    "); fields.jld2 con ρ completa en esos tiempos")
    end
end

"t, sitio, rol, n_up, n_dn, Re/Im ρ_updn, sx, sy, sz, n_tot  (convencion de tdnegf_rho_t.csv)."
function write_sites_csv(c::Cfg, obs, idx)
    hdr = ["t", "site", "role", "n_up", "n_dn", "Re_rho_updn", "Im_rho_updn", "sx", "sy", "sz", "n_tot"]
    data = Matrix{Any}(undef, length(idx) * c.Nx, length(hdr))
    r = 0
    for s in 1:c.Nx, i in idx
        sx, sy, sz, nt = obs.σx_i[s, 1, i], obs.σx_i[s, 2, i], obs.σx_i[s, 3, i], obs.n_i[s, i]
        r += 1
        data[r, :] = Any[obs.t[i], s, site_role(c, s), 0.5 * (nt + sz), 0.5 * (nt - sz),
                         0.5 * sx, -0.5 * sy, sx, sy, sz, nt]
    end
    writedlm(joinpath(c.outdir, "sites_rho_t.csv"), vcat(permutedims(hdr), data), ",")
end

function write_spins_csv(c::Cfg, t, S_hist, idx)
    hdr = ["t"]
    for m in 1:3, a in ("x", "y", "z")
        push!(hdr, "M$(m)_$(a)")
    end
    data = Matrix{Float64}(undef, length(idx), length(hdr))
    for (r, i) in enumerate(idx)
        data[r, 1] = t[i]
        for m in 1:3, a in 1:3
            data[r, 1 + 3 * (m - 1) + a] = S_hist[a, m, i]
        end
    end
    writedlm(joinpath(c.outdir, "spins_t.csv"), vcat(permutedims(hdr), Any.(data)), ",")
end

"Corrientes de los leads, con el mismo signo y factor 1/2 que oscillators.jl."
function write_lead_csv(c::Cfg, obs, idx)
    hdr = ["t", "I_L", "I_R", "Isx_L", "Isy_L", "Isz_L", "Isx_R", "Isy_R", "Isz_R"]
    data = Matrix{Float64}(undef, length(idx), length(hdr))
    for (r, i) in enumerate(idx)
        data[r, :] = [obs.t[i], 0.5 * obs.Iα[1, i], -0.5 * obs.Iα[2, i],
                      0.5 * obs.Iαx[1, 1, i], 0.5 * obs.Iαx[1, 2, i], 0.5 * obs.Iαx[1, 3, i],
                      -0.5 * obs.Iαx[2, 1, i], -0.5 * obs.Iαx[2, 2, i], -0.5 * obs.Iαx[2, 3, i]]
    end
    writedlm(joinpath(c.outdir, "lead_currents_t.csv"), vcat(permutedims(hdr), Any.(data)), ",")
end

"""
Bloques de ρ en TODOS los enlaces de primeros vecinos (n, n+1), sitios 1-based.
Mismo formato que tdnegf_bond_rho_t.csv / tdnegf_bond_H.csv (columna lead = "C"),
asi bond_currents de inbedding_leads_plot.py los lee tal cual. Convencion
ρ_ab = ⟨c_b† c_a⟩ (la de ρ_ab en TDNEGF).
"""
function write_bond_csv(c::Cfg, t_b, rho_t, H0, N_loc)
    hdr = ["t", "lead", "n", "m"]
    for blk in ("nm", "mn"), s in ("uu", "ud", "du", "dd"), cc in ("Re", "Im")
        push!(hdr, "$(cc)_rho_$(blk)_$(s)")
    end
    nb = c.Nx - 1
    data = Matrix{Any}(undef, nb * length(t_b), length(hdr))
    r = 0
    for n in 1:nb, io in eachindex(t_b)
        ra, rb = get_sub(n, N_loc), get_sub(n + 1, N_loc)
        A = rho_t[ra, rb, io]; B = rho_t[rb, ra, io]
        row = Any[t_b[io], "C", n, n + 1]
        for X in (A, B), (s1, s2) in ENT
            push!(row, real(X[s1, s2])); push!(row, imag(X[s1, s2]))
        end
        r += 1
        data[r, :] = row
    end
    writedlm(joinpath(c.outdir, "bond_rho_t.csv"), vcat(permutedims(hdr), data), ",")

    hdrH = ["lead", "n", "m"]
    for blk in ("nm", "mn"), s in ("uu", "ud", "du", "dd"), cc in ("Re", "Im")
        push!(hdrH, "$(cc)_H_$(blk)_$(s)")
    end
    dataH = Matrix{Any}(undef, nb, length(hdrH))
    for n in 1:nb
        ra, rb = get_sub(n, N_loc), get_sub(n + 1, N_loc)
        row = Any["C", n, n + 1]
        for X in (H0[ra, rb], H0[rb, ra]), (s1, s2) in ENT
            push!(row, real(X[s1, s2])); push!(row, imag(X[s1, s2]))
        end
        dataH[n, :] = row
    end
    writedlm(joinpath(c.outdir, "bond_H.csv"), vcat(permutedims(hdrH), dataH), ",")
end

# corrida
function run_setup(setup::Symbol, axis::Symbol; outroot::AbstractString,
                   N_BUF::Int = 4, damping_dyn::Float64 = 0.0)
    haskey(LAYOUTS, setup) || error("setup desconocido: $setup")
    axis in (:z, :y) || error("eje de precesion desconocido: $axis")
    Nx = 2 * N_BUF + 3
    c0 = Cfg(setup, axis, LAYOUTS[setup], N_BUF, Nx, damping_dyn, "")
    outdir = joinpath(outroot, "$(setup)_" * param_tag(c0)); mkpath(outdir)
    c = Cfg(setup, axis, LAYOUTS[setup], N_BUF, Nx, damping_dyn, outdir)
    BLAS.set_num_threads(N_BLAS)

    println("="^72)
    @printf("DOS OSCILADORES + DRIVER   %s   precesion alrededor de %s\n", setup, axis)
    @printf("Nx=%d  N_BUF=%d  LMMs en los sitios %s  roles %s\n", Nx, N_BUF,
            string([spin_site(c, m) for m in 1:3]), string(c.roles))
    @printf("γ=%.6f γso=%.3f j_sd=%.3f θ=%.1f° Ω=%.4f (T=%.1f)  damping relax/dyn = %.2f/%.2f\n",
            γ, γso, j_sd, rad2deg(θ_max), Ω, T_drive, damping_relax, damping_dyn)
    @printf("t_leads=%.0f  t_on=%.0f  t_rise=%.0f  t_final=%.0f (%.1f periodos de driver)\n",
            t_leads, t_on, t_rise, t_final, (t_final - t_on) / T_drive)
    println("salidas en ", outdir)
    println("="^72); flush(stdout)
    write_params_txt(c); write_geometry_csv(c)

    p_model = ModelParamsTDNEGF(Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb, Nα = 2,
                                N_λ1 = N_λ1, N_λ2 = N_λ2)
    H0 = build_H_ab(; Nx = Nx, Ny = Ny, Nσ = Nσ, N_orb = N_orb, γ = γ, γso = complex(γso, 0.0))

    Rλ, zλ = load_poles_square(N_λ1, N_λ2)
    Σᴸ = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)
    Σᴳ = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)
    χ  = build_χ_nλ(zλ,      Ny, Nσ, N_orb, N_λ1, N_λ2; β = β, γ = γ, μ = E_F)
    ξ_L = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = 1,  y_coup = 1:Ny)
    ξ_R = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol = Nx, y_coup = 1:Ny)
    blocks = [SelfEnergyBlock(:left,  p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_L),
              SelfEnergyBlock(:right, p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_R)]
    ξ_L0, ξ_R0 = copy(ξ_L), copy(ξ_R)
    set_lead_coupling!(blocks, ξ_L0, ξ_R0, lead_switch(0.0))

    p_model.H0_ab .= H0
    p_model.H_ab  .= H0
    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks, ComplexF64[0.0, 0.0], p_model)
    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)
    @printf("Vector de estado: |u| = %d  (%.2f MB)\n", length(u0), 16 * length(u0) / 1e6)

    sys = init_spins(c)
    N_loc = p_model.N_loc
    site_ranges = [get_sub(i, N_loc) for i in 1:p_model.N_sites]

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_final), p_blocks)
    intg = init(prob, Vern7(); dt = Δt, save_everystep = false, adaptive = true, dense = false)

    llg_relax = Langevin(Δt; damping = damping_relax, kT = kT)
    # sin damping intrinseco: integrador conservativo (Langevin exige damping > 0)
    # (Sunny.ImplicitMidpoint: DifferentialEquations exporta otro ImplicitMidpoint)
    llg_dyn = damping_dyn > 0 ? Langevin(Δt; damping = damping_dyn, kT = kT) : Sunny.ImplicitMidpoint(Δt)

    N_steps = Int(round(t_final / Δt))
    obs = ObservablesTDNEGF(p_model; N_tmax = N_steps, N_leads = 2)
    S_hist = Array{Float64}(undef, 3, 3, N_steps)
    idx_out = 1:OUT_STRIDE:N_steps
    Ns = p_blocks.dims_ρ_ab[1]
    rho_t = Array{ComplexF64}(undef, Ns, Ns, length(idx_out))     # ρ completa (el sistema es chico)
    t_b = Vector{Float64}(undef, length(idx_out))

    started = time()
    for i in 1:N_steps
        obs.idx = i
        llg = intg.t < t_relax ? llg_relax : llg_dyn

        DifferentialEquations.step!(intg, Δt, true)
        set_lead_coupling!(blocks, ξ_L0, ξ_R0, lead_switch(intg.t))
        Sunny.step!(sys, llg)
        force_driven!(sys, c, intg.t)

        dv = pointer_blocks(intg.u, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        obs.t[i] = intg.t
        obs_n_i!(dv, p_model, obs)
        obs_σ_i!(dv, p_model, obs)
        obs_Ixα!(dv, p_blocks, obs)
        if (i - 1) % OUT_STRIDE == 0
            io = (i - 1) ÷ OUT_STRIDE + 1
            t_b[io] = intg.t
            rho_t[:, :, io] .= dv.ρ_ab
        end
        for m in 1:3, a in 1:3
            S_hist[a, m, i] = sys.dipoles[m, 1, 1, 1][a]
        end

        update_H_s_free!(sys, c, obs.σx_i[:, :, i])
        update_H_e!(p_model, site_ranges, full_dipoles(sys, c), j_sd)

        if i % 1000 == 0
            etapa = intg.t < t_leads ? "encendiendo leads" :
                    (intg.t < t_on ? "relajacion" :
                     (intg.t < t_on + t_rise ? "rampa del cono" : "driver estacionario"))
            M = [sys.dipoles[m, 1, 1, 1] for m in free_spins(c)]
            @printf("  t=%8.1f/%.0f  [%s]  M_F=%s  I_L=% .3e  elapsed=%.0fs\n", intg.t, t_final,
                    etapa, join([@sprintf("(% .3f,% .3f,% .3f)", v[1], v[2], v[3]) for v in M], " "),
                    0.5 * obs.Iα[1, i], time() - started)
            flush(stdout)
        end
    end
    @printf("\nDinamica lista en %.1f s\n", time() - started)

    # chequeos: ρ hermitica y hopping de TDNEGF igual al T̂ = -γσ0 - iγso σy del analitico
    eh = maximum(maximum(abs.(rho_t[:, :, io] - adjoint(rho_t[:, :, io]))) for io in eachindex(t_b))
    That = -γ * ComplexF64[1 0; 0 1] - im * γso * ComplexF64[0 -im; im 0]
    eT = maximum(maximum(abs.(H0[get_sub(n, N_loc), get_sub(n + 1, N_loc)] - That)) for n in 1:Nx-1)
    @printf("chequeos: max|ρ - ρ†| = %.2e   max|H_{n,n+1} - T̂| = %.2e\n", eh, eT)
    eT > 1e-10 && println("   AVISO: el hopping de TDNEGF no coincide con T̂ = -γσ0 - iγso σy")

    # salidas
    ckpt = joinpath(outdir, "checkpoint_t$(round(Int, t_final)).jld2")
    S_final = [sys.dipoles[m, 1, 1, 1][a] for a in 1:3, m in 1:3]
    jldsave(ckpt; u = collect(intg.u), t = intg.t, dipoles = S_final,
            geometry = geometry(c), params = prep_params(c))
    jldsave(joinpath(outdir, "fields.jld2");
            t = obs.t, spins = S_hist, sigma_i = obs.σx_i, n_i = obs.n_i,
            I_alpha = obs.Iα, I_alpha_x = obs.Iαx,
            t_rho = t_b, rho = rho_t, H0 = Matrix(H0),
            geometry = geometry(c), params = prep_params(c))
    write_sites_csv(c, obs, idx_out)
    write_spins_csv(c, obs.t, S_hist, idx_out)
    write_lead_csv(c, obs, idx_out)
    write_bond_csv(c, t_b, rho_t, H0, N_loc)
    println("salidas: fields.jld2  checkpoint  geometry.csv  params.txt  sites_rho_t.csv  " *
            "spins_t.csv  lead_currents_t.csv  bond_rho_t.csv  bond_H.csv")
    return outdir
end
