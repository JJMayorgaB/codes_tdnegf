#!/usr/bin/env julia
#
# Dinámica acoplada LLG + TDNEGF en una constricción.
#
# Zona central Nx=15 × Ny=8. Los 5 columnas iniciales y finales son rectángulos
# completos; en las 5 intermedias se recortan dos triángulos (bases 5-3-1) desde
# los bordes y=0 e y=7, dejando un cuello de 2 sitios en x=7.
#
# Los sitios recortados se eliminan anulando sus hoppings (fila y columna de
# H_ab a cero). Quedan exactamente desacoplados. Los espines clásicos en esos
# sitios se aíslan anulando sus bonds de exchange.

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


# Configuración de las corridas
# config : :ferro | :antiferro   → textura inicial
# J_x, J_y : exchange en cada dirección (+ antiferro, − ferro en convención Sunny)
const RUNS = [
    (name = "FM_sym",   config = :ferro,     J_x = -0.05, J_y = -0.05),
    (name = "FM_asym",  config = :ferro,     J_x = -0.05, J_y = -0.02),
    (name = "AFM_sym",  config = :antiferro, J_x = +0.05, J_y = +0.05),
    (name = "AFM_asym", config = :antiferro, J_x = +0.05, J_y = +0.02),
]


# Parámetros
const Nx, Ny, Nσ, N_orb = 15, 8, 2, 1
const N_λ1, N_λ2 = 49, 20
const β     = 40.0
const γ     = 1.0
const γso   = 0.0 + 0.0im    
const E_F   = 0.0
const j_sd  = 0.1               # acoplamiento sd electrón–espín
const j_ani = 0.01              # anisotropía uniaxial
const Bx    = 1e-5              # campo semilla para romper degeneración
const a0    = 1.0

const V_bias   = 0.5           
const t_on     = 100.0          
const t_rise   = 10.0           
const Δt       = 0.1
const t_end    = 200.0

const damping  = 0.5
const kT       = 0.0

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# Geometría: máscara de la constricción
# Especificada en las coordenadas x en (0,14), y en (0,7)
const REMOVED_0BASED = vcat(
    [(x, 7) for x in 5:9], [(x, 6) for x in 6:8], [(7, 5)],   # triángulo superior
    [(x, 0) for x in 5:9], [(x, 1) for x in 6:8], [(7, 2)],   # triángulo inferior
)

#Máscara `keep[x,y]` en índices 1-based de Julia
function build_mask()
    keep = trues(Nx, Ny)
    for (x0, y0) in REMOVED_0BASED
        keep[x0 + 1, y0 + 1] = false
    end
    return keep
end

const KEEP    = build_mask()
const REMOVED = [(x0 + 1, y0 + 1) for (x0, y0) in REMOVED_0BASED]

@inline site_index(x::Int, y::Int) = (x - 1) * Ny + y

function print_geometry(keep)
    println("Geometría (■ presente, · eliminado):")
    print("      x:")
    for x in 1:Nx; @printf("%3d", x - 1); end
    println()
    for y in Ny:-1:1
        @printf("  y=%d :  ", y - 1)
        for x in 1:Nx
            print(keep[x, y] ? " ■ " : " · ")
        end
        println()
    end
    print("  ancho:  ")
    for x in 1:Nx; @printf("%3d", count(keep[x, :])); end
    println("\n  sitios activos: ", count(keep), " de ", Nx*Ny)
end


# Lado electrónico
#Anula filas y columnas de los sitios eliminados: desacople exacto
function carve!(H::Matrix{ComplexF64}, keep, Nloc::Int)
    for x in 1:Nx, y in 1:Ny
        keep[x, y] && continue
        l = site_index(x, y)
        r = (l - 1)*Nloc + 1 : l*Nloc
        H[r, :] .= 0
        H[:, r] .= 0
    end
    return H
end

function init_electrons(Rλ, zλ)
    p_model = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb,
                                Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)

    H0 = build_H_ab(; Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, γ=γ, γso=γso)
    carve!(H0, KEEP, p_model.N_loc)

    Σᴸ = build_Σᴸ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=E_F)
    Σᴳ = build_Σᴳ_nλ(Rλ, zλ, Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=E_F)
    χ  = build_χ_nλ(zλ,      Ny, Nσ, N_orb, N_λ1, N_λ2; β=β, γ=γ, μ=E_F)

    ξ_L = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=1,  y_coup=1:Ny)
    ξ_R = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=Nx, y_coup=1:Ny)

    blocks = [SelfEnergyBlock(:left,  p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_L),
              SelfEnergyBlock(:right, p_model.Nc, N_λ1, N_λ2, Σᴸ, Σᴳ, χ, ξ_R)]

    p_model.H0_ab .= H0
    p_model.H_ab  .= H0

    # SIN copy: p_blocks.H_ab debe aliasar p_model.H_ab para que update_H_e!
    # (que muta p_model.H_ab) llegue efectivamente al RHS del ODE.
    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks,
                                          ComplexF64[0.0, 0.0], p_model)

    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)
    return p_model, p_blocks, u0, H0
end

# Lado magnético (Sunny)
function init_spins(; config::Symbol, J_x::Float64, J_y::Float64)
    latvecs   = lattice_vectors(a0, a0*(1 + 1e-3), 4*a0, 90, 90, 90)
    positions = [[0.5, 0.5, 0.0]]
    cryst     = Crystal(latvecs, positions)
    moments   = [1 => Moment(s = 1.0, g = 1.0)]

    sys = System(cryst, moments, :dipole; dims = (Nx, Ny, 1))
    set_field!(sys, [Bx, 0.0, 0.0])
    set_exchange!(sys, J_x, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J_y, Bond(1, 1, [0, 1, 0]))
    set_onsite_coupling!(sys, S -> -j_ani*S[3]^2, 1)

    sys = to_inhomogeneous(sys)
    remove_periodicity!(sys, (true, true, true))

    # aísla los sitios eliminados: exchange cero con todos sus vecinos
    for (x, y) in REMOVED
        for (dx, dy) in ((1,0), (-1,0), (0,1), (0,-1))
            xn, yn = x + dx, y + dy
            (1 <= xn <= Nx && 1 <= yn <= Ny) || continue
            set_exchange_at!(sys, 0.0, (x, y, 1, 1), (xn, yn, 1, 1);
                             offset = (dx, dy, 0))
        end
        set_field_at!(sys, [0.0, 0.0, 0.0], (x, y, 1, 1))
    end

    # textura inicial
    # Los sitios inertes NO pueden llevar dipolo nulo: en modo :dipole Sunny
    # renormaliza a |S| = s en cada paso y normalizar el vector cero da NaN.
    # Se les pone (0,0,1), que con campo y exchange nulos es un punto fijo.
    for x in 1:Nx, y in 1:Ny
        s = if !KEEP[x, y]
            Sunny.SVector(0.0, 0.0, 1.0)          # sitio inerte, estático
        elseif config === :antiferro
            Sunny.SVector(0.0, 0.0, float((-1)^(x + y)))
        else
            Sunny.SVector(0.0, 0.0, 1.0)
        end
        sys.dipoles[x, y, 1, 1] = s
    end
    return sys
end

#Dipolos con ceros en los sitios eliminados, para alimentar update_H_e!
function masked_dipoles(sys)
    S = Matrix{SVector{3,Float64}}(undef, Nx, Ny)
    for x in 1:Nx, y in 1:Ny
        S[x, y] = KEEP[x, y] ? SVector{3,Float64}(sys.dipoles[x, y, 1, 1]) :
                               SVector{3,Float64}(0.0, 0.0, 0.0)
    end
    return S
end

# Perfil del bias
smooth_switch(t; ti = t_rise) = t < 0 ? 0.0 : (t < ti ? sin((π/2)*t/ti)^2 : 1.0)

# Evolución acoplada
function run_case(cfg, Rλ, zλ)
    @printf("\n[%s]  config=%s  J_x=%+.3f  J_y=%+.3f\n",
            cfg.name, cfg.config, cfg.J_x, cfg.J_y)
    flush(stdout)

    p_model, p_blocks, u0, _ = init_electrons(Rλ, zλ)
    sys = init_spins(config = cfg.config, J_x = cfg.J_x, J_y = cfg.J_y)

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_end), p_blocks)
    intg = init(prob, Vern7(); dt = Δt, save_everystep = false,
                adaptive = true, dense = false)
    llg  = Langevin(Δt; damping = damping, kT = kT)

    N_steps = Int(round(t_end / Δt))
    obs = ObservablesTDNEGF(p_model; N_tmax = N_steps, N_leads = 2)
    site_ranges = [get_sub(i, p_model.N_loc) for i in 1:p_model.N_sites]

    started = time()
    for i in 1:N_steps
        obs.idx = i
        DifferentialEquations.step!(intg, Δt, true)
        Sunny.step!(sys, llg)

        dv = pointer_blocks(intg.u, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        ρ  = ρ_eq(E_F, β, p_model.H_ab, N_λ2, Nx, Ny, Nσ, N_orb)

        obs.t[i] = intg.t
        obs_n_i!(dv, p_model, obs)
        obs_σ_i!(dv, p_model, obs)
        obs_Ixα!(dv, p_blocks, obs)
        obs_s_i!(sys.dipoles[:, :, 1, 1], p_model, obs)
        obs_σ_i_eq!(ρ, p_model, obs)

        # realimentación mutua
        update_H_s!(Nx, Ny, sys, obs.σx_i[:, :, i], j_sd)
        update_H_e!(p_model, site_ranges, masked_dipoles(sys), j_sd)

        Δ = smooth_switch(intg.t - t_on) * V_bias + 0im
        p_blocks.Δ_blocks[1] = +Δ/2
        p_blocks.Δ_blocks[2] = -Δ/2

        if i % 200 == 0
            @printf("  t=%6.1f/%.0f   I_L=% .4e   elapsed=%.0fs\n",
                    intg.t, t_end, obs.Iα[1, i], time() - started)
            flush(stdout)
        end
    end
    return obs
end


# update_H_s! : campo efectivo sobre los espines (adaptado del notebook 02)
function update_H_s!(Nx::Int, Ny::Int, sys, σx_i::Array{Float64,2},
                     j_sd::Float64, B0::Vector{Float64} = [Bx, 0.0, 0.0])
    @inbounds for x in 1:Nx, y in 1:Ny
        KEEP[x, y] || continue                    # los inertes no reciben campo
        idx = site_index(x, y)
        set_field_at!(sys, -σx_i[idx, :] .* j_sd .+ B0, (x, y, 1, 1))
    end
    return nothing
end


# Outputs
# Magnetización y vector de Néel promediados sobre sitios activos
function order_parameters(obs, nt)
    M = zeros(3, nt); Neel = zeros(3, nt)
    n_act = count(KEEP)
    for t in 1:nt, x in 1:Nx, y in 1:Ny
        KEEP[x, y] || continue
        l = site_index(x, y); sgn = (-1)^(x + y)
        for c in 1:3
            M[c, t]    += obs.sx_i[l, c, t]
            Neel[c, t] += obs.sx_i[l, c, t] * sgn
        end
    end
    return M ./ n_act, Neel ./ n_act
end

function save_case(cfg, obs)
    nt = length(obs.t)
    M, Nv = order_parameters(obs, nt)

    # ½ corrige el factor 2 de obs.Iα / obs.Iαx (observables.jl)
    I_L  =  0.5 .* obs.Iα[1, :];   I_R  = -0.5 .* obs.Iα[2, :]
    Is_L =  0.5 .* obs.Iαx[1, :, :];  Is_R = -0.5 .* obs.Iαx[2, :, :]

    writedlm(joinpath(OUT, "trace_$(cfg.name).csv"),
        vcat(["t" "I_L" "I_R" "Isx_L" "Isy_L" "Isz_L" "Isx_R" "Isy_R" "Isz_R" "M_x" "M_y" "M_z" "Neel_x" "Neel_y" "Neel_z"],
             hcat(obs.t, I_L, I_R,
                  Is_L[1,:], Is_L[2,:], Is_L[3,:],
                  Is_R[1,:], Is_R[2,:], Is_R[3,:],
                  M[1,:], M[2,:], M[3,:], Nv[1,:], Nv[2,:], Nv[3,:])), ",")

    jldsave(joinpath(OUT, "fields_$(cfg.name).jld2");
            t = obs.t, keep = KEEP,
            n_i = obs.n_i, sigma_i = obs.σx_i,
            sigma_eq = obs.σx_i_eq, s_i = obs.sx_i,
            config = String(cfg.config), J_x = cfg.J_x, J_y = cfg.J_y)

    @printf("  → trace_%s.csv  y  fields_%s.jld2\n", cfg.name, cfg.name)
end


function main()
    sel = isempty(ARGS) ? RUNS : filter(c -> c.name in ARGS, RUNS)
    if isempty(sel)
        error("Ninguna corrida coincide con $(ARGS). Opciones: " *
              join((c.name for c in RUNS), ", "))
    end

    BLAS.set_num_threads(1)
    Rλ, zλ = load_poles_square(N_λ1, N_λ2)

    print_geometry(KEEP)
    @printf("\nNc=%d  Ns=%d  N_λ=%d   pasos=%d (Δt=%.2f, t_end=%.0f)\n",
            Ny*Nσ*N_orb, Nx*Ny*Nσ*N_orb, N_λ1 + N_λ2,
            Int(round(t_end/Δt)), Δt, t_end)
    @printf("Corridas: %s\nHilos de Julia: %d (se usan %d)\n",
            join((c.name for c in sel), ", "), Threads.nthreads(), length(sel))
    if Threads.nthreads() < length(sel)
        @printf("AVISO: hay %d corridas y solo %d hilos. Relanza con --threads=%d\n",
                length(sel), Threads.nthreads(), length(sel))
    end
    flush(stdout)

    started = time()
    Threads.@threads for i in eachindex(sel)
        cfg = sel[i]
        obs = run_case(cfg, Rλ, zλ)
        save_case(cfg, obs)
    end
    @printf("\nListo en %.1f s. Salidas en %s\n", time() - started, OUT)
end

main()
