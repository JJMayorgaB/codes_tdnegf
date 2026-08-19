# Definiciones compartidas por relax_state.jl y constriction_dynamics.jl.


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


# Parámetros comunes
const Nx, Ny, Nσ, N_orb = 15, 8, 2, 1
const N_λ1, N_λ2 = 49, 20
const β     = 40.0
const γ     = 1.0
const γso   = 0.0 + 0.0im
const E_F   = 0.0
const j_sd  = 0.1               # acoplamiento sd electrón–espín
const j_ani = 0.01              # anisotropía uniaxial eje fácil z
const Bx    = 1e-5              # campo semilla para romper degeneración
const a0    = 1.0
const Δt    = 0.1

# Amortiguamiento y temperatura.
const damping_relax = 0.5
const damping_dyn   = 0.007
const kT_relax      = 0.002       
const kT_dyn        = 0.002

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# Geometría: sin constricción. Rectángulo Nx × Ny completo, los 120 sitios
# activos. Es el caso de referencia contra el que comparar los dos setups con
# cuello: cualquier efecto que aparezca allá y no acá viene de la constricción.
#
# La lista vacía mantiene intacta toda la maquinaria que la usa —build_mask,
# carve!, el aislamiento de bonds en init_spins— sin necesidad de tocarla: los
# bucles simplemente no iteran.
const REMOVED_0BASED = Tuple{Int,Int}[]

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

    p_blocks = ExperimentalBlockRHSParams(p_model.H_ab, blocks,
                                          ComplexF64[0.0, 0.0], p_model)

    u0 = zeros(ComplexF64, p_blocks.dims_ρ_ab[1]^2 + p_blocks.aux_layout.total_size)
    return p_model, p_blocks, u0, H0
end

# Lado magnético (Sunny)
# `init` puede ser:
#   nothing                        → textura ideal según `config`
#   Matrix/Array de SVector o 3×N  → configuración explícita (estado relajado)
function init_spins(; config::Symbol, J_x::Float64, J_y::Float64, init = nothing)
    # a ≠ b rompe la simetría tetragonal: sin eso Sunny trata los bonds ±x y ±y
    latvecs   = lattice_vectors(a0, a0*(1 + 1e-3), 4*a0, 90, 90, 90)
    positions = [[0.5, 0.5, 0.0]]
    cryst     = Crystal(latvecs, positions)
    moments   = [1 => Moment(s = 1.0, g = 1.0)]

    sys = System(cryst, moments, :dipole; dims = (Nx, Ny, 1))
    set_field!(sys, [Bx, 0.0, 0.0])
    set_exchange!(sys, J_x, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, J_y, Bond(1, 1, [0, 1, 0]))

    # -K (S^z)², eje fácil a lo largo de z
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
    for x in 1:Nx, y in 1:Ny
        s = if !KEEP[x, y]
            # En modo :dipole Sunny renormaliza a |S|=1 cada paso, así que un dipolo nulo daría 0/0 = NaN. (0,0,1) con campo y exchange nulos es un punto fijo estático.
            Sunny.SVector(0.0, 0.0, 1.0)
        elseif init !== nothing
            v = init[x, y]
            Sunny.SVector(v[1], v[2], v[3])
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

# Perfil de encendido del bias
smooth_switch(t, ti) = t < 0 ? 0.0 : (t < ti ? sin((π/2)*t/ti)^2 : 1.0)

# Mismo esquema que el notebook 02: ahí la continuación se hace pasando
# (u0, m0, p0) de una llamada a la siguiente dentro de la misma sesión, y
# arrancando la segunda en t_0 = obs.t[end]. Como aquí son dos scripts distintos,
# lo mismo hay que serializarlo:
#   u        estado electrónico completo (ρ_ab + modos auxiliares), lo que en el
#            notebook es `u0`
#   dipoles  textura de espín, lo que es `m0`
"""
Escritura de resultados de una etapa. `tag` vale "" para la dinámica y "_relax"
para la relajación.

Las dos etapas escriben exactamente el mismo esquema de columnas y de campos, de
modo que la graficación pueda concatenarlas en el tiempo sin casos especiales y
mostrar la evolución completa como una sola corrida.
"""
function save_stage(cfg, obs; tag::String = "")
    nt = length(obs.t)
    M, Nv = order_parameters(obs, nt)

    # ½ corrige el factor 2 de obs.Iα / obs.Iαx (observables.jl)
    I_L  =  0.5 .* obs.Iα[1, :];      I_R  = -0.5 .* obs.Iα[2, :]
    Is_L =  0.5 .* obs.Iαx[1, :, :];  Is_R = -0.5 .* obs.Iαx[2, :, :]

    writedlm(joinpath(OUT, "trace$(tag)_$(cfg.name).csv"),
        vcat(["t" "I_L" "I_R" "Isx_L" "Isy_L" "Isz_L" "Isx_R" "Isy_R" "Isz_R" "M_x" "M_y" "M_z" "Neel_x" "Neel_y" "Neel_z"],
             hcat(obs.t, I_L, I_R,
                  Is_L[1,:], Is_L[2,:], Is_L[3,:],
                  Is_R[1,:], Is_R[2,:], Is_R[3,:],
                  M[1,:], M[2,:], M[3,:], Nv[1,:], Nv[2,:], Nv[3,:])), ",")

    jldsave(joinpath(OUT, "fields$(tag)_$(cfg.name).jld2");
            t = obs.t, keep = KEEP,
            n_i = obs.n_i, sigma_i = obs.σx_i,
            sigma_eq = obs.σx_i_eq, s_i = obs.sx_i,
            config = String(cfg.config), J_x = cfg.J_x, J_y = cfg.J_y)

    @printf("  → trace%s_%s.csv  y  fields%s_%s.jld2\n",
            tag, cfg.name, tag, cfg.name)
end

# Mismo esquema que el notebook 02: ahí la continuación se hace pasando
# (u0, m0, p0) de una llamada a la siguiente dentro de la misma sesión, y
# arrancando la segunda en t_0 = obs.t[end]. Como aquí son dos scripts distintos,
# lo mismo hay que serializarlo:
#   u        estado electrónico completo (ρ_ab + modos auxiliares), lo que en el
#            notebook es `u0`
#   dipoles  textura de espín, lo que es `m0`

#   t        tiempo final de la etapa previa, para continuar en t_0 = t

relaxed_path(name) = joinpath(OUT, "relaxed_$(name).jld2")

function save_relaxed(name, sys, u, t_final)
    A = Array{Float64}(undef, 3, Nx, Ny)
    for x in 1:Nx, y in 1:Ny
        v = sys.dipoles[x, y, 1, 1]
        A[1, x, y] = v[1]; A[2, x, y] = v[2]; A[3, x, y] = v[3]
    end
    jldsave(relaxed_path(name);
            dipoles = A, u = Vector{ComplexF64}(u), keep = KEEP,
            t = t_final, name = name)
    @printf("  → relaxed_%s.jld2  (t=%.1f, %d componentes de u)\n",
            name, t_final, length(u))
end

"Devuelve (dipolos::Matrix{SVector{3}}, u::Vector{ComplexF64}, t_0::Float64)."
function load_relaxed(name)
    p = relaxed_path(name)
    isfile(p) || error("Falta $(p). Corre antes relax_state.jl")
    A, u, t0 = load(p, "dipoles"), load(p, "u"), load(p, "t")
    S = [SVector{3,Float64}(A[1, x, y], A[2, x, y], A[3, x, y])
         for x in 1:Nx, y in 1:Ny]
    return S, u, t0
end
