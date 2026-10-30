#!/usr/bin/env julia
#=
bond_current_llg.jl -- matriz densidad de los F (enlace a -> b y sitios a, b) a partir de la
G lesser vestida a primer orden con los momentos de la LLG efectiva, en el estado estacionario.

Funciones de Green del lead derecho con el driver por embedding (kernel_elements.jl, k = 0, ±1):

    G^μ_{mn}(t,t') = Σ_k e^{-ikΩt} g^μ_{mn,k}(t-t'),   g^μ_{mn,k}(τ) = ∫dω/2π e^{-iωτ} 𝒢^μ_{mn,k}(ω)

(coeff_tau de kernel_elements.jl; t = tiempo de Floquet = t_lab - t_on). Primer orden en los F:

    𝒢^<_{ij}(t,t) = G^<_{ij}(t,t) - J_sd Σ_{ℓ≠k} ∫dt'' [ G^<_{iℓ}(t,t'') σ·M_ℓ(t'') G^a_{ℓj}(t'',t)
                                                      + G^r_{iℓ}(t,t'') σ·M_ℓ(t'') G^<_{ℓj}(t'',t) ]

Como G^a_{ℓj}(t'',t) y G^r_{iℓ}(t,t'') son causales, t'' = t - τ con τ ∈ [0, τmax]:

    𝒢^<_{ij}(t,t) = Σ_k e^{-ikΩt} g^<_{ij,k}(0)
        - J_sd Σ_ℓ Σ_{k,k'} e^{-i(k+k')Ωt} ∫_0^{τmax} dτ e^{ik'Ωτ}
              [ g^<_{iℓ,k}(τ) σ·M_ℓ(t-τ) g^a_{ℓj,k'}(-τ) + g^r_{iℓ,k}(τ) σ·M_ℓ(t-τ) g^<_{ℓj,k'}(-τ) ]

Integral en τ por trapecio con paso dτ. En τ = 0 se usan los limites laterales que pide la
integral: g^r_{mm,0}(0+) = -i 𝕀, g^a_{mm,0}(0-) = +i 𝕀 (coeff_tau da el punto medio, -i/2 y +i/2).
M_ℓ(t) = trayectorias de laboratorio de la LLG (spins_t_llg.csv), interpoladas (Catmull-Rom)
a la malla dτ.

Estado estacionario: la salida son los tiempos de la LLG en sus ultimos --last periodos,
corridos un numero entero de periodos para que caigan sobre los ultimos --last periodos de
TDNEGF (misma fase del driver, como spins_t_cmp). La memoria [t - τmax, t] cae entera en la
parte final de la LLG (punto fijo del marco rotante); si no, el script se detiene.

Salida, con el mismo formato que bond_rho_t.csv de TDNEGF (sitios de TDNEGF, lead = C,
rho_nm = -i 𝒢^<_{nm}(t,t), rho_mn = -i 𝒢^<_{mn}(t,t)):
    output/<tag>/F<a>-<b>/bond_rho_llg.csv     pares (n,m) = (a,b), (a,a), (b,b) (t de la LLG)
La corriente y la figura las hace plot_bond_current_llg.py (con bond_currents, como TDNEGF).

Uso (desde LLG_equation):
    julia -t auto bond_current_llg.jl --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005 --sites 1,2
        [--last 5] [--taumax 1000] [--dtau 0.1] [--log2n 21] [--eta 1e-4]
=#

include(joinpath(@__DIR__, "..", "kernel_elements.jl"))

const HERE_B  = @__DIR__
const TDROOT_B = joinpath(HERE_B, "..", "..", "two_oscillators_setup", "y_precession", "output", "DFF")
const SIGS = (σx, σy, σz)

# ---------------------------------------------------------------- g^μ_{mn,k}(τ) como matriz 2x2
"Matriz 2x2 de g^μ_{mn,k}(τ) en la columna r de la tabla A de coeff_tau (μ = :r, :a, :l)."
@inline function gmat(A::Matrix{ComplexF64}, μ::Symbol, k::Int, r::Int)
    i0 = μ === :r ? 0 : (μ === :a ? 4 : 8)
    k == 0 && return A[i0+1, r] * σ0 + A[i0+2, r] * σy
    k == 1 && return A[i0+3, r] * τp
    return A[i0+4, r] * τm
end

"Tablas en τ de los pares (m,n) de los F, con los limites laterales en τ = 0 del diagonal."
function green_tau(sites, p::KParams, c::TauConfig)
    Ag = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    Nτ = 0
    for m in sites, n in sites
        t0 = time()
        A, Nτ = coeff_tau(m, n, p, c)
        if m == n
            A[IAR, Nτ+1] += -im / 2                 # -i/2 -> g^r(0+) = -i
            A[IAA, Nτ+1] +=  im / 2                 # +i/2 -> g^a(0-) = +i
        end
        Ag[(m, n)] = A
        @printf("  g^μ_(%d%d,k)(τ) en %.1f s\n", m, n, time() - t0)
    end
    return Ag, Nτ
end

# ---------------------------------------------------------------- kernel de 𝒢^<_{ij}
"""
Kc[(K+2)*4 + q, j+1, β, ℓ]: entrada q (orden de columnas de la 2x2) del armonico K = k + k' de
    w_j e^{ik'Ωτ_j} [ g^<_{iℓ,k}(τ_j) σ^β g^a_{ℓj,k'}(-τ_j) + g^r_{iℓ,k}(τ_j) σ^β g^<_{ℓj,k'}(-τ_j) ]
(w_j: pesos del trapecio), sumado sobre k + k' = K.
"""
function kernel_pair(Ag, i::Int, jj::Int, sites, p::KParams, Nτ::Int, dτ::Float64)
    Kc = zeros(ComplexF64, 20, Nτ + 1, 3, length(sites))
    for (il, l) in enumerate(sites)
        Ail = Ag[(i, l)]; Alj = Ag[(l, jj)]
        @threads for j in 0:Nτ
            rp = Nτ + 1 + j; rm = Nτ + 1 - j
            w = (j == 0 || j == Nτ) ? dτ / 2 : dτ
            for k in -1:1, kp in -1:1
                Lp = gmat(Ail, :l, k, rp); Rp = gmat(Ail, :r, k, rp)
                Am = gmat(Alj, :a, kp, rm); Lm = gmat(Alj, :l, kp, rm)
                ph = w * cis(kp * p.Ω * j * dτ)
                e0 = (k + kp + 2) * 4
                for (ib, s) in enumerate(SIGS)
                    X = ph * (Lp * s * Am + Rp * s * Lm)
                    for q in 1:4
                        Kc[e0+q, j+1, ib, il] += X[q]
                    end
                end
            end
        end
    end
    return Kc
end

"""
𝒢^<_{ij}(t,t) en los tiempos de salida: G[:, :, n]. Mf[:, β, ℓ] = M^β_ℓ en la malla fina,
base[n] = indice de t_n en esa malla, tF = tiempos de Floquet de la salida.
"""
function dressed_lesser(Ag, i::Int, jj::Int, sites, p::KParams, Nτ::Int, dτ::Float64,
                        Mf::Array{Float64,3}, base::Vector{Int}, tF::Vector{Float64})
    Kc = kernel_pair(Ag, i, jj, sites, p, Nτ, dτ)
    A0 = Ag[(i, jj)]
    G0k = [gmat(A0, :l, k, Nτ + 1) for k in -1:1]
    nt = length(base)
    G = Array{ComplexF64}(undef, 2, 2, nt)
    @threads for n in 1:nt
        acc = zeros(ComplexF64, 20)
        for il in eachindex(sites), ib in 1:3
            for j in 0:Nτ
                mv = Mf[base[n]-j, ib, il]
                @inbounds @simd for e in 1:20
                    acc[e] += Kc[e, j+1, ib, il] * mv
                end
            end
        end
        g = zeros(ComplexF64, 2, 2)
        for k in -1:1
            g .+= cis(-k * p.Ω * tF[n]) .* G0k[k+2]
        end
        for K in -2:2
            ph = cis(-K * p.Ω * tF[n]); e0 = (K + 2) * 4
            for q in 1:4
                g[q] -= p.J_sd * ph * acc[e0+q]
            end
        end
        G[:, :, n] = g
    end
    return G
end

# ---------------------------------------------------------------- entrada
"Interpolacion de Catmull-Rom en una malla uniforme t."
function catmull_rom(t::Vector{Float64}, y::AbstractVector{Float64}, tq::Float64)
    h = t[2] - t[1]
    x = (tq - t[1]) / h
    s = clamp(floor(Int, x), 1, length(t) - 3)          # puntos s-1, s, s+1, s+2 (base 0)
    u = x - s
    P0, P1, P2, P3 = y[s], y[s+1], y[s+2], y[s+3]
    return 0.5 * (2P1 + (-P0 + P2) * u + (2P0 - 5P1 + 4P2 - P3) * u^2 + (-P0 + 3P1 - 3P2 + P3) * u^3)
end

function main_bond(args = ARGS)
    opt = Dict("--tag" => "setup2_gso0p1_jsd0p2_th10deg_Om0p005", "--sites" => "1,2", "--last" => "5",
               "--taumax" => "1000", "--dtau" => "0.1", "--log2n" => "21", "--eta" => "1e-4", "--tdnegf" => "")
    i = 1
    while i ≤ length(args)
        haskey(opt, args[i]) && i < length(args) || error("argumento no reconocido: $(args[i])")
        opt[args[i]] = args[i+1]; i += 2
    end
    tag = opt["--tag"]
    a, b = parse.(Int, split(opt["--sites"], ','))
    llgdir = joinpath(HERE_B, "output", tag, "F$a-$b")
    ptxt = read(joinpath(llgdir, "params_llg.txt"), String)
    t_on = parse(Float64, match(r"t_on = ([-+0-9.eE]+)", ptxt).captures[1])
    tdir = opt["--tdnegf"]
    if isempty(tdir)
        tdir = strip(match(r"tdnegf = (.*)", ptxt).captures[1])
        isdir(tdir) || (tdir = joinpath(TDROOT_B, "F$a-$b", replace(tag, r"^setup\d+_" => "") * "_buf4_adyn0p0"))
    end
    isdir(tdir) || error("no encuentro la corrida de TDNEGF: $tdir")
    p = read_params(joinpath(tdir, "params.txt"))
    c = TauConfig(dτ = parse(Float64, opt["--dtau"]), τmax = parse(Float64, opt["--taumax"]),
                  log2N = parse(Int, opt["--log2n"]), η = parse(Float64, opt["--eta"]))
    T = 2π / p.Ω; dτ = c.dτ
    @printf("output/%s/F%d-%d   τmax = %g   dτ = %g   N = 2^%d   η = %g   hilos = %d\n",
            tag, a, b, c.τmax, dτ, c.log2N, c.η, nthreads())

    # ---- ventanas (como spins_t_cmp de plot_llg_two_moments.py)
    L = readdlm(joinpath(llgdir, "spins_t_llg.csv"), ',', Float64; skipstart = 1)
    tL = L[:, 1]
    tD = readdlm(joinpath(tdir, "spins_t.csv"), ',', Float64; skipstart = 1)[end, 1]
    shift = round((tL[end] - tD) / T) * T
    t_ini = tD - parse(Float64, opt["--last"]) * T
    sel = findall(t -> t_ini + shift - 1e-9 ≤ t ≤ tD + shift + 1e-9, tL)
    t_out = tL[sel]
    Nτ = ceil(Int, tmaxof(p, c) / dτ)
    g0 = t_out[1] - Nτ * dτ
    g0 ≥ tL[2] || error("la memoria [t - τmax, t] se sale de la LLG (baja --last o --taumax)")
    @printf("  ventana de la LLG: [%.1f, %.1f] (= TDNEGF [%.1f, %.1f] + %dT), memoria desde %.1f (%.1f T de LLG)\n",
            t_out[1], t_out[end], t_ini, tD, round(Int, shift / T), g0, (g0 - tL[1]) / T)
    Lf = round(Int, (t_out[end] - g0) / dτ) + 1
    Mf = Array{Float64}(undef, Lf, 3, 2)
    for (il, c0) in enumerate((5, 8)), ib in 1:3                # M2 = F1 (col 5:7), M3 = F2 (col 8:10)
        y = view(L, :, c0 + ib - 1)
        for q in 1:Lf
            Mf[q, ib, il] = catmull_rom(tL, y, g0 + (q - 1) * dτ)
        end
    end
    base = [round(Int, (t - g0) / dτ) + 1 for t in t_out]
    tF = t_out .- t_on

    # ---- funciones de Green y 𝒢^< vestida
    sites = (a, b)
    Ag, Nτ2 = green_tau(sites, p, c)
    Nτ2 == Nτ || error("Nτ inconsistente")
    for m in sites                                               # control: orden cero en t = 0 = s̄_m
        ρ0 = -im * sum(gmat(Ag[(m, m)], :l, k, Nτ + 1) for k in -1:1)
        @printf("  orden cero, s̄_%d = (%+.6e, %+.6e, %+.6e)\n", m, (real(tr(s * ρ0)) for s in SIGS)...)
    end
    t0 = time()
    G = Dict{Tuple{Int,Int},Array{ComplexF64,3}}()
    for (m, n) in ((a, b), (b, a), (a, a), (b, b))
        G[(m, n)] = dressed_lesser(Ag, m, n, sites, p, Nτ, dτ, Mf, base, tF)
    end
    @printf("  𝒢^< vestida (4 pares, %d tiempos) en %.1f s\n", length(t_out), time() - t0)

    # ---- salida en el formato de bond_rho_t.csv (sitios de TDNEGF)
    geo = readdlm(joinpath(tdir, "geometry.csv"), ',', String; skipstart = 1)
    sD = parse(Int, geo[findfirst(==("D"), geo[:, 3]), 2])
    path = joinpath(llgdir, "bond_rho_llg.csv")
    ent = ((1, 1), (1, 2), (2, 1), (2, 2))                       # uu, ud, du, dd
    open(path, "w") do io
        print(io, "t,lead,n,m")
        for blk in ("nm", "mn"), s in ("uu", "ud", "du", "dd")
            print(io, ",Re_rho_", blk, "_", s, ",Im_rho_", blk, "_", s)
        end
        println(io)
        for (m, n) in ((a, b), (a, a), (b, b))
            for k in eachindex(t_out)
                @printf(io, "%.10g,C,%d,%d", t_out[k], sD + m, sD + n)
                for X in (G[(m, n)], G[(n, m)]), (r, s) in ent
                    v = -im * X[r, s, k]
                    @printf(io, ",%.15e,%.15e", real(v), imag(v))
                end
                println(io)
            end
        end
    end
    println("  -> ", path)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_bond()
end
