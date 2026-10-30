#!/usr/bin/env julia
#=
bond_current_llg.jl -- matriz densidad de los F (enlace a -> b y sitios a, b) a partir de la
G lesser vestida a primer orden con los momentos de la LLG efectiva, en el estado estacionario.

Funciones de Green del lead derecho con el driver por embedding (kernel_elements.jl, k = 0, ±1):

    G^μ_{mn}(t,t') = Σ_k e^{-ikΩt} 𝒢^μ_{mn,k}(t-t'),   𝒢^μ_{mn,k}(τ) = ∫dω/2π e^{-iωτ} 𝒢^μ_{mn,k}(ω)

(coeff_tau de kernel_elements.jl; t = tiempo de Floquet = t_lab - t_on). G lesser vestida a
primer orden en los F (𝙂 = G vestida), con σ·M_j = Σ_β M^β_j σ^β y j = 1, 2 (los dos F):

    𝙂^<_{mn}(t,t) = G^<_{mn}(t,t) - J_sd Σ_{j=1,2} Σ_β ∫dt'' X^β_{mn;j}(t,t'') M^β_j(t''),
    X^β_{mn;j}(t,t'') = G^<_{mj}(t,t'') σ^β G^a_{jn}(t'',t) + G^r_{mj}(t,t'') σ^β G^<_{jn}(t'',t).

X (2x2) en la base de Pauli: X = ½ Tr[X] 𝕀 + ½ Σ_α Tr[σ^α X] σ^α. Con el kernel de siempre
χ^{αβ} = i J_sd Tr[σ^α X] y la componente de carga χ^{0β} = -J_sd Tr[X]:

    𝙂^<_{mn}(t,t) = G^<_{mn}(t,t) + ½ Σ_{j=1,2} Σ_β ∫dt'' [ χ^{0β}_{mn;j}(t,t'') 𝕀 + i Σ_α χ^{αβ}_{mn;j}(t,t'') σ^α ] M^β_j(t'')

(con m = n, χ^{αβ}_{mm;j} es el χ^{αβ}_{mj} de la LLG). Floquet, τ = t - t'':

    χ^{μβ}_{mn;j}(t,t-τ) = Σ_K e^{-iKΩt} χ̌^{μβ}_{mn;j,K}(τ),    K = k1 + k2 = 0, ±1, ±2,
    χ̌^{μβ}_{mn;j,K}(τ) = c_μ Σ_{k1+k2=K} e^{ik2Ωτ} Tr[ σ^μ ( 𝒢^<_{mj,k1}(τ) σ^β 𝒢^a_{jn,k2}(-τ) + 𝒢^r_{mj,k1}(τ) σ^β 𝒢^<_{jn,k2}(-τ) ) ],
    c_0 = -J_sd (σ^0 = 𝕀),  c_α = i J_sd,

calculado en τ con los 𝒢^μ_{mn,k}(τ) de coeff_tau (como kernel_tau de kernel_elements.jl). Para
m = n se compara con kernel_tau (las 6 componentes xx0 yy0 xz0 xy1 yx1 xx2) como control.
𝒢^a(-τ) y 𝒢^r(τ) se anulan para τ < 0 (analiticidad de ω + iη), asi que la integral va en
τ ∈ [0, τmax], por trapecio con paso dτ; en τ = 0 se usan los limites laterales
𝒢^r_{mm,0}(0+) = -i 𝕀, 𝒢^a_{mm,0}(0-) = +i 𝕀 (coeff_tau da el punto medio, -i/2 y +i/2).
M_j(t) = trayectorias de laboratorio de la LLG (spins_t_llg.csv), interpoladas (Catmull-Rom)
a la malla dτ.

Estado estacionario: la salida son los tiempos de la LLG en sus ultimos --last + 1 periodos
(el alineamiento con TDNEGF lo hace plot_bond_current_llg.py). La memoria [t - τmax, t] cae
entera en la parte final de la LLG (punto fijo del marco rotante); si no, el script se detiene.

Salida, con el mismo formato que bond_rho_t.csv de TDNEGF (sitios de TDNEGF, lead = C,
rho_nm = -i 𝙂^<_{nm}(t,t), rho_mn = -i 𝙂^<_{mn}(t,t)):
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

# ---------------------------------------------------------------- 𝒢^μ_{mn,k}(τ) como matriz 2x2
"Matriz 2x2 de 𝒢^μ_{mn,k}(τ) en la columna r de la tabla A de coeff_tau (μ = :r, :a, :l)."
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
            A[IAR, Nτ+1] += -im / 2                 # -i/2 -> 𝒢^r(0+) = -i
            A[IAA, Nτ+1] +=  im / 2                 # +i/2 -> 𝒢^a(0-) = +i
        end
        Ag[(m, n)] = A
        # control de causalidad (sale de ω + iη, no se impone): 𝒢^r(τ < 0) ≈ 0 y 𝒢^a(τ > 0) ≈ 0
        # (se excluye |τ| < 2, oscilaciones de Gibbs del corte en ω)
        τs = (-Nτ:Nτ) .* c.dτ
        neg = findall(<(-2.0), τs); pos = findall(>(2.0), τs)
        cr = maximum(abs, A[[IAR, IBR, IGPR, IGMR], neg]) / maximum(abs, A[[IAR, IBR, IGPR, IGMR], :])
        ca = maximum(abs, A[[IAA, IBA, IGPA, IGMA], pos]) / maximum(abs, A[[IAA, IBA, IGPA, IGMA], :])
        @printf("  𝒢^μ_(%d%d,k)(τ) en %.1f s   causalidad: max|𝒢^r(τ<-2)|/max|𝒢^r| = %.1e   max|𝒢^a(τ>2)|/max|𝒢^a| = %.1e\n",
                m, n, time() - t0, cr, ca)
    end
    return Ag, Nτ
end

# ---------------------------------------------------------------- kernel χ̌^{μβ}_{mn;j,K}(τ)
const PAULI4 = (σ0, σx, σy, σz)
"Indice de (μ, K) en las tablas del kernel: μ = 0 (𝕀), 1 (x), 2 (y), 3 (z); K = -2..2."
@inline ek(μ::Int, K::Int) = 1 + μ + 4 * (K + 2)

"""
    kernel_bond(Ag, m, n, sites, p, Nτ, dτ) -> χ

χ[ek(μ,K), s+1, β, j] = χ̌^{μβ}_{mn;j,K}(τ_s), τ_s = s dτ (s = 0..Nτ), j = indice del F en sites:

    χ̌^{μβ}_{mn;j,K}(τ) = c_μ Σ_{k1+k2=K} e^{ik2Ωτ} Tr[ σ^μ ( 𝒢^<_{mj,k1}(τ) σ^β 𝒢^a_{jn,k2}(-τ)
                                                         + 𝒢^r_{mj,k1}(τ) σ^β 𝒢^<_{jn,k2}(-τ) ) ],
    c_0 = -J_sd,  c_{x,y,z} = +i J_sd.
"""
function kernel_bond(Ag, m::Int, n::Int, sites, p::KParams, Nτ::Int, dτ::Float64)
    χ = zeros(ComplexF64, 20, Nτ + 1, 3, length(sites))
    cμ = (-p.J_sd, im * p.J_sd, im * p.J_sd, im * p.J_sd)
    for (ij, j) in enumerate(sites)
        Amj = Ag[(m, j)]; Ajn = Ag[(j, n)]
        @threads for s in 0:Nτ
            rp = Nτ + 1 + s; rm = Nτ + 1 - s
            for k1 in -1:1, k2 in -1:1
                Lp = gmat(Amj, :l, k1, rp); Rp = gmat(Amj, :r, k1, rp)
                Am = gmat(Ajn, :a, k2, rm); Lm = gmat(Ajn, :l, k2, rm)
                ph = cis(k2 * p.Ω * s * dτ)
                for (ib, sb) in enumerate(SIGS)
                    X = Lp * sb * Am + Rp * sb * Lm
                    for μ in 0:3
                        χ[ek(μ, k1 + k2), s+1, ib, ij] += cμ[μ+1] * ph * tr(PAULI4[μ+1] * X)
                    end
                end
            end
        end
    end
    return χ
end

"""
Control para m = n: las 6 componentes de kernel_tau de kernel_elements.jl (xx0 yy0 xz0 xy1 yx1 xx2,
mismas sumas tsums de los 𝒯_ab) frente a las de kernel_bond, en τ > 0. Devuelve el maximo
|diferencia| / max|χ̌| entre las 6 componentes.
"""
function check_kernel(Ag, m::Int, j::Int, ij::Int, χ, p::KParams, Nτ::Int, dτ::Float64)
    comp = ((1, 1, 0), (2, 2, 0), (1, 3, 0), (1, 2, 1), (2, 1, 1), (1, 1, 2))   # (μ, β, K)
    Amj = Ag[(m, j)]; Ajm = Ag[(j, m)]
    err = 0.0; scl = 0.0
    for s in 1:Nτ
        τ = s * dτ; ph = cis(p.Ω * τ)
        X = Amj[:, Nτ+1+s]; Y = Ajm[:, Nτ+1-s]
        tv = tsums(X, conj(ph) .* X, ph .* X, Y)
        for (c, (μ, β, K)) in enumerate(comp)
            ref = im * p.J_sd * tv[c]
            err = max(err, abs(χ[ek(μ, K), s+1, β, ij] - ref)); scl = max(scl, abs(ref))
        end
    end
    return err / scl
end

"""
𝙂^<_{mn}(t,t) en los tiempos de salida, G[:, :, q]:

    𝙂^<_{mn}(t,t) = Σ_k e^{-ikΩt} 𝒢^<_{mn,k}(0)
        + ½ Σ_{j,β,K} e^{-iKΩt} ∫_0^{τmax} dτ [ χ̌^{0β}_{mn;j,K}(τ) 𝕀 + i Σ_α χ̌^{αβ}_{mn;j,K}(τ) σ^α ] M^β_j(t-τ)

(trapecio). Mf[:, β, j] = M^β_j en la malla fina, base[q] = indice de t_q en esa malla,
tF = tiempos de Floquet de la salida.
"""
function dressed_lesser(Ag, m::Int, n::Int, sites, p::KParams, Nτ::Int, dτ::Float64,
                        Mf::Array{Float64,3}, base::Vector{Int}, tF::Vector{Float64}; check::Bool = false)
    χ = kernel_bond(Ag, m, n, sites, p, Nτ, dτ)
    if check && m == n
        for (ij, j) in enumerate(sites)
            @printf("  control kernel χ̌_(%d%d) frente a kernel_tau: max|Δ|/max|χ̌| = %.1e\n",
                    m, j, check_kernel(Ag, m, j, ij, χ, p, Nτ, dτ))
        end
    end
    w = fill(dτ, Nτ + 1); w[1] = w[end] = dτ / 2
    A0 = Ag[(m, n)]
    G0k = [gmat(A0, :l, k, Nτ + 1) for k in -1:1]
    nt = length(base)
    G = Array{ComplexF64}(undef, 2, 2, nt)
    @threads for q in 1:nt
        acc = zeros(ComplexF64, 20)                      # ∫dτ χ̌^{μβ}_{K} M^β, sumado en β y j
        for ij in eachindex(sites), ib in 1:3
            for s in 0:Nτ
                mv = w[s+1] * Mf[base[q]-s, ib, ij]
                @inbounds @simd for e in 1:20
                    acc[e] += χ[e, s+1, ib, ij] * mv
                end
            end
        end
        g = zeros(ComplexF64, 2, 2)
        for k in -1:1
            g .+= cis(-k * p.Ω * tF[q]) .* G0k[k+2]
        end
        for K in -2:2
            ph = cis(-K * p.Ω * tF[q])
            g .+= (ph / 2) .* (acc[ek(0, K)] .* σ0 .+ im .* (acc[ek(1, K)] .* σx .+ acc[ek(2, K)] .* σy .+ acc[ek(3, K)] .* σz))
        end
        G[:, :, q] = g
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

    # ---- ventana: ultimos (--last + 1) periodos de la LLG. El periodo extra deja que
    # plot_bond_current_llg.py elija dentro los --last periodos corridos un numero entero de
    # periodos respecto del final de TDNEGF (no depende de la duracion de TDNEGF).
    L = readdlm(joinpath(llgdir, "spins_t_llg.csv"), ',', Float64; skipstart = 1)
    tL = L[:, 1]
    t_lo = tL[end] - (parse(Float64, opt["--last"]) + 1) * T
    sel = findall(≥(t_lo - 1e-9), tL)
    t_out = tL[sel]
    Nτ = ceil(Int, tmaxof(p, c) / dτ)
    g0 = t_out[1] - Nτ * dτ
    g0 ≥ tL[2] || error("la memoria [t - τmax, t] se sale de la LLG (baja --last o --taumax)")
    @printf("  ventana de la LLG: [%.1f, %.1f] (%.1f T), memoria desde %.1f (%.1f T despues del inicio de la LLG)\n",
            t_out[1], t_out[end], (t_out[end] - t_out[1]) / T, g0, (g0 - tL[1]) / T)
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

    # ---- funciones de Green y 𝙂^< vestida
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
        G[(m, n)] = dressed_lesser(Ag, m, n, sites, p, Nτ, dτ, Mf, base, tF; check = true)
    end
    @printf("  𝙂^< vestida (4 pares, %d tiempos) en %.1f s\n", length(t_out), time() - t0)

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
