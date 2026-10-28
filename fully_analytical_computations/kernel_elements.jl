#!/usr/bin/env julia
#=
kernel_elements.jl -- armonicos de Floquet k = 0, ±1 de las funciones de Green
del lead DERECHO (sitios m, n ≥ 1) para el driver precesando alrededor de y,
implementados directamente con las formulas de la seccion "Overview":

    𝒢^μ_{mn,0}(ω)  = α^μ_{mn} σ0 + β^μ_{mn} σy
    𝒢^μ_{mn,±1}(ω) = γ^μ_{mn,±} τ±,          τ± = (σz ± iσx)/2,   μ = r, a, <

con
    G^μ_{mn}(t,t') = Σ_k e^{-ikΩt} ∫dω/2π e^{-iω(t-t')} 𝒢^μ_{mn,k}(ω).

Este archivo es una BIBLIOTECA: ninguna funcion depende del sitio ni del numero
de sitios. Lo que depende solo de ω (centro, g₁₁ en ω y ω±Ω, tablas de
potencias) se calcula una vez por ω y se reutiliza para todos los pares (m,n)
pedidos; cada par cuesta O(1).

Uso como biblioteca:
    include("kernel_elements.jl")
    p = read_params(".../params.txt")            # o KParams(...)
    c = lead_coeffs(0.3, 1, 2, p; η = 1e-3)      # NamedTuple con los 12 escalares
    H = harmonics_grid(ωs, [(1,1), (1,2)], p; η = 1e-3)   # (Nω, npares, 12)
    G = Gmat(c, :r, 1)                            # matriz 2x2 de 𝒢^r_{mn,+1}
    for J in 0.1:0.1:0.5                          # barrido de cualquier parametro
        H = harmonics_grid(ωs, [(1,1)], with(p; J_sd = J); η = 1e-3)
    end

Uso desde la linea de comandos (lo llama plot_kernel_elements.py):
    julia -t 8 kernel_elements.jl --params RUTA/params.txt --pairs "1,1;1,2" \
          --wmin -2.5 --wmax 2.5 --nw 20001 [--eta 5e-4] [--out salida.csv]
    (sin --out: output/<tag>/data/kernel_elements_<pares>.csv, tag = carpeta del params.txt)
    Cualquier parametro se puede cambiar sobre los del params.txt (θ en grados):
          --gamma --gso --jsd --theta --Omega --mu --beta
    y entonces tag = gam.._gso.._jsd.._th..deg_Om.._mu.._b.. (los valores usados).
    Con --params none se parte de KParams() en vez de un params.txt.
    julia kernel_elements.jl --validate [--params ...]   # compara con la implementacion
                                                         # matricial ya validada contra Sambe
    julia -t auto kernel_elements.jl --kernel --pairs "1,1"   # kernel χ̌(ω) e i∂_ωχ̌(ω)
          [--k-nomega 40] [--k-eta η] [--k-wide 6.0] [--k-nwide 1601] [--k-narrow 2.5]
          [--k-margin 2.0] [--datadir DIR]
      dω = Ω/nomega, η (defecto 5dω); ventana ancha ±wide γ̃ con nwide puntos y
      estrecha ±narrow Ω con paso dω. Escribe kernel_chi_m{i}n{j}.csv y el resumen
      kernel_chi0_kernel_<pares>.csv (χ̌(0), i∂_ωχ̌(0)) en DIR (defecto output/<tag>/data).
      (el kernel_chi0_<pares>.csv que lee llg_elements.jl lo escribe --chi0)
    julia -t auto kernel_elements.jl --tau --pairs "1,1"     # kernel en tiempo χ̌_k(τ)
          [--t-dtau 0.1] [--t-taumax T] [--t-log2n 22] [--t-eta 5e-5] [--datadir DIR]
      FFT de los armonicos analiticos (ω ∈ [-π/dτ, π/dτ), N = 2^log2n) y producto en τ.
      Escribe kernel_tau_m{i}n{j}.csv (τ ∈ [-τmax, τmax], 6 componentes) en DIR.
    julia -t auto kernel_elements.jl --chi0 --sites 10       # solo χ̌(0), i∂χ̌(0), pares [1,10]^2
          [--pairs "..."] [--k-nomega 40] [--k-eta η] [--k-margin 2.0] [--datadir DIR]
      Escribe kernel_chi0_sites1-10.csv (o kernel_chi0_<pares>.csv) para llg_elements.jl.
    julia -t auto kernel_elements.jl --sbar --sites 10       # s̄_m = ⟨σ⟩_{0,m}(t=0), m = 1..10
          [--k-nomega 40] [--k-margin 2.0] [--datadir DIR]
      Integra en ω los armonicos lesser de 𝒢_{mm} (k = 0, ±1), con nΩ y 2nΩ, y extrapola
      a η -> 0. Escribe sbar_sites1-10.csv (m, sx, sy, sz, n) para la LLG de los F.
      Para los 8 conjuntos: bash run_sbar.sh
    --sites N vale para --kernel, --tau y --chi0 (todos los pares de [1,N]^2) y --sbar (sitios 1..N).
    --tag NOMBRE fija la carpeta output/NOMBRE (si no, la de params.txt o la de los valores).

Orden de las 12 columnas complejas (COEFS):
    alpha_r beta_r gp_r gm_r  alpha_a beta_a gp_a gm_a  alpha_l beta_l gp_l gm_l
(gp = γ₊, gm = γ₋, l = lesser).
=#

using LinearAlgebra
using Printf
using DelimitedFiles
using Base.Threads

# ---------------------------------------------------------------- parametros
"""
Parametros del modelo (mismos nombres que el texto):
    γ, γSO  hopping y Rashba del lead,   γ̃ = sqrt(γ² + γSO²),  tan θ_SO = γSO/γ
    J_sd    acoplamiento s-d del driver, θ angulo del cono,  Ω frecuencia
    μ, β    potencial quimico e inverso de la temperatura (comunes a todos los reservorios)
"""
Base.@kwdef struct KParams
    γ::Float64    = sqrt(1 - 0.1^2)
    γSO::Float64  = 0.1
    J_sd::Float64 = 0.2
    θ::Float64    = deg2rad(10.0)
    Ω::Float64    = 0.005
    μ::Float64    = 0.0
    β::Float64    = 40.0
end

γt(p::KParams)  = hypot(p.γ, p.γSO)
θSO(p::KParams) = atan(p.γSO, p.γ)
fermi(ω::Real, p::KParams) = 1 / (1 + exp(clamp(p.β * (ω - p.μ), -700.0, 700.0)))

"""
Lee los parametros del params.txt de una corrida de TDNEGF (pares "clave = valor").
Exige precesion alrededor de y y PREC_SIGN = 1 (el sentido de las formulas).
"""
function read_params(path::AbstractString)
    txt = read(path, String)
    d = Dict{String,Float64}()
    for m in eachmatch(r"(\S+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", txt)
        d[m.captures[1]] = parse(Float64, m.captures[2])
    end
    ax = match(r"eje de precesion\s*=\s*(\w)", txt)
    (ax === nothing || ax.captures[1] == "y") ||
        error("$path: las formulas son para precesion alrededor de y (eje = $(ax.captures[1]))")
    get(d, "PREC_SIGN", 1.0) == 1.0 || error("$path: PREC_SIGN = $(d["PREC_SIGN"]) no soportado")
    for k in ("γ", "γso", "j_sd", "θ_max", "Ω", "E_F", "β")
        haskey(d, k) || error("$path: falta la clave '$k'")
    end
    return KParams(; γ = d["γ"], γSO = d["γso"], J_sd = d["j_sd"], θ = deg2rad(d["θ_max"]),
                     Ω = d["Ω"], μ = d["E_F"], β = d["β"])
end

# ---------------------------------------------------------------- derivada exacta en ω
"""
Jet: numero (valor, derivada) respecto de ω, f(ω) -> (f, ∂_ω f). Las formulas de
los armonicos se evaluan con ω = Jet(ω, 1) y dan su derivada exacta (diferenciacion
automatica en modo directo, sin dependencias). Solo hacen falta +, -, *, /, conj,
real, imag, abs2 y las dos funciones no polinomiales (Sʳ y la Fermi), que tienen
metodos propios. Valores reales se guardan como complejos con parte imaginaria 0.
"""
struct Jet <: Number
    v::ComplexF64
    d::ComplexF64
end
Base.convert(::Type{Jet}, x::Jet) = x
Base.convert(::Type{Jet}, x::Number) = Jet(ComplexF64(x), zero(ComplexF64))
Base.promote_rule(::Type{Jet}, ::Type{<:Number}) = Jet
# Operaciones mixtas explicitas (Jet con reales/complejos): evitan pasar por la
# maquinaria de promote en cada operacion, lo que acorta mucho la inferencia de tipos.
const RC = Union{Real,Complex}
Base.:+(a::Jet, b::RC) = Jet(a.v + b, a.d)
Base.:+(a::RC, b::Jet) = Jet(a + b.v, b.d)
Base.:-(a::Jet, b::RC) = Jet(a.v - b, a.d)
Base.:-(a::RC, b::Jet) = Jet(a - b.v, -b.d)
Base.:*(a::Jet, b::RC) = Jet(a.v * b, a.d * b)
Base.:*(a::RC, b::Jet) = Jet(a * b.v, a * b.d)
Base.:/(a::Jet, b::RC) = Jet(a.v / b, a.d / b)
Base.:/(a::RC, b::Jet) = Jet(a / b.v, -a * b.d / (b.v * b.v))
Base.:+(a::Jet, b::Jet) = Jet(a.v + b.v, a.d + b.d)
Base.:-(a::Jet, b::Jet) = Jet(a.v - b.v, a.d - b.d)
Base.:-(a::Jet) = Jet(-a.v, -a.d)
Base.:*(a::Jet, b::Jet) = Jet(a.v * b.v, a.v * b.d + a.d * b.v)
Base.:/(a::Jet, b::Jet) = Jet(a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v))
Base.conj(a::Jet) = Jet(conj(a.v), conj(a.d))
Base.real(a::Jet) = Jet(real(a.v), real(a.d))
Base.imag(a::Jet) = Jet(imag(a.v), imag(a.d))
Base.abs2(a::Jet) = Jet(abs2(a.v), 2 * real(conj(a.v) * a.d))
Base.zero(::Type{Jet}) = Jet(0, 0)
Base.one(::Type{Jet}) = Jet(1, 0)
Base.zero(::Jet) = Jet(0, 0)
Base.one(::Jet) = Jet(1, 0)
value(x::Jet) = x.v
deriv(x::Jet) = x.d

"ω + iη (con su derivada si ω es un Jet)."
zη(ω::Real, η::Real) = complex(ω, η)
zη(ω::Jet, η::Real)  = Jet(ω.v + im * η, ω.d)

"Fermi con derivada exacta: ∂_ω f = -β f (1 - f)."
function fermi(ω::Jet, p::KParams)
    f = fermi(real(ω.v), p)
    return Jet(f, -p.β * f * (1 - f) * ω.d)
end

# ---------------------------------------------------------------- S, g₁₁
"Sʳ(ω) = sqrt((ω+iη)² - 4γ̃²) en la rama con Im Sʳ ≥ 0."
@inline function Sr(ω::Real, p::KParams, η::Real)
    z = complex(ω, η); b = 2γt(p)
    return sqrt(z - b) * sqrt(z + b)
end

"Sʳ con derivada: de S² = z² - 4γ̃², ∂_ω S = z/S."
@inline function Sr(ω::Jet, p::KParams, η::Real)
    w = real(ω.v)
    s = Sr(w, p, η)
    return Jet(s, complex(w, η) / s * ω.d)
end

"g₁₁ʳ(ω) = [(ω+iη) - Sʳ(ω)]/(2γ̃²)."
@inline g11r(ω, p::KParams, η::Real) = (zη(ω, η) - Sr(ω, p, η)) / (2γt(p)^2)

# ---------------------------------------------------------------- centro
"""
Coeficientes de los armonicos del sitio central en ω (solo dependen de ω):
    𝒢^μ_{C,0} = A0 σ0 + B0 σy,   𝒢^μ_{C,+1} = Ap τ+,   𝒢^μ_{C,-1} = Am τ-
para μ = r, a (= conjugado de r) y < (formulas de X<↑, X<↓, A<_{C,±1}).
"""
struct Central{T}
    A0r::T; B0r::T; Apr::T; Amr::T
    A0a::T; B0a::T; Apa::T; Ama::T
    A0l::T; B0l::T; Apl::T; Aml::T
end

function central(ω, p::KParams, η::Real)
    Ω  = p.Ω
    Jc = p.J_sd * cos(p.θ); Js = p.J_sd * sin(p.θ); J2 = p.J_sd^2
    Sm, S0, Sp = Sr(ω - Ω, p, η), Sr(ω, p, η), Sr(ω + Ω, p, η)
    Δ0  = S0 * Sp + Jc * (S0 - Sp) - J2              # Δ₀ʳ(ω)
    Δ0m = Sm * S0 + Jc * (Sm - S0) - J2              # Δ₀ʳ(ω-Ω)

    # retardada
    x   = (Sm - Jc) / Δ0m
    y   = (Sp + Jc) / Δ0
    A0r = (x + y) / 2; B0r = (x - y) / 2
    Apr = -Js / Δ0;    Amr = -Js / Δ0m

    # lesser
    fm, f0, fp = fermi(ω - Ω, p), fermi(ω, p), fermi(ω + Ω, p)
    Im_m, Im_0, Im_p = imag(Sm), imag(S0), imag(Sp)
    Xup = 2im / abs2(Δ0m) * (f0 * Im_0 * abs2(Sm - Jc) + fm * Im_m * Js^2)
    Xdn = 2im / abs2(Δ0)  * (f0 * Im_0 * abs2(Sp + Jc) + fp * Im_p * Js^2)
    Apl = -2im * Js / abs2(Δ0)  * (f0 * Im_0 * (conj(Sp) + Jc) + fp * Im_p * (S0 - Jc))
    Aml = -2im * Js / abs2(Δ0m) * (f0 * Im_0 * (conj(Sm) - Jc) + fm * Im_m * (S0 + Jc))

    T = typeof(A0r)
    return Central{T}(A0r, B0r, Apr, Amr,
                      conj(A0r), conj(B0r), conj(Apr), conj(Amr),
                      (Xup + Xdn) / 2, (Xup - Xdn) / 2, Apl, Aml)
end

# ---------------------------------------------------------------- tablas por ω
"""
Todo lo del lead que depende solo de ω, tabulado hasta el sitio Mmax:
    g0[s+1] = [g₁₁ʳ(ω)]^s,  gp[s+1] = [g₁₁ʳ(ω+Ω)]^s,  gm[s+1] = [g₁₁ʳ(ω-Ω)]^s   (s = 0..Mmax)
    geo[L]  = Σ_{p=0}^{L-1} [γ̃ g₁₁ʳ(ω)]^{2p}                                      (L = 1..Mmax)
    mγ[s+1] = (-γ̃)^s                                                              (s = 0..2Mmax)
"""
struct LeadTables{T,R}
    g0::Vector{T}; gp::Vector{T}; gm::Vector{T}
    geo::Vector{T}; mγ::Vector{Float64}
    f0::R; fp::R; fm::R
end

function lead_tables(ω, p::KParams, η::Real, Mmax::Int)
    pw(g) = (v = Vector{typeof(g)}(undef, Mmax + 1); v[1] = one(g); for s in 1:Mmax; v[s+1] = v[s] * g; end; v)
    g0 = g11r(ω, p, η)
    q  = (γt(p) * g0)^2
    geo = Vector{typeof(g0)}(undef, Mmax)
    acc = zero(g0); t = one(g0)
    for L in 1:Mmax
        acc += t; geo[L] = acc; t *= q
    end
    mγ = [(-γt(p))^s for s in 0:2Mmax]
    return LeadTables(pw(g0), pw(g11r(ω + p.Ω, p, η)), pw(g11r(ω - p.Ω, p, η)), geo, mγ,
                      fermi(ω, p), fermi(ω + p.Ω, p), fermi(ω - p.Ω, p))
end

# ---------------------------------------------------------------- lead (m,n)
const COEFS = ("alpha_r", "beta_r", "gp_r", "gm_r",
               "alpha_a", "beta_a", "gp_a", "gm_a",
               "alpha_l", "beta_l", "gp_l", "gm_l")

"""
Los 12 escalares (α^μ, β^μ, γ^μ_±; μ = r, a, <) del par (m,n), m,n ≥ 1, a partir
de las tablas de ω (T), el centro (C) y los parametros. Formulas de la seccion
Overview, termino a termino.
"""
function lead_coeffs(T::LeadTables, C::Central, m::Int, n::Int, p::KParams)
    (m ≥ 1 && n ≥ 1) || throw(ArgumentError("sitios del lead: m, n ≥ 1"))
    θs  = θSO(p)
    d   = abs(m - n)
    cs, sn = cos((n - m) * θs), sin((n - m) * θs)
    pre = T.mγ[m+n+1]                                   # (-γ̃)^{m+n}

    # lead libre: c^r_{mn}, c^a = conj, g< ∝ Im c^r
    cr = T.mγ[d+1] * T.g0[d+2] * T.geo[min(m, n)]
    ca = conj(cr)
    cl = -2im * T.f0 * imag(cr)                         # g<_{mn} = cl e^{i(n-m)θ_SO σy}

    g0m, g0n = T.g0[m+1], T.g0[n+1]
    imn = imag(g0n)                                     # Im{[g₁₁ʳ(ω)]^n}

    # P_k para k = 0, +1, -1 (gk = g₁₁ʳ(ω+kΩ), fk = f(ω+kΩ))
    @inline function Ps(gkm, fk)
        rr = pre * gkm * g0n
        aa = conj(rr)
        la = -2im * fk * pre * imag(gkm) * conj(g0n)
        ra = pre * gkm * conj(g0n)
        rl = -2im * T.f0 * pre * gkm * imn
        return (rr = rr, aa = aa, la = la, ra = ra, rl = rl)
    end
    P0 = Ps(g0m, T.f0)
    Pp = Ps(T.gp[m+1], T.fp)
    Pm = Ps(T.gm[m+1], T.fm)

    # bloques [A cos + i B sin] y [B cos + i A sin]
    a_(A, B) = A * cs + im * B * sn
    b_(A, B) = B * cs + im * A * sn
    php = cis(-(m + n) * θs)                            # e^{-i(m+n)θ_SO}  (k = +1)
    phm = cis(+(m + n) * θs)                            # e^{+i(m+n)θ_SO}  (k = -1)

    αr = cr * cs      + P0.rr * a_(C.A0r, C.B0r)
    βr = im * cr * sn + P0.rr * b_(C.A0r, C.B0r)
    γpr = php * Pp.rr * C.Apr
    γmr = phm * Pm.rr * C.Amr

    αa = ca * cs      + P0.aa * a_(C.A0a, C.B0a)
    βa = im * ca * sn + P0.aa * b_(C.A0a, C.B0a)
    γpa = php * Pp.aa * C.Apa
    γma = phm * Pm.aa * C.Ama

    αl = cl * cs + P0.la * a_(C.A0a, C.B0a) + P0.ra * a_(C.A0l, C.B0l) + P0.rl * a_(C.A0r, C.B0r)
    βl = im * cl * sn + P0.la * b_(C.A0a, C.B0a) + P0.ra * b_(C.A0l, C.B0l) + P0.rl * b_(C.A0r, C.B0r)
    γpl = php * (Pp.la * C.Apa + Pp.ra * C.Apl + Pp.rl * C.Apr)
    γml = phm * (Pm.la * C.Ama + Pm.ra * C.Aml + Pm.rl * C.Amr)

    return (alpha_r = αr, beta_r = βr, gp_r = γpr, gm_r = γmr,
            alpha_a = αa, beta_a = βa, gp_a = γpa, gm_a = γma,
            alpha_l = αl, beta_l = βl, gp_l = γpl, gm_l = γml)
end

"Version de un solo punto: los 12 escalares del par (m,n) en ω."
lead_coeffs(ω::Real, m::Int, n::Int, p::KParams; η::Real) =
    lead_coeffs(lead_tables(ω, p, η, max(m, n)), central(ω, p, η), m, n, p)

# ---------------------------------------------------------------- matrices 2x2
const σ0 = ComplexF64[1 0; 0 1]
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const τp = (σz + im * σx) / 2
const τm = (σz - im * σx) / 2

"""
Matriz 2x2 (base σz) de 𝒢^μ_{mn,k} a partir de los escalares c (μ = :r, :a, :l).
"""
function Gmat(c, μ::Symbol, k::Int)
    s = μ === :l ? "l" : String(μ)
    k ==  0 && return getfield(c, Symbol("alpha_", s)) * σ0 + getfield(c, Symbol("beta_", s)) * σy
    k ==  1 && return getfield(c, Symbol("gp_", s)) * τp
    k == -1 && return getfield(c, Symbol("gm_", s)) * τm
    return zeros(ComplexF64, 2, 2)
end

# ---------------------------------------------------------------- malla de ω
"""
    harmonics_grid(ωs, pairs, p; η) -> Array{ComplexF64}(Nω, length(pairs), 12)

Los 12 escalares (orden COEFS) para cada ω de la malla y cada par (m,n) pedido.
Centro y tablas se calculan una vez por ω; el bucle en ω va en hilos.
"""
function harmonics_grid(ωs::AbstractVector{<:Real}, pairs::AbstractVector{Tuple{Int,Int}},
                        p::KParams; η::Real)
    isempty(pairs) && throw(ArgumentError("lista de pares vacia"))
    Mmax = maximum(max(m, n) for (m, n) in pairs)
    H = Array{ComplexF64}(undef, length(ωs), length(pairs), length(COEFS))
    @threads for iω in eachindex(ωs)
        ω = ωs[iω]
        C = central(ω, p, η)
        T = lead_tables(ω, p, η, Mmax)
        for (ip, (m, n)) in enumerate(pairs)
            c = lead_coeffs(T, C, m, n, p)
            for (ic, v) in enumerate(c)
                H[iω, ip, ic] = v
            end
        end
    end
    return H
end

# ---------------------------------------------------------------- kernel
#=
Componentes independientes del kernel (seccion "Kernel Floquet harmonics"):

    χ̌_{ij,k}(ω) = +i J_sd ∫dω'/2π  Σ 𝒯_{ab}(ω,ω'),   𝒯 = f(armonicos_ij(ω+ω'+bΩ), armonicos_ji(ω'))

El prefactor es +i J_sd: ⟨σ⟩ = -i Tr[σ 𝒢^<] y, a primer orden, 𝒢^< = G^< - J_sd ∫[...],
asi que δ⟨σ⟩ = (+i J_sd) Tr[...] (respuesta fisica: χ_mm(0) > 0 on-site). Los CSV del
kernel llevan chi_sign=1 en el encabezado; los que no lo tienen se calcularon con el
prefactor viejo -i J_sd y se corrigen con fix_kernel_sign.py.

con las 10 cantidades auxiliares 𝒯. Malla: ω' = q dω con dω = Ω/nΩ (nΩ entero),
asi que ω+ω'+bΩ cae en la malla si ω = s dω: todo son desplazamientos de indice,
sin interpolar. Los armonicos ij se tabulan con su derivada exacta (Jet); los ji
solo en valor. i ∂_ω χ̌ sale de la misma suma con la derivada del factor ij.
=#

const KCOMP = ("xx0", "yy0", "xz0", "xy1", "yx1", "xx2")      # χ^{xx}_0 ... χ^{xx}_2
const IAR, IBR, IGPR, IGMR = 1, 2, 3, 4                        # indices en COEFS
const IAA, IBA, IGPA, IGMA = 5, 6, 7, 8
const IAL, IBL, IGPL, IGML = 9, 10, 11, 12

"""
Tabla (12, Nq) de los escalares del par (m,n) en ω = q dω, q ∈ qs. Con
withder = true devuelve tambien la tabla de ∂_ω (diferenciacion exacta con Jet).
"""
function coeff_table(qs::AbstractVector{Int}, dω::Real, m::Int, n::Int, p::KParams;
                     η::Real, withder::Bool)
    N = length(qs); Mmax = max(m, n)
    V = Matrix{ComplexF64}(undef, 12, N)
    D = withder ? Matrix{ComplexF64}(undef, 12, N) : Matrix{ComplexF64}(undef, 0, 0)
    # primera llamada en el hilo principal: compila aqui (pila grande) y no dentro de
    # las tareas de @threads
    coeff_point!(V, D, 1, qs[1] * dω, m, n, p, η, Mmax, withder)
    @threads for iq in 1:N
        coeff_point!(V, D, iq, qs[iq] * dω, m, n, p, η, Mmax, withder)
    end
    return V, D
end

"Columna iq de las tablas: los 12 escalares en ω (y su derivada si withder)."
@noinline function coeff_point!(V, D, iq::Int, ω::Float64, m::Int, n::Int, p::KParams,
                                η::Real, Mmax::Int, withder::Bool)
    if withder
        x = Jet(ω, 1.0)
        c = lead_coeffs(lead_tables(x, p, η, Mmax), central(x, p, η), m, n, p)
        for (ic, v) in enumerate(c)
            V[ic, iq] = v.v; D[ic, iq] = v.d
        end
    else
        c = lead_coeffs(lead_tables(ω, p, η, Mmax), central(ω, p, η), m, n, p)
        for (ic, v) in enumerate(c)
            V[ic, iq] = v
        end
    end
    return nothing
end

"""
Las 6 sumas Σ_{ab} 𝒯_{ab} (orden KCOMP) en un punto, con X0, Xm, Xp = columna de
los escalares ij en ω+ω', ω+ω'-Ω, ω+ω'+Ω (o sus derivadas) e Y = columna ji en ω'.
Cada 𝒯 es lineal en el factor ij, asi que la misma funcion sirve para ∂_ω.
"""
@inline function tsums(X0, Xm, Xp, Y)
    αaY, βaY, αlY, βlY = Y[IAA], Y[IBA], Y[IAL], Y[IBL]
    αl, βl, αr, βr = X0[IAL], X0[IBL], X0[IAR], X0[IBR]
    t_xx0 = 2 * (αl * αaY - βl * βaY + αr * αlY - βr * βlY)
    t_yy0 = 2 * (αl * αaY + βl * βaY + αr * αlY + βr * βlY) -
            (Xm[IGPL] * Y[IGMA] + Xm[IGPR] * Y[IGML]) -                   # 𝒯^{yy}_{1,-1}
            (Xp[IGML] * Y[IGPA] + Xp[IGMR] * Y[IGPL])                     # 𝒯^{yy}_{-1,1}
    t_xz0 = 2im * (βl * αaY - αl * βaY + βr * αlY - αr * βlY)
    t_xy1 = -im * (X0[IGPL] * (αaY - βaY) + X0[IGPR] * (αlY - βlY)) +     # 𝒯^{xy}_{1,0}
             im * ((Xp[IAL] + Xp[IBL]) * Y[IGPA] + (Xp[IAR] + Xp[IBR]) * Y[IGPL])   # 𝒯^{xy}_{0,1}
    t_yx1 =  im * (X0[IGPL] * (αaY + βaY) + X0[IGPR] * (αlY + βlY)) -     # 𝒯^{yx}_{1,0}
             im * ((Xp[IAL] - Xp[IBL]) * Y[IGPA] + (Xp[IAR] - Xp[IBR]) * Y[IGPL])   # 𝒯^{yx}_{0,1}
    t_xx2 = -(Xp[IGPL] * Y[IGPA] + Xp[IGPR] * Y[IGPL])                    # 𝒯^{xx}_{1,1}
    return (t_xx0, t_yy0, t_xz0, t_xy1, t_yx1, t_xx2)
end

"""
Configuracion numerica del kernel:
    nΩ      dω = Ω/nΩ (malla comun de ω' y de los desplazamientos)
    η       ensanchamiento (por defecto 5 dω)
    wide    semiancho de la ventana ancha en unidades de γ̃;  nwide puntos
    narrow  semiancho de la ventana estrecha en unidades de Ω (paso dω)
    margin  margen bajo el estado ligado inferior para la integral en ω'
"""
Base.@kwdef struct KernelConfig
    nΩ::Int         = 40
    η::Float64      = NaN
    wide::Float64   = 6.0
    nwide::Int      = 1601
    narrow::Float64 = 2.5
    margin::Float64 = 2.0
end
kdω(p::KParams, c::KernelConfig) = p.Ω / c.nΩ
keta(p::KParams, c::KernelConfig) = isnan(c.η) ? 5 * kdω(p, c) : c.η

"Indices s (ω = s dω) de las dos ventanas: ancha (en γ̃) y estrecha (en Ω)."
function kernel_windows(p::KParams, c::KernelConfig)
    dω = kdω(p, c)
    K  = (c.nwide - 1) ÷ 2
    st = max(1, round(Int, c.wide * γt(p) / (K * dω)))
    sw = collect(-K*st:st:K*st)
    Kn = round(Int, c.narrow * c.nΩ)
    sn = collect(-Kn:Kn)
    return sw, sn
end

"""
    kernel_chi(i, j, p, c; svals) -> (χ, iχ')

χ̌ y i ∂_ω χ̌ de las 6 componentes (KCOMP) del par (i,j) para ω = s dω, s ∈ svals.
Devuelve dos matrices (length(svals), 6).
"""
function kernel_chi(i::Int, j::Int, p::KParams, c::KernelConfig; svals::AbstractVector{Int})
    dω = kdω(p, c); η = keta(p, c); nΩ = c.nΩ
    smin, smax = extrema(svals)
    # soporte de los factores lesser: [Lmin, Lmax] (banda + estados ligados + Fermi)
    Lmin = -(sqrt(4γt(p)^2 + p.J_sd^2) + c.margin)
    Lmax = p.μ + (isfinite(p.β) ? 40 / p.β : 0.0) + 2p.Ω
    jmin = floor(Int, (Lmin - p.Ω) / dω) - smax - 1           # ω' de la integral
    jmax = ceil(Int, (Lmax + p.Ω) / dω) - smin + 1
    qmin = jmin + smin - nΩ; qmax = jmax + smax + nΩ          # ω+ω'+bΩ
    Vij, Dij = coeff_table(qmin:qmax, dω, i, j, p; η = η, withder = true)
    Vji, _   = coeff_table(jmin:jmax, dω, j, i, p; η = η, withder = false)
    pre  = im * p.J_sd * dω / (2π)                       # +i J_sd
    χ    = zeros(ComplexF64, length(svals), 6)
    iχp  = zeros(ComplexF64, length(svals), 6)
    kernel_point!(χ, iχp, 1, svals[1], Vij, Dij, Vji, jmin, jmax, qmin, nΩ, pre)   # compila en el hilo principal
    @threads for is in eachindex(svals)
        kernel_point!(χ, iχp, is, svals[is], Vij, Dij, Vji, jmin, jmax, qmin, nΩ, pre)
    end
    return χ, iχp
end

"Integral en ω' para un solo ω = s dω (fila is de χ e iχp)."
@noinline function kernel_point!(χ, iχp, is::Int, s::Int, Vij, Dij, Vji,
                                 jmin::Int, jmax::Int, qmin::Int, nΩ::Int, pre::ComplexF64)
    acc = zeros(ComplexF64, 6); accd = zeros(ComplexF64, 6)
    for jj in jmin:jmax
        q  = jj + s - qmin + 1                                     # columna de ω+ω'
        Y  = view(Vji, :, jj - jmin + 1)
        tv = tsums(view(Vij, :, q), view(Vij, :, q - nΩ), view(Vij, :, q + nΩ), Y)
        td = tsums(view(Dij, :, q), view(Dij, :, q - nΩ), view(Dij, :, q + nΩ), Y)
        for k in 1:6
            acc[k] += tv[k]; accd[k] += td[k]
        end
    end
    for k in 1:6
        χ[is, k]   = pre * acc[k]
        iχp[is, k] = im * pre * accd[k]
    end
    return nothing
end

"""
Escribe el kernel de varios pares:
    <datadir>/kernel_chi_m{i}n{j}.csv  ventana, ω, Re/Im de χ̌ y de i∂_ωχ̌ (6 componentes)
    <datadir>/kernel_chi0_<pares>.csv  χ̌(0) e i∂_ωχ̌(0) de todos los pares
Imprime la comprobacion de realidad de k = 0: Im χ̌_0(0) = 0 e Im[i∂χ̌_0(0)] = 0.
"""
function kernel_run(pairs, p::KParams, c::KernelConfig, datadir::AbstractString, label::AbstractString)
    mkpath(datadir)
    dω = kdω(p, c); η = keta(p, c)
    sw, sn = kernel_windows(p, c)
    svals = sort(unique(vcat(sw, sn)))
    pos = Dict(s => k for (k, s) in enumerate(svals))
    meta = @sprintf("# gamma=%.15g gammaSO=%.15g J_sd=%.15g theta=%.15g Omega=%.15g mu=%.15g beta=%.15g eta=%.15g nOmega=%d wide=%.15g nwide=%d narrow=%.15g margin=%.15g chi_sign=1\n",
                    p.γ, p.γSO, p.J_sd, p.θ, p.Ω, p.μ, p.β, η, c.nΩ, c.wide, c.nwide, c.narrow, c.margin)
    rows0 = String[]
    @printf("kernel: dω=%.3e (Ω/%d)  η=%.3e  ventana ancha ±%.2f γ̃ (%d pts)  estrecha ±%.2f Ω (%d pts)\n",
            dω, c.nΩ, η, c.wide, length(sw), c.narrow, length(sn))
    for (i, j) in pairs
        t0 = time()
        χ, iχp = kernel_chi(i, j, p, c; svals = svals)
        path = joinpath(datadir, "kernel_chi_m$(i)n$(j).csv")
        open(path, "w") do io
            print(io, meta)
            print(io, "window,omega")
            for k in KCOMP
                print(io, ",Re_chi_", k, ",Im_chi_", k, ",Re_idchi_", k, ",Im_idchi_", k)
            end
            println(io)
            for (lab, ss) in (("gamma", sw), ("Omega", sn)), s in ss
                r = pos[s]
                @printf(io, "%s,%.12e", lab, s * dω)
                for k in 1:6
                    @printf(io, ",%.12e,%.12e,%.12e,%.12e", real(χ[r, k]), imag(χ[r, k]),
                            real(iχp[r, k]), imag(iχp[r, k]))
                end
                println(io)
            end
        end
        r0 = pos[0]
        for (k, lab) in enumerate(KCOMP)
            push!(rows0, @sprintf("%d,%d,%s,%.15e,%.15e,%.15e,%.15e", i, j, lab,
                                  real(χ[r0, k]), imag(χ[r0, k]), real(iχp[r0, k]), imag(iχp[r0, k])))
        end
        rel(z) = abs(imag(z)) / max(abs(z), 1e-300)
        @printf("  (%d,%d) en %.1f s -> %s\n", i, j, time() - t0, path)
        @printf("      realidad k=0 en ω=0, |Im|/|z|:  χ̌ xx %.1e  yy %.1e  xz %.1e   i∂χ̌ xx %.1e  yy %.1e  xz %.1e\n",
                rel(χ[r0, 1]), rel(χ[r0, 2]), rel(χ[r0, 3]), rel(iχp[r0, 1]), rel(iχp[r0, 2]), rel(iχp[r0, 3]))
    end
    # resumen en ω = 0 de esta corrida (el de la LLG lo escribe --chi0: kernel_chi0_<label>.csv)
    path0 = joinpath(datadir, "kernel_chi0_kernel_" * label * ".csv")
    open(path0, "w") do io
        print(io, meta)
        println(io, "m,n,comp,Re_chi0,Im_chi0,Re_idchi0,Im_idchi0")
        foreach(r -> println(io, r), rows0)
    end
    println("  -> ", path0)
    return nothing
end

"""
Solo χ̌_k(0) e i∂_ωχ̌_k(0) (sin ventanas en ω) para muchos pares: lo que necesita
la LLG en la aproximacion de Markov. Escribe <datadir>/kernel_chi0_<label>.csv con
el mismo formato que kernel_run (lo lee llg_elements.jl).
"""
function chi0_run(pairs, p::KParams, c::KernelConfig, datadir::AbstractString, label::AbstractString)
    mkpath(datadir)
    dω = kdω(p, c); η = keta(p, c)
    meta = @sprintf("# gamma=%.15g gammaSO=%.15g J_sd=%.15g theta=%.15g Omega=%.15g mu=%.15g beta=%.15g eta=%.15g nOmega=%d wide=0 nwide=1 narrow=0 margin=%.15g chi_sign=1\n",
                    p.γ, p.γSO, p.J_sd, p.θ, p.Ω, p.μ, p.β, η, c.nΩ, c.margin)
    @printf("kernel χ̌(0): dω=%.3e (Ω/%d)  η=%.3e  margen=%.2f  |  %d pares\n", dω, c.nΩ, η, c.margin, length(pairs))
    rows = String[]
    rel(z) = abs(imag(z)) / max(abs(z), 1e-300)
    worst = (0.0, 0.0); t0 = time()
    for (n, (i, j)) in enumerate(pairs)
        χ, iχp = kernel_chi(i, j, p, c; svals = [0])
        for (k, lab) in enumerate(KCOMP)
            push!(rows, @sprintf("%d,%d,%s,%.15e,%.15e,%.15e,%.15e", i, j, lab,
                                 real(χ[1, k]), imag(χ[1, k]), real(iχp[1, k]), imag(iχp[1, k])))
        end
        worst = (max(worst[1], maximum(rel(χ[1, k]) for k in 1:3)),
                 max(worst[2], maximum(rel(iχp[1, k]) for k in 1:3)))
        (n % 10 == 0 || n == length(pairs)) &&
            @printf("  %3d/%d pares  (%.0f s)\n", n, length(pairs), time() - t0)
    end
    @printf("  realidad k=0 en ω=0, peor |Im|/|z| entre todos los pares:  χ̌ %.1e   i∂χ̌ %.1e\n", worst...)
    path = joinpath(datadir, "kernel_chi0_" * label * ".csv")
    open(path, "w") do io
        print(io, meta)
        println(io, "m,n,comp,Re_chi0,Im_chi0,Re_idchi0,Im_idchi0")
        foreach(r -> println(io, r), rows)
    end
    println("  -> ", path)
    return nothing
end

# ---------------------------------------------------------------- densidad de espin s̄
#=
Densidad de espin bombeada por el driver en los sitios del lead (sin los F), en t = 0,
que es la que entra en la LLG del marco rotante, s̄_m = ⟨σ⟩_{0,m}(t = 0):

    ⟨σ⟩_{0,m}(t) = Σ_k e^{-ikΩt} Tr[σ ρ_{mm,k}],   ρ_{mm,k} = -i ∫dω/2π 𝒢^<_{mm,k}(ω)
    s̄_m = Σ_{k=0,±1} Tr[σ ρ_{mm,k}]

Con 𝒢^<_{mm,0} = α^< 𝕀 + β^< σy y 𝒢^<_{mm,±1} = γ^<_± τ± (Tr[σx τ±] = ±i, Tr[σz τ±] = 1,
Tr[σy τ±] = 0):

    s̄^x = ∫dω/2π (γ^<_+ - γ^<_-),   s̄^y = -2i ∫dω/2π β^<,   s̄^z = -i ∫dω/2π (γ^<_+ + γ^<_-)
    n   = -2i ∫dω/2π α^<   (ocupacion, solo como control)

Malla ω = s dω (dω = Ω/nΩ, η = 5 dω, como el kernel) sobre el soporte de 𝒢^<. El
error es lineal en η, asi que se calcula con nΩ y 2nΩ y se extrapola a η -> 0:
s̄ = 2 s̄(η/2) - s̄(η) (Richardson). Las partes imaginarias deben ser ~ 0.
=#

"s̄^x, s̄^y, s̄^z y n de los sitios 1..Ns con un nΩ dado. Devuelve (S (Ns,4), η, Nω)."
function sbar_integrals(Ns::Int, p::KParams, nΩ::Int, margin::Real)
    dω = p.Ω / nΩ; η = 5 * dω
    Lmin = -(sqrt(4γt(p)^2 + p.J_sd^2) + margin)                   # banda + estados ligados
    Lmax = p.μ + (isfinite(p.β) ? 40 / p.β : 0.0) + 2p.Ω           # Fermi + replicas
    ωs = collect(floor(Int, Lmin / dω):ceil(Int, Lmax / dω)) .* dω
    pairs = [(m, m) for m in 1:Ns]
    H = harmonics_grid(ωs, pairs, p; η = η)
    w = dω / (2π)
    S = zeros(ComplexF64, Ns, 4)
    for m in 1:Ns
        a  = w * sum(view(H, :, m, IAL)); b  = w * sum(view(H, :, m, IBL))
        gp = w * sum(view(H, :, m, IGPL)); gm = w * sum(view(H, :, m, IGML))
        S[m, 1] = gp - gm
        S[m, 2] = -2im * b
        S[m, 3] = -im * (gp + gm)
        S[m, 4] = -2im * a
        # control de las trazas con las matrices 2x2: s̄ = Tr[σ (ρ₀ + ρ₊ + ρ₋)]
        ρ = -im * (a * σ0 + b * σy + gp * τp + gm * τm)
        sm = [tr(σ * ρ) for σ in (σx, σy, σz)]
        maximum(abs.(sm .- S[m, 1:3])) < 1e-12 * max(1.0, maximum(abs.(sm))) ||
            error("sbar: las trazas no coinciden en el sitio $m")
    end
    return S, η, length(ωs)
end

"""
Escribe <datadir>/sbar_sites1-<Ns>.csv con m, sx, sy, sz, n (extrapolados a η -> 0)
para la LLG de los F (s̄_m del marco rotante).
"""
function sbar_run(Ns::Int, p::KParams, c::KernelConfig, datadir::AbstractString)
    mkpath(datadir)
    t0 = time()
    S1, η1, N1 = sbar_integrals(Ns, p, c.nΩ, c.margin)
    S2, η2, N2 = sbar_integrals(Ns, p, 2c.nΩ, c.margin)
    S = 2 .* S2 .- S1                                              # Richardson η -> 0
    @printf("s̄: η = %.3e y %.3e (%d y %d puntos en ω), extrapolado a η -> 0  (%.1f s)\n",
            η1, η2, N1, N2, time() - t0)
    imx = maximum(abs.(imag.(S[:, 1:3]))) / max(maximum(abs.(real.(S[:, 1:3]))), 1e-300)
    cvg = maximum(abs.(S2[:, 1:3] .- S1[:, 1:3])) / max(maximum(abs.(S[:, 1:3])), 1e-300)
    @printf("  max |Im s̄| / max |s̄| = %.1e   cambio η -> η/2: %.1e (relativo)\n", imx, cvg)
    path = joinpath(datadir, "sbar_sites1-$Ns.csv")
    open(path, "w") do io
        @printf(io, "# gamma=%.15g gammaSO=%.15g J_sd=%.15g theta=%.15g Omega=%.15g mu=%.15g beta=%.15g eta=%.15g eta2=%.15g nOmega=%d margin=%.15g extrapolated=eta0\n",
                p.γ, p.γSO, p.J_sd, p.θ, p.Ω, p.μ, p.β, η1, η2, c.nΩ, c.margin)
        println(io, "m,sx,sy,sz,n")
        for m in 1:Ns
            @printf(io, "%d,%.15e,%.15e,%.15e,%.15e\n", m, real(S[m, 1]), real(S[m, 2]), real(S[m, 3]), real(S[m, 4]))
        end
    end
    for m in 1:min(Ns, 3)
        @printf("  m=%d  s̄ = (%+.6e, %+.6e, %+.6e)   n = %.6f\n", m, real.(S[m, :])...)
    end
    println("  -> ", path)
    return nothing
end

# ---------------------------------------------------------------- kernel en tiempo
#=
Kernel en tiempo, τ = t - t':

    χ(t, t-τ) = Σ_k e^{-ikΩt} χ̌_k(τ),     χ̌_k(ω) = ∫dτ e^{iωτ} χ̌_k(τ)

La convolucion en ω' de las 𝒯(ω,ω') es un producto en τ:

    χ̌_k(τ) = +i J_sd Σ_{a+b=k} 𝒯_{ab}(τ),
    𝒯_{ab}(τ) = 𝒯_{ab}[escalares_ij(τ), escalares_ji(-τ)] · e^{ibΩτ}

con los escalares en τ dados por  x(τ) = ∫dω/2π e^{-iωτ} x(ω).  Esa transformada
se hace con FFT (propia, radix 2) sobre ω_q = ω₀ + qΔω, q = 0..N-1:
    ω₀ = -π/dτ,  Δω = 2π/(N dτ)  =>  τ_n = n dτ exacto en la malla de la FFT,
    periodo en τ: P = 2π/Δω = N dτ  (aliasing ~ e^{-ηP}, despreciable si ηP ≫ 1).
La cola 1/ω de los α^{r,a}_{mm} on-site se resta en ω y se suma exacta en τ:
    1/(ω+iη) -> -i θ(τ) e^{-ητ},   1/(ω-iη) -> +i θ(-τ) e^{ητ}.
Los factores con e^{ibΩτ} reutilizan tsums: Xm = e^{-iΩτ} X (b = -1),
Xp = e^{+iΩτ} X (b = +1).
=#

"FFT directa in situ, radix 2: a_n <- Σ_q a_q e^{-2πi qn/N} (N potencia de 2)."
function fft!(a::AbstractVector{ComplexF64})
    n = length(a)
    ispow2(n) || throw(ArgumentError("fft!: N tiene que ser potencia de 2"))
    j = 0                                                     # permutacion bit-reverse
    for i in 0:n-1
        if i < j
            a[i+1], a[j+1] = a[j+1], a[i+1]
        end
        bit = n >> 1
        while bit > 0 && (j & bit) != 0
            j ⊻= bit
            bit >>= 1
        end
        j |= bit
    end
    tw = [cis(-2π * k / n) for k in 0:(n÷2 - 1)]              # factores exactos (sin acumular)
    len = 2
    while len <= n
        half = len >> 1
        step = n ÷ len
        for s in 1:len:n
            for k in 0:half-1
                w = tw[k*step+1]
                u = a[s+k]; v = a[s+k+half] * w
                a[s+k] = u + v
                a[s+k+half] = u - v
            end
        end
        len <<= 1
    end
    return a
end

"""
Configuracion del kernel en τ:
    dτ     paso en τ (y tamaño de la FFT: ω ∈ [-π/dτ, π/dτ))
    τmax   τ ∈ [-τmax, τmax] (NaN -> T = 2π/Ω)
    log2N  N = 2^log2N puntos de la FFT (Δω = 2π/(N dτ))
    η      ensanchamiento (en τ: amortiguamiento e^{-η|τ|})
"""
Base.@kwdef struct TauConfig
    dτ::Float64    = 0.1
    τmax::Float64  = NaN
    log2N::Int     = 22
    η::Float64     = 5e-5
end
tmaxof(p::KParams, c::TauConfig) = isnan(c.τmax) ? 2π / p.Ω : c.τmax

"""
    coeff_tau(m, n, p, c) -> (A, Nτ)

Los 12 escalares del par (m,n) en τ_n = n dτ, n = -Nτ..Nτ: A[coef, n+Nτ+1].
"""
function coeff_tau(m::Int, n::Int, p::KParams, c::TauConfig)
    N  = 2^c.log2N; dτ = c.dτ; η = c.η
    Δω = 2π / (N * dτ); ω0 = -π / dτ
    Nτ = ceil(Int, tmaxof(p, c) / dτ)
    2Nτ + 1 ≤ N || error("coeff_tau: la FFT es muy corta para τmax (sube log2N)")
    Mmax = max(m, n); diag = m == n
    X = Matrix{ComplexF64}(undef, N, 12)                      # columnas contiguas por escalar
    coeff_omega!(X, 1, ω0, m, n, p, η, Mmax, diag)            # compila en el hilo principal
    @threads for q in 1:N
        coeff_omega!(X, q, ω0 + (q - 1) * Δω, m, n, p, η, Mmax, diag)
    end
    A   = Matrix{ComplexF64}(undef, 12, 2Nτ + 1)
    col = Vector{ComplexF64}(undef, N)
    for ic in 1:12
        copyto!(col, view(X, :, ic))
        fft!(col)
        for nn in -Nτ:Nτ
            τ = nn * dτ
            A[ic, nn+Nτ+1] = col[mod(nn, N)+1] * cis(-ω0 * τ) * Δω / (2π)
        end
    end
    if diag                                                   # colas exactas de α^{r,a}_{mm}
        for nn in -Nτ:Nτ
            τ = nn * dτ; r = nn + Nτ + 1
            A[IAR, r] += τ > 0 ? -im * exp(-η * τ) : (τ == 0 ? -im / 2 : zero(ComplexF64))
            A[IAA, r] += τ < 0 ?  im * exp( η * τ) : (τ == 0 ?  im / 2 : zero(ComplexF64))
        end
    end
    return A, Nτ
end

"Fila q de X: los 12 escalares en ω (con la cola 1/(ω±iη) restada si diag)."
@noinline function coeff_omega!(X, q::Int, ω::Float64, m::Int, n::Int, p::KParams,
                                η::Real, Mmax::Int, diag::Bool)
    cf = lead_coeffs(lead_tables(ω, p, η, Mmax), central(ω, p, η), m, n, p)
    for (ic, v) in enumerate(cf)
        X[q, ic] = v
    end
    if diag
        X[q, IAR] -= 1 / complex(ω, η)
        X[q, IAA] -= 1 / complex(ω, -η)
    end
    return nothing
end

"""
    kernel_tau(i, j, p, c) -> (τs, χ)

χ̌_{ij,k}(τ) de las 6 componentes (KCOMP) en τ ∈ [-τmax, τmax]: χ[n, comp].
"""
function kernel_tau(i::Int, j::Int, p::KParams, c::TauConfig)
    Aij, Nτ = coeff_tau(i, j, p, c)
    Aji = i == j ? Aij : coeff_tau(j, i, p, c)[1]
    nt = 2Nτ + 1
    τs = [(n - Nτ - 1) * c.dτ for n in 1:nt]
    χ  = zeros(ComplexF64, nt, 6)
    pre = im * p.J_sd                                          # +i J_sd
    for r in 1:nt
        τ  = τs[r]
        X  = Aij[:, r]
        Y  = view(Aji, :, nt + 1 - r)                          # escalares ji en -τ
        ph = cis(p.Ω * τ)
        tv = tsums(X, conj(ph) .* X, ph .* X, Y)
        for k in 1:6
            χ[r, k] = pre * tv[k]
        end
    end
    return τs, χ
end

"""
Escribe χ̌_k(τ) de varios pares en <datadir>/kernel_tau_m{i}n{j}.csv e imprime
las comprobaciones: causalidad (χ̌(τ<0) ≈ 0) y realidad de k = 0 (Im ≈ 0).
"""
function tau_run(pairs, p::KParams, c::TauConfig, datadir::AbstractString)
    mkpath(datadir)
    τmax = tmaxof(p, c); N = 2^c.log2N; Δω = 2π / (N * c.dτ)
    meta = @sprintf("# gamma=%.15g gammaSO=%.15g J_sd=%.15g theta=%.15g Omega=%.15g mu=%.15g beta=%.15g eta=%.15g dtau=%.15g taumax=%.15g log2N=%d chi_sign=1\n",
                    p.γ, p.γSO, p.J_sd, p.θ, p.Ω, p.μ, p.β, c.η, c.dτ, τmax, c.log2N)
    @printf("kernel(τ): dτ=%.3g  τmax=%.1f (T=%.1f)  N=2^%d  Δω=%.2e  ω∈[%.1f,%.1f)  η=%.1e  ηP=%.1f\n",
            c.dτ, τmax, 2π / p.Ω, c.log2N, Δω, -π / c.dτ, π / c.dτ, c.η, c.η * N * c.dτ)
    for (i, j) in pairs
        t0 = time()
        τs, χ = kernel_tau(i, j, p, c)
        path = joinpath(datadir, "kernel_tau_m$(i)n$(j).csv")
        open(path, "w") do io
            print(io, meta)
            print(io, "tau")
            for k in KCOMP
                print(io, ",Re_chi_", k, ",Im_chi_", k)
            end
            println(io)
            for r in eachindex(τs)
                @printf(io, "%.10g", τs[r])
                for k in 1:6
                    @printf(io, ",%.12e,%.12e", real(χ[r, k]), imag(χ[r, k]))
                end
                println(io)
            end
        end
        # |τ| < 2: oscilaciones de Gibbs del corte en ω (|ω| < π/dτ); se reportan aparte
        caus  = maximum(abs.(χ[τs .< -2.0, :])) / maximum(abs.(χ))
        caus0 = maximum(abs.(χ[(τs .< -0.5 * c.dτ) .& (τs .>= -2.0), :])) / maximum(abs.(χ))
        reim = [maximum(abs.(imag.(χ[:, k]))) / maximum(abs.(χ[:, k])) for k in 1:3]
        # tiempo de memoria: primer τ ≥ 0 a partir del cual la envolvente de los k = 0
        # queda siempre por debajo del 1 % de su maximo
        pos  = findall(τs .>= 0)
        env  = [maximum(abs.(χ[r, 1:3])) for r in pos]
        tail = reverse(accumulate(max, reverse(env)))
        kmem = findfirst(<(0.01 * maximum(env)), tail)
        τmem = kmem === nothing ? Inf : τs[pos[kmem]]
        @printf("  (%d,%d) en %.1f s -> %s\n", i, j, time() - t0, path)
        @printf("      causalidad max|χ̌|/max|χ̌|: τ<-2 %.1e  (-2≤τ<0: %.1e, Gibbs)   realidad k=0 max|Im|/max|χ̌|: xx %.1e  yy %.1e  xz %.1e\n",
                caus, caus0, reim...)
        @printf("      memoria: τ_mem(1%%) = %.1f   Ω τ_mem = %.3f\n", τmem, p.Ω * τmem)
    end
    return nothing
end

"""
Escribe la malla en CSV: omega, m, n y Re/Im de cada uno de los 12 escalares.
"""
function write_csv(path::AbstractString, ωs, pairs, H, p::KParams, η::Real)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        @printf(io, "# gamma=%.15g gammaSO=%.15g J_sd=%.15g theta=%.15g Omega=%.15g mu=%.15g beta=%.15g eta=%.15g\n",
                p.γ, p.γSO, p.J_sd, p.θ, p.Ω, p.μ, p.β, η)
        print(io, "omega,m,n")
        for c in COEFS
            print(io, ",Re_", c, ",Im_", c)
        end
        println(io)
        for (ip, (m, n)) in enumerate(pairs), iω in eachindex(ωs)
            @printf(io, "%.10g,%d,%d", ωs[iω], m, n)
            for ic in eachindex(COEFS)
                v = H[iω, ip, ic]
                @printf(io, ",%.12e,%.12e", real(v), imag(v))
            end
            println(io)
        end
    end
    return path
end

# ---------------------------------------------------------------- validacion
"""
Compara con fully_analitical_inbedding_leads.jl (implementacion matricial ya
validada contra el embedding de Sambe) para r y <, y comprueba la avanzada con
la identidad 𝒢^a_{mn,k}(ω) = [𝒢^r_{nm,-k}(ω+kΩ)]†.
"""
function validate(p::KParams = KParams(); η::Real = 2e-3, M::Int = 4)
    ref = joinpath(@__DIR__, "..", "analytical_computations", "fully_analitical_inbedding_leads.jl")
    isfile(ref) || error("no encuentro $ref")
    FA = Module(:FAref)
    Base.include(FA, ref)
    # Las funciones de FA nacen despues de compilar validate: el cuerpo se llama con
    # invokelatest para que todo acceso a FA ocurra en el "world" mas reciente
    # (Julia ≥ 1.12 avisa si no).
    return Base.invokelatest(_validate_body, FA, p, η, M)
end

function _validate_body(FA::Module, p::KParams, η::Real, M::Int)
    q  = FA.AParams(; λ = p.γSO, t = p.γ, J_sd = p.J_sd, θ = p.θ, Ω = p.Ω, μ = p.μ, β = p.β)
    pairs = [(m, n) for m in 1:M for n in 1:M]
    er = el = ea = sr = sl = 0.0
    for ω in range(-2.8, 2.8; length = 561)
        h = FA.central(ω, q; η = η)
        for (m, n) in pairs
            c   = lead_coeffs(ω, m, n, p; η = η)
            hr  = FA.lead_harm(ω, m, n, q; η = η, C = h)
            for k in -1:1
                er = max(er, maximum(abs.(Gmat(c, :r, k) .- hr.Gr[k]))); sr = max(sr, maximum(abs.(hr.Gr[k])))
                el = max(el, maximum(abs.(Gmat(c, :l, k) .- hr.Gl[k]))); sl = max(sl, maximum(abs.(hr.Gl[k])))
                cs = lead_coeffs(ω + k * p.Ω, n, m, p; η = η)
                ea = max(ea, maximum(abs.(Gmat(c, :a, k) .- adjoint(Gmat(cs, :r, -k)))))
            end
        end
    end
    # derivada exacta (Jet) frente a diferencias centradas de 4o orden, h = η/50
    ed = sd = 0.0; h = η / 50
    for ω in range(-2.3, 1.7; length = 41), (m, n) in ((1, 1), (1, 2), (2, 1), (3, 2))
        cj = lead_coeffs(lead_tables(Jet(ω, 1.0), p, η, max(m, n)), central(Jet(ω, 1.0), p, η), m, n, p)
        f(x) = collect(lead_coeffs(x, m, n, p; η = η))
        fd = (8 .* (f(ω + h) .- f(ω - h)) .- (f(ω + 2h) .- f(ω - 2h))) ./ (12h)
        dj = [v.d for v in cj]
        ed = max(ed, maximum(abs.(dj .- fd))); sd = max(sd, maximum(abs.(fd)))
    end
    # FFT propia frente a la DFT directa
    xs = [complex(sin(0.3k) + 0.1k, cos(1.7k)) for k in 0:255]
    dft = [sum(xs[q+1] * cis(-2π * q * n / 256) for q in 0:255) for n in 0:255]
    ef = maximum(abs.(fft!(copy(xs)) .- dft)) / maximum(abs.(dft))
    ok(x) = x < 1e-10 ? "OK" : "FALLA"
    println("="^72)
    @printf("  ∂_ω exacta (Jet) vs diferencias finitas: %.1e [%s]\n", ed / sd, ed / sd < 1e-6 ? "OK" : "FALLA")
    @printf("  FFT propia vs DFT directa (N=256): %.1e [%s]\n", ef, ok(ef))
    @printf("kernel_elements.jl  vs  fully_analitical_inbedding_leads.jl   (m,n ≤ %d, η = %.1e)\n", M, η)
    @printf("  𝒢ʳ_{mn,k}: %.1e [%s]\n", er / sr, ok(er / sr))
    @printf("  𝒢<_{mn,k}: %.1e [%s]\n", el / sl, ok(el / sl))
    @printf("  𝒢ᵃ_{mn,k} = [𝒢ʳ_{nm,-k}(ω+kΩ)]†: %.1e [%s]\n", ea / sr, ok(ea / sr))
    println("="^72)
    return nothing
end

# ---------------------------------------------------------------- CLI
const DEFAULT_PARAMS = joinpath(@__DIR__, "..", "two_oscillators_setup", "y_precession", "output",
                                "DFF", "F1-2", "gso0p1_jsd0p2_th10deg_Om0p005_buf4_adyn0p0", "params.txt")

function parse_pairs(s::AbstractString)
    out = Tuple{Int,Int}[]
    for tok in split(s, ';'; keepempty = false)
        a = split(strip(tok), ',')
        length(a) == 2 || error("par mal formado: '$tok' (formato m,n;m,n)")
        push!(out, (parse(Int, a[1]), parse(Int, a[2])))
    end
    return out
end

"""
Copia de p con los campos de `kw` reemplazados (para barridos desde Julia):
    with(p; J_sd = 0.3, θ = deg2rad(20))
"""
with(p::KParams; kw...) = KParams(; (f => get(kw, f, getfield(p, f)) for f in fieldnames(KParams))...)

"""
Etiqueta a partir de la carpeta de la corrida de TDNEGF, sin los campos que solo
tienen sentido alli (buf.., adyn..). En la estructura nueva se antepone setup2_ (DFF)
o setup1_ (FDF), para que el tag sea el mismo que con el nombre viejo:
    .../y_precession/output/DFF/F1-2/gso0p1_..._Om0p005_buf4_adyn0p0 -> setup2_gso0p1_..._Om0p005
    .../y_precession/output/setup2_gso0p1_..._Om0p005_buf4_adyn0p0  -> setup2_gso0p1_..._Om0p005
"""
function run_tag(params::AbstractString)
    d = dirname(abspath(params))
    tag = join(filter(t -> !startswith(t, "buf") && !startswith(t, "adyn"),
                      split(basename(d), '_')), '_')
    lay = basename(dirname(dirname(d)))
    pre = Dict("DFF" => "setup2_", "FDF" => "setup1_")
    return startswith(tag, "setup") ? tag : get(pre, lay, "") * tag
end

"Numero para etiquetas: %g con '.' -> 'p' y '-' -> 'm' (0.005 -> 0p005)."
tagnum(x::Real) = replace(@sprintf("%g", x), "." => "p", "-" => "m")

"Etiqueta construida con los valores de los parametros (se usa cuando hay overrides)."
param_tag(p::KParams) = "gam$(tagnum(p.γ))_gso$(tagnum(p.γSO))_jsd$(tagnum(p.J_sd))_th$(tagnum(rad2deg(p.θ)))deg_" *
                        "Om$(tagnum(p.Ω))_mu$(tagnum(p.μ))_b$(tagnum(p.β))"

"""
Parametros finales: los del params.txt (o los de KParams() si params = "none")
con los overrides de la linea de comandos que no esten vacios.
θ se da en grados.
"""
function build_params(params::AbstractString, ov::Dict{String,String})
    base = params == "none" ? KParams() : read_params(params)
    g(k, v) = isempty(ov[k]) ? v : parse(Float64, ov[k])
    return KParams(; γ = g("--gamma", base.γ), γSO = g("--gso", base.γSO), J_sd = g("--jsd", base.J_sd),
                     θ = deg2rad(g("--theta", rad2deg(base.θ))), Ω = g("--Omega", base.Ω),
                     μ = g("--mu", base.μ), β = g("--beta", base.β))
end

const OVERRIDES = ("--gamma", "--gso", "--jsd", "--theta", "--Omega", "--mu", "--beta")

function main(args = ARGS)
    opt = Dict("--params" => DEFAULT_PARAMS, "--pairs" => "1,1;1,2;2,1;2,2", "--wmin" => "-3.5",
               "--wmax" => "3.5", "--nw" => "28001", "--eta" => "", "--out" => "", "--datadir" => "",
               "--k-nomega" => "40", "--k-eta" => "", "--k-wide" => "6.0", "--k-nwide" => "1601",
               "--k-narrow" => "2.5", "--k-margin" => "2.0",
               "--t-dtau" => "0.1", "--t-taumax" => "", "--t-log2n" => "22", "--t-eta" => "5e-5",
               "--sites" => "", "--tag" => "")
    for k in OVERRIDES
        opt[k] = ""
    end
    validate_only = false
    do_kernel = false
    do_tau = false
    do_chi0 = false
    do_sbar = false
    i = 1
    while i ≤ length(args)
        a = args[i]
        if a == "--validate"
            validate_only = true; i += 1
        elseif a == "--kernel"
            do_kernel = true; i += 1
        elseif a == "--tau"
            do_tau = true; i += 1
        elseif a == "--chi0"
            do_chi0 = true; i += 1
        elseif a == "--sbar"
            do_sbar = true; i += 1
        elseif haskey(opt, a) && i < length(args)
            opt[a] = args[i+1]; i += 2
        else
            error("argumento no reconocido: $a")
        end
    end
    p = build_params(opt["--params"], opt)
    overridden = opt["--params"] == "none" || any(!isempty(opt[k]) for k in OVERRIDES)
    if validate_only
        Base.invokelatest(validate, p)          # llamadas dinamicas: la inferencia de
        return nothing                          # main no recorre todo el arbol de llamadas
    end
    # --sites N: todos los pares (i,j) ∈ [1,N]^2 (etiqueta sites1-N); si no, --pairs
    if isempty(opt["--sites"])
        pairs = parse_pairs(opt["--pairs"])
        plabel = join(["$m-$n" for (m, n) in pairs], "_")
    else
        Ns = parse(Int, opt["--sites"])
        pairs = [(m, n) for m in 1:Ns for n in 1:Ns]
        plabel = "sites1-$Ns"
    end
    # salida por defecto: output/<tag>/data/...
    # tag = carpeta del params.txt, o la etiqueta de los valores si hay overrides
    tag = !isempty(opt["--tag"]) ? opt["--tag"] : overridden ? param_tag(p) : run_tag(opt["--params"])
    datadir = isempty(opt["--datadir"]) ? joinpath(@__DIR__, "output", tag, "data") : opt["--datadir"]
    if do_tau
        c = TauConfig(; dτ = parse(Float64, opt["--t-dtau"]),
                        τmax = isempty(opt["--t-taumax"]) ? NaN : parse(Float64, opt["--t-taumax"]),
                        log2N = parse(Int, opt["--t-log2n"]), η = parse(Float64, opt["--t-eta"]))
        @printf("γ̃=%.4f θ_SO=%.4f J_sd=%.3f θ=%.1f° Ω=%.4g μ=%.2f β=%.1f | pares %s | hilos %d\n",
                γt(p), θSO(p), p.J_sd, rad2deg(p.θ), p.Ω, p.μ, p.β,
                join(["($m,$n)" for (m, n) in pairs], " "), nthreads())
        Base.invokelatest(tau_run, pairs, p, c, datadir)
        return nothing
    end
    if do_sbar
        c = KernelConfig(; nΩ = parse(Int, opt["--k-nomega"]), margin = parse(Float64, opt["--k-margin"]))
        Ns = isempty(opt["--sites"]) ? 10 : parse(Int, opt["--sites"])
        @printf("γ̃=%.4f θ_SO=%.4f J_sd=%.3f θ=%.1f° Ω=%.4g μ=%.2f β=%.1f | sitios 1..%d | hilos %d\n",
                γt(p), θSO(p), p.J_sd, rad2deg(p.θ), p.Ω, p.μ, p.β, Ns, nthreads())
        Base.invokelatest(sbar_run, Ns, p, c, datadir)
        return nothing
    end
    if do_chi0
        c = KernelConfig(; nΩ = parse(Int, opt["--k-nomega"]),
                           η = isempty(opt["--k-eta"]) ? NaN : parse(Float64, opt["--k-eta"]),
                           margin = parse(Float64, opt["--k-margin"]))
        @printf("γ̃=%.4f θ_SO=%.4f J_sd=%.3f θ=%.1f° Ω=%.4g μ=%.2f β=%.1f | hilos %d\n",
                γt(p), θSO(p), p.J_sd, rad2deg(p.θ), p.Ω, p.μ, p.β, nthreads())
        Base.invokelatest(chi0_run, pairs, p, c, datadir, plabel)
        return nothing
    end
    if do_kernel
        c = KernelConfig(; nΩ = parse(Int, opt["--k-nomega"]),
                           η = isempty(opt["--k-eta"]) ? NaN : parse(Float64, opt["--k-eta"]),
                           wide = parse(Float64, opt["--k-wide"]), nwide = parse(Int, opt["--k-nwide"]),
                           narrow = parse(Float64, opt["--k-narrow"]), margin = parse(Float64, opt["--k-margin"]))
        @printf("γ̃=%.4f θ_SO=%.4f J_sd=%.3f θ=%.1f° Ω=%.4g μ=%.2f β=%.1f | pares %s | hilos %d\n",
                γt(p), θSO(p), p.J_sd, rad2deg(p.θ), p.Ω, p.μ, p.β,
                join(["($m,$n)" for (m, n) in pairs], " "), nthreads())
        Base.invokelatest(kernel_run, pairs, p, c, datadir, plabel)
        return nothing
    end
    ωs = range(parse(Float64, opt["--wmin"]), parse(Float64, opt["--wmax"]);
               length = parse(Int, opt["--nw"]))
    η  = isempty(opt["--eta"]) ? 2 * step(ωs) : parse(Float64, opt["--eta"])
    out = isempty(opt["--out"]) ?
          joinpath(datadir, "kernel_elements_" * join(["$m-$n" for (m, n) in pairs], "_") * ".csv") :
          opt["--out"]
    @printf("γ̃=%.4f θ_SO=%.4f J_sd=%.3f θ=%.1f° Ω=%.4g μ=%.2f β=%.1f | Nω=%d dω=%.2e η=%.2e | pares %s | hilos %d\n",
            γt(p), θSO(p), p.J_sd, rad2deg(p.θ), p.Ω, p.μ, p.β, length(ωs), step(ωs), η,
            join(["($m,$n)" for (m, n) in pairs], " "), nthreads())
    t0 = time()
    H  = Base.invokelatest(harmonics_grid, ωs, pairs, p; η = η)
    @printf("  armonicos calculados en %.2f s\n", time() - t0)
    println("  -> ", write_csv(out, ωs, pairs, H, p, η))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
