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

# ---------------------------------------------------------------- S, g₁₁
"Sʳ(ω) = sqrt((ω+iη)² - 4γ̃²) en la rama con Im Sʳ ≥ 0."
@inline function Sr(ω::Real, p::KParams, η::Real)
    z = complex(ω, η); b = 2γt(p)
    return sqrt(z - b) * sqrt(z + b)
end

"g₁₁ʳ(ω) = [(ω+iη) - Sʳ(ω)]/(2γ̃²)."
@inline g11r(ω::Real, p::KParams, η::Real) = (complex(ω, η) - Sr(ω, p, η)) / (2γt(p)^2)

# ---------------------------------------------------------------- centro
"""
Coeficientes de los armonicos del sitio central en ω (solo dependen de ω):
    𝒢^μ_{C,0} = A0 σ0 + B0 σy,   𝒢^μ_{C,+1} = Ap τ+,   𝒢^μ_{C,-1} = Am τ-
para μ = r, a (= conjugado de r) y < (formulas de X<↑, X<↓, A<_{C,±1}).
"""
struct Central
    A0r::ComplexF64; B0r::ComplexF64; Apr::ComplexF64; Amr::ComplexF64
    A0a::ComplexF64; B0a::ComplexF64; Apa::ComplexF64; Ama::ComplexF64
    A0l::ComplexF64; B0l::ComplexF64; Apl::ComplexF64; Aml::ComplexF64
end

function central(ω::Real, p::KParams, η::Real)
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

    return Central(A0r, B0r, Apr, Amr,
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
struct LeadTables
    g0::Vector{ComplexF64}; gp::Vector{ComplexF64}; gm::Vector{ComplexF64}
    geo::Vector{ComplexF64}; mγ::Vector{Float64}
    f0::Float64; fp::Float64; fm::Float64
end

function lead_tables(ω::Real, p::KParams, η::Real, Mmax::Int)
    pw(g) = (v = Vector{ComplexF64}(undef, Mmax + 1); v[1] = 1; for s in 1:Mmax; v[s+1] = v[s] * g; end; v)
    g0 = g11r(ω, p, η)
    q  = (γt(p) * g0)^2
    geo = Vector{ComplexF64}(undef, Mmax)
    acc = zero(ComplexF64); t = one(ComplexF64)
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
    ok(x) = x < 1e-10 ? "OK" : "FALLA"
    println("="^72)
    @printf("kernel_elements.jl  vs  fully_analitical_inbedding_leads.jl   (m,n ≤ %d, η = %.1e)\n", M, η)
    @printf("  𝒢ʳ_{mn,k}: %.1e [%s]\n", er / sr, ok(er / sr))
    @printf("  𝒢<_{mn,k}: %.1e [%s]\n", el / sl, ok(el / sl))
    @printf("  𝒢ᵃ_{mn,k} = [𝒢ʳ_{nm,-k}(ω+kΩ)]†: %.1e [%s]\n", ea / sr, ok(ea / sr))
    println("="^72)
    return nothing
end

# ---------------------------------------------------------------- CLI
const DEFAULT_PARAMS = joinpath(@__DIR__, "..", "two_oscillators_setup", "y_precession", "output",
                                "setup2_gso0p1_jsd0p2_th10deg_Om0p005_buf4_adyn0p0", "params.txt")

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
    opt = Dict("--params" => DEFAULT_PARAMS, "--pairs" => "1,1", "--wmin" => "-2.5",
               "--wmax" => "2.5", "--nw" => "20001", "--eta" => "", "--out" => "")
    for k in OVERRIDES
        opt[k] = ""
    end
    validate_only = false
    i = 1
    while i ≤ length(args)
        a = args[i]
        if a == "--validate"
            validate_only = true; i += 1
        elseif haskey(opt, a) && i < length(args)
            opt[a] = args[i+1]; i += 2
        else
            error("argumento no reconocido: $a")
        end
    end
    p = build_params(opt["--params"], opt)
    overridden = opt["--params"] == "none" || any(!isempty(opt[k]) for k in OVERRIDES)
    if validate_only
        validate(p)
        return nothing
    end
    ωs = range(parse(Float64, opt["--wmin"]), parse(Float64, opt["--wmax"]);
               length = parse(Int, opt["--nw"]))
    η  = isempty(opt["--eta"]) ? 2 * step(ωs) : parse(Float64, opt["--eta"])
    pairs = parse_pairs(opt["--pairs"])
    # salida por defecto: output/<tag>/data/kernel_elements_<pares>.csv
    # tag = carpeta del params.txt, o la etiqueta de los valores si hay overrides
    tag = overridden ? param_tag(p) : basename(dirname(abspath(opt["--params"])))
    out = isempty(opt["--out"]) ?
          joinpath(@__DIR__, "output", tag, "data",
                   "kernel_elements_" * join(["$m-$n" for (m, n) in pairs], "_") * ".csv") :
          opt["--out"]
    @printf("γ̃=%.4f θ_SO=%.4f J_sd=%.3f θ=%.1f° Ω=%.4g μ=%.2f β=%.1f | Nω=%d dω=%.2e η=%.2e | pares %s | hilos %d\n",
            γt(p), θSO(p), p.J_sd, rad2deg(p.θ), p.Ω, p.μ, p.β, length(ωs), step(ωs), η,
            join(["($m,$n)" for (m, n) in pairs], " "), nthreads())
    t0 = time()
    H  = harmonics_grid(ωs, pairs, p; η = η)
    @printf("  armonicos calculados en %.2f s\n", time() - t0)
    println("  -> ", write_csv(out, ωs, pairs, H, p, η))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
