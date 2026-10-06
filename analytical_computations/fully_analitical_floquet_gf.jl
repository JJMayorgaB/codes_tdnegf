#!/usr/bin/env julia
#=
fully_analitical_floquet_gf.jl -- funcion de Green de Floquet del sitio central
calculada SOLO con las formulas cerradas (sectores de la carga conservada
Q = n - σz/2). No invierte matrices de Sambe: lo unico numerico son las
integrales en ω que llevan los armonicos al tiempo (ρₖ = 𝒢<ₖ(τ=0)).

Convenciones (las del texto):
    H_sd = -J_sd σ·M,   σ·M(t) = Σₖ e^{-ikΩt} Mₖ,  M₀ = cosθ σz, M₊₁ = sinθ σ₊, M₋₁ = sinθ σ₋
    G(t,t') = Σₖ e^{-ikΩt} 𝒢ₖ(t-t'),   𝒢ₖ(ω) = Ǧₖ₀(ω)
    Sʳ(ω) = sqrt((ω+i0⁺)² - 4γ²),  Sᵃ = (Sʳ)*,  dₙ = S(ω+nΩ)
    Δ₀  = (d₀ - Jc)(d₁ + Jc) - Js²,   Δ₋₁ = (d₋₁ - Jc)(d₀ + Jc) - Js²
    𝒢₀ = diag((d₋₁-Jc)/Δ₋₁, (d₁+Jc)/Δ₀),  𝒢₊₁ = -Js/Δ₀ σ₊,  𝒢₋₁ = -Js/Δ₋₁ σ₋
    Σ<(ω) = 2i f(ω) Im Sʳ(ω)
    𝒢<ₖ(ω) = Σₚ 𝒢ʳₖ₋ₚ(ω+pΩ) Σ<(ω+pΩ) 𝒢ᵃₚ(ω),  p = 0,±1

Salidas (output/):
    fa_floquet_ldos.csv   LDOS resuelta en espin, -i𝒢<₀ y armonicos k=±1 de 𝒢ʳ
    fa_floquet_rho_t.csv  ρ(t), n(t) y ⟨σx,σy,σz⟩(t) sobre Nper periodos

Los tests (validate) comparan contra la diagonalizacion exacta de
floquet_gf.jl, que se carga aislada en el modulo FGF.
=#

using LinearAlgebra
using Printf
using DelimitedFiles

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# referencia numerica (solo para los tests), aislada para no chocar nombres
module FGF
    include(joinpath(@__DIR__, "floquet_gf.jl"))
end

const σ0 = ComplexF64[1 0; 0 1]
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

#  parametros
Base.@kwdef struct AParams
    λ::Float64    = 0.1              # Rashba γ_SO
    t::Float64    = sqrt(1 - λ^2)    # hopping γ
    J_sd::Float64 = 0.2
    θ::Float64    = deg2rad(10.0)    # angulo del cono
    Ω::Float64    = 0.005
    μ::Float64    = 0.0              # E_F
    β::Float64    = 40.0
end

γ_band(p::AParams) = sqrt(p.t^2 + p.λ^2)

"Copia de `p` cambiando los campos dados."
function with(p::AParams; kw...)
    nt = NamedTuple{fieldnames(AParams)}(Tuple(getfield(p, f) for f in fieldnames(AParams)))
    return AParams(; merge(nt, (; kw...))...)
end

to_fgf(p::AParams; N::Int) = FGF.FloquetParams(; λ = p.λ, t = p.t, J_sd = p.J_sd, θ = p.θ,
                                                 Ω = p.Ω, μ = p.μ, β = p.β, N = N)

fermi(ω, p::AParams) = 1 / (1 + exp(clamp(p.β * (ω - p.μ), -500.0, 500.0)))

#S, Σ<
"Sʳ(ω) = sqrt((ω+iη)² - 4γ²), rama con Im Sʳ ≥ 0 (forma factorizada)."
function Sr(ω::Real, p::AParams; η::Real)
    z  = complex(ω, η)
    g2 = 2γ_band(p)
    return sqrt(z - g2) * sqrt(z + g2)
end

"Sᵃ(ω) = (Sʳ(ω))*  (equivale a +i0⁺ -> -i0⁺)."
Sa(ω::Real, p::AParams; η::Real) = conj(Sr(ω, p; η = η))

"Σ<(ω) = 2i f(ω) Im Sʳ(ω)  (dos leads, μ_L = μ_R = μ)."
Σless(ω::Real, p::AParams; η::Real) = 2im * fermi(ω, p) * imag(Sr(ω, p; η = η))

# armonicos
"""
    harm(ω, p, S; η) -> (uu0, dd0, ud1, du_1)

Unicas componentes no nulas de los armonicos k = 0, ±1, con S = Sr (retardada)
o S = Sa (avanzada):
    uu0  = 𝒢₀^{↑↑},  dd0 = 𝒢₀^{↓↓},  ud1 = 𝒢₊₁^{↑↓},  du_1 = 𝒢₋₁^{↓↑}.
"""
function harm(ω::Real, p::AParams, S::Function; η::Real)
    Jc  = p.J_sd * cos(p.θ)
    Js  = p.J_sd * sin(p.θ)
    dm1 = S(ω - p.Ω, p; η = η)
    d0  = S(ω,       p; η = η)
    d1  = S(ω + p.Ω, p; η = η)
    Δ0  = (d0  - Jc) * (d1 + Jc) - Js^2
    Δm1 = (dm1 - Jc) * (d0 + Jc) - Js^2
    return (uu0 = (dm1 - Jc) / Δm1, dd0 = (d1 + Jc) / Δ0,
            ud1 = -Js / Δ0,          du_1 = -Js / Δm1)
end

"Armonico k como matriz 2x2 (base ↑,↓) a partir de las componentes."
function as_matrix(h, k::Int)
    k ==  0 && return ComplexF64[h.uu0 0; 0 h.dd0]
    k ==  1 && return ComplexF64[0 h.ud1; 0 0]
    k == -1 && return ComplexF64[0 0; h.du_1 0]
    return zeros(ComplexF64, 2, 2)
end

Gr_k(ω, k, p; η) = as_matrix(harm(ω, p, Sr; η = η), k)
Ga_k(ω, k, p; η) = as_matrix(harm(ω, p, Sa; η = η), k)

"""
    lesser(ω, p; η) -> (uu0, dd0, ud1, du_1)

Componentes no nulas de 𝒢<ₖ(ω), k = 0, ±1 (Σ< escalar sale de cada producto):
    𝒢<₀^{↑↑}  = Σ<(ω)𝒢ʳ₀^{↑↑}𝒢ᵃ₀^{↑↑}           + Σ<(ω-Ω)𝒢ʳ₊₁^{↑↓}(ω-Ω)𝒢ᵃ₋₁^{↓↑}
    𝒢<₀^{↓↓}  = Σ<(ω)𝒢ʳ₀^{↓↓}𝒢ᵃ₀^{↓↓}           + Σ<(ω+Ω)𝒢ʳ₋₁^{↓↑}(ω+Ω)𝒢ᵃ₊₁^{↑↓}
    𝒢<₊₁^{↑↓} = Σ<(ω)𝒢ʳ₊₁^{↑↓}𝒢ᵃ₀^{↓↓}          + Σ<(ω+Ω)𝒢ʳ₀^{↑↑}(ω+Ω)𝒢ᵃ₊₁^{↑↓}
    𝒢<₋₁^{↓↑} = Σ<(ω)𝒢ʳ₋₁^{↓↑}𝒢ᵃ₀^{↑↑}          + Σ<(ω-Ω)𝒢ʳ₀^{↓↓}(ω-Ω)𝒢ᵃ₋₁^{↓↑}
"""
function lesser(ω::Real, p::AParams; η::Real)
    Ω  = p.Ω
    r0 = harm(ω,     p, Sr; η = η)
    rp = harm(ω + Ω, p, Sr; η = η)
    rm = harm(ω - Ω, p, Sr; η = η)
    a0 = harm(ω,     p, Sa; η = η)
    s0 = Σless(ω,     p; η = η)
    sp = Σless(ω + Ω, p; η = η)
    sm = Σless(ω - Ω, p; η = η)
    return (uu0  = s0 * r0.uu0  * a0.uu0 + sm * rm.ud1  * a0.du_1,
            dd0  = s0 * r0.dd0  * a0.dd0 + sp * rp.du_1 * a0.ud1,
            ud1  = s0 * r0.ud1  * a0.dd0 + sp * rp.uu0  * a0.ud1,
            du_1 = s0 * r0.du_1 * a0.uu0 + sm * rm.dd0  * a0.du_1)
end

Gl_k(ω, k, p; η) = as_matrix(lesser(ω, p; η = η), k)

#observables
"LDOS resuelta en espin: A(ω) = -1/π Im 𝒢ʳ₀(ω) -> (↑, ↓)."
function ldos(ω::Real, p::AParams; η::Real)
    h = harm(ω, p, Sr; η = η)
    return (-imag(h.uu0) / π, -imag(h.dd0) / π)
end

"""
    rho_harmonics(p; η, ωmin, ωmax, Nω) -> Dict{Int,Matrix{ComplexF64}}

ρₖ = -i ∫ dω/2π 𝒢<ₖ(ω) = -i 𝒢<ₖ(τ=0), k = 0, ±1 (los unicos no nulos).
"""
function rho_harmonics(p::AParams; η::Real, ωmin::Real, ωmax::Real, Nω::Int)
    ωs  = range(ωmin, ωmax; length = Nω)
    dω  = step(ωs)
    acc = Dict(k => zeros(ComplexF64, 2, 2) for k in -1:1)
    for ω in ωs
        L = lesser(ω, p; η = η)
        for k in -1:1
            acc[k] .+= as_matrix(L, k)
        end
    end
    return Dict(k => -im .* acc[k] .* dω ./ (2π) for k in -1:1)
end

"ρ(t) = Σₖ e^{-ikΩt} ρₖ."
function rho_of_t(t::Real, ρks::Dict{Int, Matrix{ComplexF64}}, Ω::Real)
    ρt = zeros(ComplexF64, 2, 2)
    for (k, ρk) in ρks
        ρt .+= exp(-im * k * Ω * t) .* ρk
    end
    return ρt
end

spin(ρ) = (real(tr(σx * ρ)), real(tr(σy * ρ)), real(tr(σz * ρ)))
occ(ρ)  = real(tr(ρ))

function spin_time_series(p::AParams; η::Real, ωmin::Real, ωmax::Real, Nω::Int,
                          Nper::Int = 3, Nt::Int = 601)
    ρks = rho_harmonics(p; η = η, ωmin = ωmin, ωmax = ωmax, Nω = Nω)
    T   = 2π / p.Ω
    ts  = collect(range(0, Nper * T; length = Nt))
    ρts = [rho_of_t(t, ρks, p.Ω) for t in ts]
    return (t = ts, ρ = ρts, ρks = ρks,
            sx = [spin(r)[1] for r in ρts], sy = [spin(r)[2] for r in ρts],
            sz = [spin(r)[3] for r in ρts], n  = [occ(r) for r in ρts])
end

#tests
"""
Referencia por diagonalizacion exacta (floquet_gf.jl) con la MISMA convencion
de la derivacion: diagonal ω + nΩ + iη - Σʳ(ω+nΩ) = Sʳ(ω+nΩ) (por eso se suma
iη a floquet_matrix, que no lo trae) y Σ̌<ₙₙ = Σ<(ω+nΩ) analitica.
"""
function num_ref(ω::Real, p::AParams; η::Real, N::Int)
    q  = to_fgf(p; N = N)
    Gr = inv(FGF.floquet_matrix(ω, q; η = η) + im * η * I)
    nf = FGF.n_floquet(q)
    S  = zeros(ComplexF64, 2nf, 2nf)
    for n in -N:N
        r = FGF.fblock(FGF.fidx(n, q))
        S[r, r] = Σless(ω + n * p.Ω, p; η = η) * σ0
    end
    return Gr, adjoint(Gr), Gr * S * adjoint(Gr), q
end

relerr(a, b) = maximum(abs.(a - b)) / max(maximum(abs.(b)), 1e-300)

function validate(p::AParams = AParams(); η::Real = 2e-3)
    ok(b) = b ? "OK" : "FALLA"
    println("="^76)
    @printf("t=%.3f  λ=%.3f  J_sd=%.3f  θ=%.1f°  Ω=%.4f  β=%.1f  η=%.1e\n",
            p.t, p.λ, p.J_sd, rad2deg(p.θ), p.Ω, p.β, η)
    println("="^76)

    # 1. rama de Sʳ
    ωl   = range(-6, 6; length = 4001)
    imin = minimum(imag(Sr(ω, p; η = η)) for ω in ωl)
    g2   = 4γ_band(p)^2
    easy = max(abs(real(Sr(50.0, p; η = η)) - sqrt(2500 - g2)),
               abs(real(Sr(-50.0, p; η = η)) + sqrt(2500 - g2)))
    q1   = to_fgf(p; N = 1)
    efgf = maximum(abs(complex(ω, η) - Sr(ω, p; η = η) - FGF.Σr(ω, q1; η = η)) for ω in ωl)
    @printf("1. Sʳ:  min Im Sʳ = %.2e [%s];  Sʳ(±50) ≈ ±sqrt(ω²-4γ²): %.1e [%s];  z-Sʳ vs Σʳ(floquet_gf): %.1e [%s]\n",
            imin, ok(imin > -1e-14), easy, ok(easy < 1e-3), efgf, ok(efgf < 1e-12))

    # 2-4. armonicos vs Sambe (N=1 y N=4), soporte |k|=2
    ωs = range(-3, 3; length = 601)
    for N in (1, 4)
        er = ea = el = 0.0; e2 = 0.0
        for ω in ωs
            Gr, Ga, Gl, q = num_ref(ω, p; η = η, N = N)
            L = lesser(ω, p; η = η)
            hr, ha = harm(ω, p, Sr; η = η), harm(ω, p, Sa; η = η)
            for k in -1:1
                er = max(er, relerr(as_matrix(hr, k), FGF.fget(Gr, k, 0, q)))
                ea = max(ea, relerr(as_matrix(ha, k), FGF.fget(Ga, k, 0, q)))
                el = max(el, relerr(as_matrix(L,  k), FGF.fget(Gl, k, 0, q)))
            end
            if N ≥ 2
                for k in (-2, 2)
                    e2 = max(e2, maximum(abs.(FGF.fget(Gr, k, 0, q))), maximum(abs.(FGF.fget(Gl, k, 0, q))))
                end
            end
        end
        @printf("%s N=%d:  𝒢ʳₖ %.1e [%s]   𝒢ᵃₖ %.1e [%s]   𝒢<ₖ %.1e [%s]",
                N == 1 ? "2-4." : "    ", N, er, ok(er < 1e-10), ea, ok(ea < 1e-10), el, ok(el < 1e-10))
        N ≥ 2 ? @printf("   max|Ǧₖ₀|, |k|=2: %.1e [%s]\n", e2, ok(e2 < 1e-12)) : println()
    end

    # 5. antihermiticidad: 𝒢<ₖ(ω) = -[𝒢<₋ₖ(ω+kΩ)]†
    e5 = maximum(maximum(abs.(Gl_k(ω, k, p; η = η) + adjoint(Gl_k(ω + k * p.Ω, -k, p; η = η))))
                 for ω in ωs, k in -1:1)
    sc = maximum(maximum(abs.(Gl_k(ω, 0, p; η = η))) for ω in ωs)
    @printf("5. 𝒢<ₖ(ω) = -[𝒢<₋ₖ(ω+kΩ)]†:  %.1e [%s]\n", e5 / sc, ok(e5 / sc < 1e-12))

    # 6. limites
    ω0 = 0.37
    p0 = with(p; J_sd = 0.0)
    e6a = max(maximum(abs.(Gr_k(ω0, 0, p0; η = η) - σ0 / Sr(ω0, p0; η = η))),
              maximum(abs.(Gr_k(ω0, 1, p0; η = η))), maximum(abs.(Gr_k(ω0, -1, p0; η = η))))
    pz = with(p; θ = 0.0)
    S0 = Sr(ω0, pz; η = η)
    e6b = max(maximum(abs.(Gr_k(ω0, 0, pz; η = η) - Diagonal([1 / (S0 + pz.J_sd), 1 / (S0 - pz.J_sd)]))),
              maximum(abs.(Gl_k(ω0, 1, pz; η = η))), maximum(abs.(Gl_k(ω0, -1, pz; η = η))))
    @printf("6. J_sd=0: %.1e [%s]   θ=0: %.1e [%s]\n", e6a, ok(e6a < 1e-12), e6b, ok(e6b < 1e-12))
    pa = with(p; Ω = 1e-10)
    σM = sin(p.θ) * σx + cos(p.θ) * σz
    e6c = e6d = 0.0
    for ω in range(-2.5, 2.5; length = 251)
        Grs = sum(Gr_k(ω, k, pa; η = η) for k in -1:1)
        Gas = sum(Ga_k(ω, k, pa; η = η) for k in -1:1)
        Gls = sum(Gl_k(ω, k, pa; η = η) for k in -1:1)
        e6c = max(e6c, relerr(Grs, inv(Sr(ω, pa; η = η) * σ0 + pa.J_sd * σM)))
        e6d = max(e6d, relerr(Gls, fermi(ω, pa) * (Gas - Grs)))
    end
    @printf("   Ω→0: Σₖ𝒢ʳₖ = (Sʳ + J_sd σ·M(0))⁻¹: %.1e [%s]   FDT Σₖ𝒢<ₖ = f(𝒢ᵃ-𝒢ʳ): %.1e [%s]\n",
            e6c, ok(e6c < 1e-6), e6d, ok(e6d < 1e-6))

    # 7. ρₖ analitico vs numerico (misma malla)
    ωmin, ωmax, Nω = -3.0, 3.0, 1201
    ηρ  = 2 * (ωmax - ωmin) / (Nω - 1)
    ρa  = rho_harmonics(p; η = ηρ, ωmin = ωmin, ωmax = ωmax, Nω = Nω)
    ρn  = Dict(k => zeros(ComplexF64, 2, 2) for k in -1:1)
    ωg  = range(ωmin, ωmax; length = Nω)
    for ω in ωg
        _, _, Gl, q = num_ref(ω, p; η = ηρ, N = 1)
        for k in -1:1
            ρn[k] .+= FGF.fget(Gl, k, 0, q)
        end
    end
    for k in -1:1
        ρn[k] .*= -im * step(ωg) / (2π)
    end
    e7  = maximum(maximum(abs.(ρa[k] - ρn[k])) for k in -1:1)
    eh  = max(maximum(abs.(ρa[-1] - adjoint(ρa[1]))), maximum(abs.(ρa[0] - adjoint(ρa[0]))))
    @printf("7. ρₖ analitico vs Sambe: %.1e [%s]   ρ₋₁ = ρ₊₁†, ρ₀ = ρ₀†: %.1e [%s]\n",
            e7, ok(e7 < 1e-12), eh, ok(eh < 1e-6))
    ρold = FGF.rho_harmonic(to_fgf(p; N = 2); η = ηρ, ωmin = ωmin, ωmax = ωmax, Nω = Nω, n = 0)
    @printf("   (info) |ρ₀ - ρ₀(floquet_gf.jl)| = %.1e   (difieren en O(η): ese script no lleva el +iη de la diagonal)\n",
            maximum(abs.(ρa[0] - ρold)))

    # 8. positividad
    lmin = minimum(min(ldos(ω, p; η = η)...) for ω in ωl)
    smin = minimum(imag(Σless(ω, p; η = η)) for ω in ωl)
    sre  = maximum(abs(real(Σless(ω, p; η = η))) for ω in ωl)
    n0   = occ(ρa[0])
    @printf("8. LDOS ≥ 0: %.1e [%s]   Σ< imaginaria (|Re|=%.0e), Im ≥ 0: %.1e [%s]   0 ≤ n ≤ 2: n=%.6f [%s]\n",
            lmin, ok(lmin > -1e-12), sre, smin, ok(smin > -1e-14 && sre == 0), n0, ok(0 ≤ n0 ≤ 2))
    println("="^76)
    return nothing
end

# ---------------------------------------------------------------- barrido
function sweep(p::AParams = AParams(); ωmin = -3.0, ωmax = 3.0, Nω = 6001, η = nothing,
               Nper = 3, Nt = 601)
    ωs = range(ωmin, ωmax; length = Nω)
    dω = step(ωs)
    ηe = isnothing(η) ? 2dω : η
    @printf("sweep: Nω=%d  dω=%.3e  η=%.3e  (η/Ω=%.2f)\n", Nω, dω, ηe, ηe / p.Ω)

    # 1. LDOS(ω), -i𝒢<₀ y armonicos k=±1 de 𝒢ʳ
    hdr1 = ["omega", "LDOS_up", "LDOS_dn", "LDOS_tot", "ReGless00_uu", "ReGless00_dd",
            "ReGr_p1_updn", "ImGr_p1_updn", "ReGr_m1_dnup", "ImGr_m1_dnup"]
    data1 = Matrix{Float64}(undef, Nω, length(hdr1))
    for (i, ω) in enumerate(ωs)
        h = harm(ω, p, Sr; η = ηe)
        L = lesser(ω, p; η = ηe)
        up, dn = -imag(h.uu0) / π, -imag(h.dd0) / π
        data1[i, :] = [ω, up, dn, up + dn, real(-im * L.uu0), real(-im * L.dd0),
                       real(h.ud1), imag(h.ud1), real(h.du_1), imag(h.du_1)]
    end
    path1 = joinpath(OUT, "fa_floquet_ldos.csv")
    writedlm(path1, vcat(permutedims(hdr1), data1), ",")
    println("  -> ", path1)

    # 2. ρ(t) y espin(t)
    ts = spin_time_series(p; η = ηe, ωmin = ωmin, ωmax = ωmax, Nω = Nω, Nper = Nper, Nt = Nt)
    hdr2 = ["t", "n_up", "n_dn", "Re_rho_updn", "Im_rho_updn", "sx", "sy", "sz", "n_tot"]
    data2 = Matrix{Float64}(undef, Nt, length(hdr2))
    for (i, ρt) in enumerate(ts.ρ)
        data2[i, :] = [ts.t[i], real(ρt[1, 1]), real(ρt[2, 2]), real(ρt[1, 2]), imag(ρt[1, 2]),
                       ts.sx[i], ts.sy[i], ts.sz[i], ts.n[i]]
    end
    path2 = joinpath(OUT, "fa_floquet_rho_t.csv")
    writedlm(path2, vcat(permutedims(hdr2), data2), ",")
    println("  -> ", path2)

    ρ0, ρ1 = ts.ρks[0], ts.ρks[1]
    @printf("\n  ρ DC (k=0): n=%.8f   ⟨σx,σy,σz⟩ = (%+.3e, %+.3e, %+.3e)\n", occ(ρ0), spin(ρ0)...)
    @printf("  ρ k=1     :            ⟨σx,σy,σz⟩ = (%+.3e, %+.3e, %+.3e)\n", spin(ρ1)...)
    @printf("  ρ(t=0)    : n=%.8f   ⟨σx,σy,σz⟩ = (%+.3e, %+.3e, %+.3e)\n",
            ts.n[1], ts.sx[1], ts.sy[1], ts.sz[1])
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    p = AParams()
    validate(p)
    println()
    sweep(p)
end
