#!/usr/bin/env julia
#=
fully_analitical_inbedding_leads.jl -- funciones de Green del lead DERECHO con
precesion alrededor de y y Rashba en σy, calculadas SOLO con las formulas
cerradas (sectores de Q = n - σy/2). Lo unico numerico son las integrales en ω
que llevan los armonicos a la matriz densidad, ρₖ = -i ∫dω/2π 𝒢<ₖ(ω).

Convenciones (las del texto):
    T̂ = -γ σ0 - iγ_SO σy = -γ̃ e^{iθ_SO σy},  γ̃ = sqrt(γ²+γ_SO²)
    sitios del lead derecho m,n ≥ 1 (1 = superficie, vecino de la impureza)
    centro: 𝒢_{C,0} = A σ0 + B σy,  𝒢_{C,±1} = A_{±1} τ±,  τ± = (σz ± iσx)/2
    lead:   𝒢_{mn,0} = g_{mn} + [Σ P 𝒢_{C,0}] e^{i(n-m)θ_SO σy}
            𝒢_{mn,±1} = e^{∓i(m+n)θ_SO} Σ P 𝒢_{C,±1}
Base de espin para todos los observables: (σx, σy, σz).

Calcula, en los sitios 1 y 2:
    1. ρ₁₁(t), ρ₂₂(t) (y ρ₁₂, ρ₂₁ para el enlace)
    2. ⟨σ^α⟩₁(t), ⟨σ^α⟩₂(t)
    3. corriente de carga y de espin del enlace 1 -> 2
y su error relativo frente a la diagonalizacion exacta de Floquet (embedding de
Sambe de inbedding_leads.jl con PREC_AXIS = SOC_AXIS = "y").

Salidas (output/, mismo formato que sweep_leads de inbedding_leads.jl, sitios
con la numeracion del codigo: 0 = sitio 1, 1 = sitio 2; enlace "0,1"):
    fa_inbedding_ldos.csv, fa_inbedding_rho_t.csv,
    fa_inbedding_bond_rho_t.csv, fa_inbedding_bond_H.csv
=#

using LinearAlgebra
using Printf
using DelimitedFiles

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# referencia numerica (solo para los tests), aislada para no chocar nombres
module IL
    include(joinpath(@__DIR__, "inbedding_leads.jl"))
end
IL.PREC_AXIS[] = "y"
IL.SOC_AXIS[]  = "y"

const σ0 = ComplexF64[1 0; 0 1]
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const τp = (σz + im * σx) / 2
const τm = (σz - im * σx) / 2
const PAULI = (σx, σy, σz)
const PAIRS = ((1, 1), (2, 2), (1, 2), (2, 1))     # sitios del lead (texto: m,n ≥ 1)

# ---------------------------------------------------------------- parametros
# Mismos valores por defecto que floquet_gf.jl (t fijado para que γ̃ = 1).
Base.@kwdef struct AParams
    λ::Float64    = 0.1              # γ_SO
    t::Float64    = sqrt(1 - λ^2)    # γ
    J_sd::Float64 = 0.2
    θ::Float64    = deg2rad(10.0)    # angulo del cono
    Ω::Float64    = 0.005
    μ::Float64    = 0.0
    β::Float64    = 40.0
end

γt(p::AParams)  = sqrt(p.t^2 + p.λ^2)
θSO(p::AParams) = atan(p.λ, p.t)
T_hop(p::AParams) = -p.t * σ0 - im * p.λ * σy
Rot(φ) = cos(φ) * σ0 + im * sin(φ) * σy           # e^{iφσy}
fermi(ω, p::AParams) = 1 / (1 + exp(clamp(p.β * (ω - p.μ), -500.0, 500.0)))
to_il(p::AParams; N::Int) = IL.FloquetParams(; λ = p.λ, t = p.t, J_sd = p.J_sd, θ = p.θ,
                                              Ω = p.Ω, μ = p.μ, β = p.β, N = N)

# ---------------------------------------------------------------- S, Σ<
"Sʳ(ω) = sqrt((ω+iη)² - 4γ̃²), rama con Im Sʳ ≥ 0."
function Sr(ω::Real, p::AParams; η::Real)
    z = complex(ω, η); g2 = 2γt(p)
    return sqrt(z - g2) * sqrt(z + g2)
end
Sa(ω::Real, p::AParams; η::Real) = conj(Sr(ω, p; η = η))
"Σ<(ω) = 2i f(ω) Im Sʳ(ω)."
Σless(ω::Real, p::AParams; η::Real) = 2im * fermi(ω, p) * imag(Sr(ω, p; η = η))

# ---------------------------------------------------------------- zona central
"""
Escalares del centro para S = Sr (retardada) o Sa (avanzada):
    x = (d₋₁-Jc)/Δ₋₁,  y = (d₁+Jc)/Δ₀,  p1 = -Js/Δ₀,  m1 = -Js/Δ₋₁
con A_{C,0} = (x+y)/2, B_{C,0} = (x-y)/2, A_{C,+1} = p1, A_{C,-1} = m1.
"""
function cen(ω::Real, p::AParams, S::Function; η::Real)
    Jc, Js = p.J_sd * cos(p.θ), p.J_sd * sin(p.θ)
    dm1, d0, d1 = S(ω - p.Ω, p; η = η), S(ω, p; η = η), S(ω + p.Ω, p; η = η)
    Δ0  = (d0  - Jc) * (d1 + Jc) - Js^2
    Δm1 = (dm1 - Jc) * (d0 + Jc) - Js^2
    return (x = (dm1 - Jc) / Δm1, y = (d1 + Jc) / Δ0, p1 = -Js / Δ0, m1 = -Js / Δm1)
end

"Escalares de la lesser del centro (X<↑ = x, X<↓ = y, A<₊₁ = p1, A<₋₁ = m1)."
function cen_less(ω::Real, p::AParams; η::Real)
    Ω  = p.Ω
    r0 = cen(ω, p, Sr; η = η); rp = cen(ω + Ω, p, Sr; η = η); rm = cen(ω - Ω, p, Sr; η = η)
    a0 = cen(ω, p, Sa; η = η)
    s0, sp, sm = Σless(ω, p; η = η), Σless(ω + Ω, p; η = η), Σless(ω - Ω, p; η = η)
    return (x  = s0 * r0.x  * a0.x + sm * rm.p1 * a0.m1,
            y  = s0 * r0.y  * a0.y + sp * rp.m1 * a0.p1,
            p1 = s0 * r0.p1 * a0.y + sp * rp.x  * a0.p1,
            m1 = s0 * r0.m1 * a0.x + sm * rm.y  * a0.m1)
end

coef(c) = (A0 = (c.x + c.y) / 2, B0 = (c.x - c.y) / 2, Ap = c.p1, Am = c.m1)

"Coeficientes A, B del centro para r, a y < en ω."
central(ω::Real, p::AParams; η::Real) =
    (r = coef(cen(ω, p, Sr; η = η)), a = coef(cen(ω, p, Sa; η = η)), l = coef(cen_less(ω, p; η = η)))

"Armonico k del centro como matriz 2x2 (base σz)."
function Cmat(c, k::Int)
    k ==  0 && return c.A0 * σ0 + c.B0 * σy
    k ==  1 && return c.Ap * τp
    k == -1 && return c.Am * τm
    return zeros(ComplexF64, 2, 2)
end

# ---------------------------------------------------------------- lead derecho
"g₁₁ʳ(ω) = [(ω+iη) - Sʳ(ω)]/(2γ̃²)."
g11r(ω::Real, p::AParams; η::Real) = (complex(ω, η) - Sr(ω, p; η = η)) / (2γt(p)^2)

"Escalar c_{mn} de la recurrencia: g^{|m-n|+1} Σ_{q=0}^{min(m,n)-1} (γ̃ g)^{2q}."
c_mn(g, m::Int, n::Int, p::AParams) =
    g^(abs(m - n) + 1) * sum((γt(p) * g)^(2q) for q in 0:(min(m, n) - 1))

"Lead libre: g_{mn} = (-γ̃)^{|m-n|} c_{mn} e^{i(n-m)θ_SO σy}  (comp = :r, :a, :<)."
function glead(ω::Real, m::Int, n::Int, p::AParams; η::Real, comp::Symbol)
    pre = (-γt(p))^abs(m - n) * Rot((n - m) * θSO(p))
    cr  = c_mn(g11r(ω, p; η = η), m, n, p)
    comp === :r && return cr * pre
    comp === :a && return conj(cr) * pre
    return -2im * fermi(ω, p) * imag(cr) * pre
end

"Escalares P^{(μ,λ)}_k(m,n)."
function Pk(ω::Real, k::Int, m::Int, n::Int, p::AParams; η::Real)
    pre = (-γt(p))^(m + n)
    grk = g11r(ω + k * p.Ω, p; η = η)
    gr  = g11r(ω, p; η = η); ga = conj(gr)
    return (rr = pre * grk^m * gr^n,
            la = -2im * fermi(ω + k * p.Ω, p) * pre * imag(grk^m) * ga^n,
            ra = pre * grk^m * ga^n,
            rl = -2im * fermi(ω, p) * pre * grk^m * imag(gr^n))
end

"""
Armonicos k = 0, ±1 de la retardada y la lesser del lead en (m,n):
devuelve (Gr = Dict(k => 2x2), Gl = Dict(k => 2x2)).
"""
function lead_harm(ω::Real, m::Int, n::Int, p::AParams; η::Real, C = central(ω, p; η = η))
    R  = Rot((n - m) * θSO(p))
    P0 = Pk(ω, 0, m, n, p; η = η)
    Gr = Dict(0 => glead(ω, m, n, p; η = η, comp = :r) + P0.rr * Cmat(C.r, 0) * R)
    Gl = Dict(0 => glead(ω, m, n, p; η = η, comp = :<) +
                   (P0.la * Cmat(C.a, 0) + P0.ra * Cmat(C.l, 0) + P0.rl * Cmat(C.r, 0)) * R)
    for s in (1, -1)
        P  = Pk(ω, s, m, n, p; η = η)
        ph = exp(-im * s * (m + n) * θSO(p))
        Gr[s] = ph * P.rr * Cmat(C.r, s)
        Gl[s] = ph * (P.la * Cmat(C.a, s) + P.ra * Cmat(C.l, s) + P.rl * Cmat(C.r, s))
    end
    return (Gr = Gr, Gl = Gl)
end

# ---------------------------------------------------------------- observables
"ρ_{mn,k} = -i ∫dω/2π 𝒢<_{mn,k}(ω) y 𝒢ʳ_{mm,0} (para la LDOS) en la malla."
function rho_harmonics(p::AParams; η::Real, ωmin::Real, ωmax::Real, Nω::Int)
    ωs  = range(ωmin, ωmax; length = Nω); dω = step(ωs)
    acc = Dict((mn..., k) => zeros(ComplexF64, 2, 2) for mn in PAIRS, k in -1:1)
    for ω in ωs
        C = central(ω, p; η = η)
        for (m, n) in PAIRS
            h = lead_harm(ω, m, n, p; η = η, C = C)
            for k in -1:1
                acc[(m, n, k)] .+= h.Gl[k]
            end
        end
    end
    return Dict(key => -im .* v .* dω ./ (2π) for (key, v) in acc)
end

rho_of_t(t, ρ, m, n, Ω) = sum(exp(-im * k * Ω * t) .* ρ[(m, n, k)] for k in -1:1)
spin(ρ) = [real(tr(s * ρ)) for s in PAULI]
occ(ρ)  = real(tr(ρ))

"""
Armonicos de la corriente del enlace 1 -> 2 (misma definicion que bond_currents
de inbedding_leads_plot.py): X = H₁₂ ρ^{12} - ρ^{21} H₂₁, ρ^{ij} = ⟨c_i†c_j⟩,
con ρ^{12} = ρ₂₁ y ρ^{21} = ρ₁₂ en la convencion ρ_{mn} = ⟨c_n†c_m⟩.
I = -i Tr X,  I^{S_α} = -i Tr σ^α X.  Devuelve Dict(k => [I, Ix, Iy, Iz]).
"""
function current_harmonics(ρ, p::AParams)
    H = T_hop(p)
    out = Dict{Int, Vector{ComplexF64}}()
    for k in -1:1
        X = H * ρ[(2, 1, k)] - ρ[(1, 2, k)] * adjoint(H)
        out[k] = vcat(-im * tr(X), [-im * tr(s * X) for s in PAULI])
    end
    return out
end

spin_harmonics(ρ) = Dict((m, k) => [tr(s * ρ[(m, m, k)]) for s in PAULI] for m in (1, 2), k in -1:1)

# ---------------------------------------------------------------- referencia
"""
Referencia por diagonalizacion exacta: embedding de Sambe de inbedding_leads.jl
(lead :R, T_inout, dress), con PREC_AXIS = SOC_AXIS = "y" y la MISMA convencion
de la derivacion: diagonal central ω+nΩ+iη-Σʳ = Sʳ(ω+nΩ) (se suma iη a
floquet_matrix) y Σ̌< central con Σ< = 2if Im Sʳ.
Devuelve Dict((m,n) => (Gr, Gl)) con las matrices de Sambe completas, y q.
"""
function ref_lead(ω::Real, p::AParams; η::Real, N::Int, pairs = PAIRS)
    q  = to_il(p; N = N); nf = IL.n_floquet(q)
    GrC = inv(IL.floquet_matrix(ω, q; η = η) + im * η * I)
    S  = zeros(ComplexF64, 2nf, 2nf)
    for j in -N:N
        r = IL.fblock(IL.fidx(j, q)); S[r, r] = Σless(ω + j * p.Ω, p; η = η) * σ0
    end
    GlC = GrC * S * adjoint(GrC)
    Tin, Tout = IL.T_inout(:R, q)
    Ti = kron(Matrix{ComplexF64}(I, nf, nf), Tin); To = kron(Matrix{ComplexF64}(I, nf, nf), Tout)
    Σr, Σa, Σl = Ti * GrC * To, Ti * adjoint(GrC) * To, Ti * GlC * To
    d(a, b, c) = IL.dress(ω, a, b, :R, q; η = η, comp = c)
    res = Dict{Tuple{Int,Int}, Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()
    for (m, n) in pairs
        i, j = m - 1, n - 1                        # indices del codigo (0 = superficie)
        Gr = d(i, j, :r) + d(i, 0, :r) * Σr * d(0, j, :r)
        Gl = d(i, j, :<) + d(i, 0, :<) * Σa * d(0, j, :a) +
             d(i, 0, :r) * Σl * d(0, j, :a) + d(i, 0, :r) * Σr * d(0, j, :<)
        res[(m, n)] = (Gr, Gl)
    end
    return res, (GrC, GlC), q
end

function rho_harmonics_ref(p::AParams; η::Real, ωmin::Real, ωmax::Real, Nω::Int, N::Int = 1)
    ωs  = range(ωmin, ωmax; length = Nω); dω = step(ωs)
    acc = Dict((mn..., k) => zeros(ComplexF64, 2, 2) for mn in PAIRS, k in -1:1)
    for ω in ωs
        res, _, q = ref_lead(ω, p; η = η, N = N)
        for (m, n) in PAIRS, k in -1:1
            acc[(m, n, k)] .+= IL.fget(res[(m, n)][2], k, 0, q)
        end
    end
    return Dict(key => -im .* v .* dω ./ (2π) for (key, v) in acc)
end

# ---------------------------------------------------------------- tests
"Acumulador de error relativo global: max|a-b| / max|b| sobre todos los puntos."
mutable struct Err; d::Float64; s::Float64; end
Err() = Err(0.0, 0.0)
add!(e::Err, a, b) = (e.d = max(e.d, maximum(abs.(a .- b))); e.s = max(e.s, maximum(abs.(b))); e)
val(e::Err) = e.d / max(e.s, 1e-300)

function validate(p::AParams = AParams(); η::Real = 2e-3)
    ok(b) = b ? "OK" : "FALLA"
    println("="^78)
    @printf("LEAD DERECHO, precesion en y + Rashba en y   γ̃=%.3f θ_SO=%.4f J_sd=%.3f θ=%.1f° Ω=%.4f β=%.0f η=%.1e\n",
            γt(p), θSO(p), p.J_sd, rad2deg(p.θ), p.Ω, p.β, η)
    println("="^78)

    # 1. S, g11, Σ<
    ωl = range(-6, 6; length = 4001); q1 = to_il(p; N = 1)
    imin = minimum(imag(Sr(ω, p; η = η)) for ω in ωl)
    e1a = maximum(abs(complex(ω, η) - Sr(ω, p; η = η) - IL.Σr(ω, q1; η = η)) for ω in ωl)
    e1b = maximum(abs(g11r(ω, p; η = η) - IL.g00_lead(ω, q1; η = η)) for ω in ωl)
    @printf("1. min Im Sʳ = %.1e [%s]   z-Sʳ vs Σʳ: %.1e [%s]   g₁₁ʳ vs g00_lead: %.1e [%s]\n",
            imin, ok(imin > -1e-14), e1a, ok(e1a < 1e-12), e1b, ok(e1b < 1e-12))

    # 2. lead libre g^{r,a,<}_{mn} vs recurrencia de inbedding_leads.jl
    e2 = Dict(c => Err() for c in (:r, :a, :<))
    for ω in range(-3, 3; length = 241), m in 1:4, n in 1:4, c in (:r, :a, :<)
        ref = c === :r ? IL.g_lead_r(ω, m - 1, n - 1, :R, q1; η = η) :
              c === :a ? IL.g_lead_a(ω, m - 1, n - 1, :R, q1; η = η) :
                         IL.g_lead_less(ω, m - 1, n - 1, :R, q1; η = η)
        add!(e2[c], glead(ω, m, n, p; η = η, comp = c), ref)
    end
    @printf("2. lead libre (m,n ≤ 4):  gʳ %.1e [%s]   gᵃ %.1e [%s]   g< %.1e [%s]\n",
            val(e2[:r]), ok(val(e2[:r]) < 1e-12), val(e2[:a]), ok(val(e2[:a]) < 1e-12),
            val(e2[:<]), ok(val(e2[:<]) < 1e-12))

    # 3-4. centro y lead vs Sambe (N = 1 y N = 3), soporte |k| = 2
    ωs = range(-3, 3; length = 601)
    for N in (1, 3)
        eC = Dict(c => Err() for c in (:r, :a, :<))
        eL = Dict(c => Err() for c in (:r, :<))
        e2k = 0.0
        for ω in ωs
            res, (GrC, GlC), q = ref_lead(ω, p; η = η, N = N)
            C = central(ω, p; η = η)
            for k in -1:1
                add!(eC[:r], Cmat(C.r, k), IL.fget(GrC, k, 0, q))
                add!(eC[:a], Cmat(C.a, k), IL.fget(adjoint(GrC), k, 0, q))
                add!(eC[:<], Cmat(C.l, k), IL.fget(GlC, k, 0, q))
            end
            for (m, n) in PAIRS
                h = lead_harm(ω, m, n, p; η = η, C = C)
                for k in -1:1
                    add!(eL[:r], h.Gr[k], IL.fget(res[(m, n)][1], k, 0, q))
                    add!(eL[:<], h.Gl[k], IL.fget(res[(m, n)][2], k, 0, q))
                end
                if N ≥ 2
                    for k in (-2, 2)
                        e2k = max(e2k, maximum(abs.(IL.fget(res[(m, n)][1], k, 0, q))),
                                       maximum(abs.(IL.fget(res[(m, n)][2], k, 0, q))))
                    end
                end
            end
        end
        @printf("%s N=%d  centro: 𝒢ʳ %.1e [%s]  𝒢ᵃ %.1e [%s]  𝒢< %.1e [%s]\n",
                N == 1 ? "3." : "  ", N, val(eC[:r]), ok(val(eC[:r]) < 1e-10),
                val(eC[:a]), ok(val(eC[:a]) < 1e-10), val(eC[:<]), ok(val(eC[:<]) < 1e-10))
        @printf("%s N=%d  lead (sitios 1,2): 𝒢ʳ_{mn,k} %.1e [%s]  𝒢<_{mn,k} %.1e [%s]",
                N == 1 ? "4." : "  ", N, val(eL[:r]), ok(val(eL[:r]) < 1e-10),
                val(eL[:<]), ok(val(eL[:<]) < 1e-10))
        N ≥ 2 ? @printf("   max|Ǧₖ₀|, |k|=2: %.1e [%s]\n", e2k, ok(e2k < 1e-12)) : println()
    end

    # 5. observables integrados: errores relativos frente a la diagonalizacion exacta
    ωmin, ωmax, Nω = -3.0, 3.0, 1201
    ηρ = 2 * (ωmax - ωmin) / (Nω - 1)
    ρa = rho_harmonics(p; η = ηρ, ωmin = ωmin, ωmax = ωmax, Nω = Nω)
    ρn = rho_harmonics_ref(p; η = ηρ, ωmin = ωmin, ωmax = ωmax, Nω = Nω, N = 1)
    println("5. observables (Nω=$Nω, η=$(ηρ)), error relativo max|an-num|/max|num| sobre k=0,±1:")
    for (lab, key) in (("ρ₁₁", (1, 1)), ("ρ₂₂", (2, 2)), ("ρ₁₂", (1, 2)), ("ρ₂₁", (2, 1)))
        e = Err(); for k in -1:1; add!(e, ρa[(key..., k)], ρn[(key..., k)]); end
        @printf("     %s :  %.1e [%s]\n", lab, val(e), ok(val(e) < 1e-10))
    end
    sa, sn = spin_harmonics(ρa), spin_harmonics(ρn)
    for m in (1, 2)
        e = Err(); for k in -1:1; add!(e, sa[(m, k)], sn[(m, k)]); end
        @printf("     ⟨σ^α⟩ sitio %d :  %.1e [%s]\n", m, val(e), ok(val(e) < 1e-10))
    end
    ca, cn = current_harmonics(ρa, p), current_harmonics(ρn, p)
    for (c, lab) in enumerate(("I (carga)", "I^Sx", "I^Sy", "I^Sz"))
        e = Err(); for k in -1:1; add!(e, ca[k][c], cn[k][c]); end
        @printf("     %-10s enlace 1->2 :  %.1e [%s]\n", lab, val(e), ok(val(e) < 1e-10))
    end

    # 6. hermiticidad ρ_{mn,-k} = (ρ_{nm,k})†
    eh = maximum(maximum(abs.(ρa[(m, n, -k)] - adjoint(ρa[(n, m, k)]))) for (m, n) in PAIRS, k in -1:1)
    @printf("6. ρ_{mn,-k} = (ρ_{nm,k})†:  %.1e [%s]\n", eh, ok(eh < 1e-6))

    # 7. informativo: inbedding_leads.jl sin el +iη en la diagonal central
    ρold = IL.rho_harmonic_lead(to_il(p; N = 1); η = ηρ, ωmin = ωmin, ωmax = ωmax, Nω = Nω,
                                n = 0, lead = :R, k = 0)
    @printf("7. (info) |ρ₁₁,DC - inbedding_leads.jl sin ajuste| = %.1e  (O(η): diagonal sin +iη y Γ con -2η)\n",
            maximum(abs.(ρa[(1, 1, 0)] - ρold)))
    println("="^78)
    return nothing
end

# ---------------------------------------------------------------- barrido + CSV
function sweep(p::AParams = AParams(); ωmin = -3.0, ωmax = 3.0, Nω = 6001, η = nothing,
               Nper = 3, Nt = 601)
    ωs = range(ωmin, ωmax; length = Nω); dω = step(ωs)
    ηe = isnothing(η) ? 2dω : η
    @printf("sweep: Nω=%d  dω=%.3e  η=%.3e\n", Nω, dω, ηe)

    # LDOS y ρ en un solo barrido de ω
    hdr1  = ["omega", "lead", "site", "LDOS_up", "LDOS_dn", "LDOS_tot"]
    data1 = Matrix{Any}(undef, 2Nω, length(hdr1))
    acc   = Dict((mn..., k) => zeros(ComplexF64, 2, 2) for mn in PAIRS, k in -1:1)
    i1 = 0
    for ω in ωs
        C = central(ω, p; η = ηe)
        for (m, n) in PAIRS
            h = lead_harm(ω, m, n, p; η = ηe, C = C)
            for k in -1:1
                acc[(m, n, k)] .+= h.Gl[k]
            end
            if m == n
                up, dn = -imag(h.Gr[0][1, 1]) / π, -imag(h.Gr[0][2, 2]) / π
                i1 += 1
                data1[i1, :] = Any[ω, "R", m - 1, up, dn, up + dn]
            end
        end
    end
    ρ = Dict(key => -im .* v .* dω ./ (2π) for (key, v) in acc)
    path1 = joinpath(OUT, "fa_inbedding_ldos.csv")
    writedlm(path1, vcat(permutedims(hdr1), data1), ",")
    println("  -> ", path1)

    tgrid = collect(range(0, Nper * 2π / p.Ω; length = Nt))

    # ρ(t), espin y ocupacion en los sitios 1, 2 (etiquetas del codigo 0, 1)
    hdr2  = ["t", "lead", "site", "n_up", "n_dn", "Re_rho_updn", "Im_rho_updn",
             "sx", "sy", "sz", "n_tot"]
    data2 = Matrix{Any}(undef, 2Nt, length(hdr2))
    i2 = 0
    for m in (1, 2), t in tgrid
        ρt = rho_of_t(t, ρ, m, m, p.Ω)
        s  = spin(ρt)
        i2 += 1
        data2[i2, :] = Any[t, "R", m - 1, real(ρt[1, 1]), real(ρt[2, 2]),
                           real(ρt[1, 2]), imag(ρt[1, 2]), s[1], s[2], s[3], occ(ρt)]
    end
    path2 = joinpath(OUT, "fa_inbedding_rho_t.csv")
    writedlm(path2, vcat(permutedims(hdr2), data2), ",")
    println("  -> ", path2)

    # enlace (codigo n=0, m=1 = sitios 1 -> 2): bloques nm = ρ₁₂, mn = ρ₂₁
    ent  = ((1, 1), (1, 2), (2, 1), (2, 2))
    hdr3 = Any["t", "lead", "n", "m"]
    for blk in ("nm", "mn"), s in ("uu", "ud", "du", "dd"), c in ("Re", "Im")
        push!(hdr3, "$(c)_rho_$(blk)_$(s)")
    end
    data3 = Matrix{Any}(undef, Nt, length(hdr3))
    for (it, t) in enumerate(tgrid)
        A = rho_of_t(t, ρ, 1, 2, p.Ω); B = rho_of_t(t, ρ, 2, 1, p.Ω)
        row = Any[t, "R", 0, 1]
        for X in (A, B), (s1, s2) in ent
            push!(row, real(X[s1, s2])); push!(row, imag(X[s1, s2]))
        end
        data3[it, :] = row
    end
    path3 = joinpath(OUT, "fa_inbedding_bond_rho_t.csv")
    writedlm(path3, vcat(permutedims(hdr3), data3), ",")
    println("  -> ", path3)

    hdrH = Any["lead", "n", "m"]
    for blk in ("nm", "mn"), s in ("uu", "ud", "du", "dd"), c in ("Re", "Im")
        push!(hdrH, "$(c)_H_$(blk)_$(s)")
    end
    H = T_hop(p)
    row = Any["R", 0, 1]
    for X in (H, adjoint(H)), (s1, s2) in ent
        push!(row, real(X[s1, s2])); push!(row, imag(X[s1, s2]))
    end
    pathH = joinpath(OUT, "fa_inbedding_bond_H.csv")
    writedlm(pathH, vcat(permutedims(hdrH), permutedims(row)), ",")
    println("  -> ", pathH)

    println()
    for m in (1, 2)
        ρ0 = ρ[(m, m, 0)]
        @printf("  sitio %d :  ocup=%.6f   ⟨σx,σy,σz⟩ DC = (%+.3e, %+.3e, %+.3e)\n", m, occ(ρ0), spin(ρ0)...)
    end
    cur = current_harmonics(ρ, p)
    @printf("  enlace 1->2, DC:  I=%+.3e   I^S=(%+.3e, %+.3e, %+.3e)   (|Im| = %.1e)\n",
            real.(cur[0])..., maximum(abs.(imag.(cur[0]))))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    p = AParams()
    validate(p)
    println()
    sweep(p)
end
