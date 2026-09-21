#!/usr/bin/env julia

using LinearAlgebra
using Printf
using DelimitedFiles

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# Pauli matrices
const σ0 = ComplexF64[1 0; 0 1]
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const σp = (σx + im*σy) / 2      # σ₊
const σm = (σx - im*σy) / 2      # σ₋

# parametros
#
# t = sqrt(1-λ²) NO es arbitrario: hace que γ_band = sqrt(t²+λ²) = 1 exacto, o
# sea banda [-2,2]. Los leads de TDNEGF usan una tabla de polos precalculada
# (data/z_Semicircle_N49.txt) con borde de banda fijo en |ω|=2 y SIN parametro
# de escala -- el γ que se le pasa a build_Σᴸ_nλ solo entra por la dispersion
# transversal del liston, que es cero para Ny=1. Asi que la banda del lead no
# se puede mover; hay que ajustar la cadena para que coincida con ella.
# λ va primero para que t pueda referenciarla en su valor por defecto.
Base.@kwdef struct FloquetParams
    λ::Float64    = 0.1              # Rashba γ_SO
    t::Float64    = sqrt(1 - λ^2)    # hopping γ, fijado para que γ_band = 1
    J_sd::Float64 = 0.2
    θ::Float64    = deg2rad(10.0)    # angulo del cono
    Ω::Float64    = 0.005
    μ::Float64    = 0.0              # E_F
    β::Float64    = 40.0
    N::Int        = 2                # truncacion de Floquet
end

γ_band(p::FloquetParams)    = sqrt(p.t^2 + p.λ^2)
n_floquet(p::FloquetParams) = 2p.N + 1
@inline fidx(n::Int, p::FloquetParams) = n + p.N + 1
@inline fblock(j::Int) = (2j - 1):(2j)
fget(G, n::Int, m::Int, p::FloquetParams) = G[fblock(fidx(n, p)), fblock(fidx(m, p))]

# leads self-energy
"Autoenergia retardada de los dos leads sumados: Σʳ"
function Σr(ω::Real, p::FloquetParams; η::Real)
    γ = γ_band(p)
    z = complex(ω, η)
    s = sqrt(z^2 - 4γ^2)
    Σ = z - s
    return imag(Σ) < 0 ? Σ : z + s
end

"Forma factorizada, rama correcta sin condicionales. Solo para verificar."
function Σr_fact(ω::Real, p::FloquetParams; η::Real)
    γ = γ_band(p)
    z = complex(ω, η)
    return z - sqrt(z - 2γ) * sqrt(z + 2γ)
end

Γ_lead(ω, p; η)  = -2 * imag(Σr(ω, p; η = η))
fermi(ω, p)      = 1 / (1 + exp(clamp(p.β * (ω - p.μ), -500.0, 500.0)))
Σless(ω, p; η)   = im * fermi(ω, p) * Γ_lead(ω, p; η = η)   # = i f Γ

# ---------------------------------------------------------------- Floquet
"""
Matriz [ωI + Ω̌ + J_sd M̌ - Σ̌ʳ]. Bloque-tridiagonal, 2N_F x 2N_F.
El signo +J_sd M̌ corresponde a H_sd = -J_sd σ·M (convenio TDNEGF).
"""
function floquet_matrix(ω::Real, p::FloquetParams; η::Real)
    nf  = n_floquet(p)
    A   = zeros(ComplexF64, 2nf, 2nf)
    M0  = p.J_sd * cos(p.θ) * σz
    Mp1 = p.J_sd * sin(p.θ) * σm     # M̂₊₁
    Mm1 = p.J_sd * sin(p.θ) * σp     # M̂₋₁
    for n in -p.N:p.N
        r  = fblock(fidx(n, p))
        ωn = ω + n * p.Ω
        A[r, r] = (ωn - Σr(ωn, p; η = η)) * σ0 + M0
        n <  p.N && (A[r, fblock(fidx(n + 1, p))] = Mp1)
        n > -p.N && (A[r, fblock(fidx(n - 1, p))] = Mm1)
    end
    return A
end


Gr_C(ω, p; η) = inv(floquet_matrix(ω, p; η = η))
Ga_C(ω, p; η) = adjoint(Gr_C(ω, p; η = η))

function Σless_floquet(ω::Real, p::FloquetParams; η::Real)
    nf = n_floquet(p)
    S  = zeros(ComplexF64, 2nf, 2nf)
    for n in -p.N:p.N
        r = fblock(fidx(n, p))
        S[r, r] = Σless(ω + n * p.Ω, p; η = η) * σ0
    end
    return S
end

"Ǧ< = Ǧʳ Σ̌< Ǧᵃ  (vale porque ρ(t=0)=0, asi que ǧ<_C = 0)."
function Gless_C(ω::Real, p::FloquetParams; η::Real)
    G = Gr_C(ω, p; η = η)
    return G * Σless_floquet(ω, p; η = η) * adjoint(G)
end

# forma exacta 2x2
#=
  El problema tiene una simetria U(1) exacta: q = n - s/2 se conserva, porque
  M̂₊₁ ∝ σ₋ solo conecta |n,↓⟩ con |n+1,↑⟩. La escalera de Floquet se parte en
  sectores 2x2 independientes y Ǧʳ tiene soporte SOLO en |n-m| ≤ 1.
=#

"Matriz 2x2 del sector {|n,↓⟩, |n+1,↑⟩}, en ese orden."
function sector2x2(ω::Real, n::Int, p::FloquetParams; η::Real)
    Jc = p.J_sd * cos(p.θ)
    Js = p.J_sd * sin(p.θ)
    d(W) = W - Σr(W, p; η = η)
    return ComplexF64[d(ω + n*p.Ω)-Jc             Js;
                       Js              d(ω + (n+1)*p.Ω)+Jc]
end

"Bloque Ǧʳ₀₀ en forma cerrada."
function G00_exact(ω::Real, p::FloquetParams; η::Real)
    up = inv(sector2x2(ω, -1, p; η = η))[2, 2]   # |0,↑⟩ es el 2º del sector n=-1
    dn = inv(sector2x2(ω,  0, p; η = η))[1, 1]   # |0,↓⟩ es el 1º del sector n=0
    return ComplexF64[up 0; 0 dn]
end

# observables
# 1. LDOS resuelta en espin: A(ω) = -1/π Im Ǧʳ(ω)
"""
    ldos(ω, p; η) -> Matrix{Float64} (2x2)

Densidad local de estados del sitio manejado, resuelta en espin:
    A(ω) = -1/π Im Ǧʳ₀₀(ω)
La entrada [1,1] (base ↑,↓) es la LDOS de espin ↑, [2,2] la de espin ↓
(el bloque es exactamente diagonal, sin coherencia ↑↓, por la simetria
U(1) q=n-s/2). La densidad electronica total es la suma de ambas.
"""
function ldos(ω::Real, p::FloquetParams; η::Real)
    G = fget(Gr_C(ω, p; η = η), 0, 0, p)
    return @. -imag(G) / π
end

ldos_up(ω, p::FloquetParams; η::Real)  = real(ldos(ω, p; η = η)[1, 1])
ldos_dn(ω, p::FloquetParams; η::Real)  = real(ldos(ω, p; η = η)[2, 2])
ldos_tot(ω, p::FloquetParams; η::Real) = ldos_up(ω, p; η = η) + ldos_dn(ω, p; η = η)

# 2. Matriz densidad y espin en el tiempo, a partir de Ǧ<
"""
    rho_harmonic(p; η, ωmin, ωmax, Nω, n=0)

Armonico k=n de la matriz densidad en el sitio manejado,
ρ(t) = Σₖ e^{-ikΩt} ρₖ,  con  ρₖ = -i ∫ dω/2π  Ǧ<ₖ₀(ω).
n=0 es la parte DC. (Indice fila=k, columna=0: la invarianza de corrimiento
Ǧₙₘ(ω) = Ǧₙ₋ₘ,₀(ω+mΩ) colapsa la suma sobre replicas de Floquet a un solo
termino con columna fija en 0.)
"""
function rho_harmonic(p::FloquetParams; η::Real, ωmin::Real, ωmax::Real,
                      Nω::Int, n::Int = 0)
    ωs  = range(ωmin, ωmax; length = Nω)
    dω  = step(ωs)
    acc = zeros(ComplexF64, 2, 2)
    for ω in ωs
        acc .+= fget(Gless_C(ω, p; η = η), n, 0, p)
    end
    return -im .* acc .* dω ./ (2π)
end

"""
    rho_harmonics(p; η, ωmin, ωmax, Nω, Kmax=min(2,p.N)) -> Dict{Int,Matrix{ComplexF64}}

Todos los armonicos ρₖ, k=-Kmax:Kmax. El soporte |k|≤2 es EXACTO e
independiente de la truncacion N (viene de Ǧ<=ǦʳΣ̌<Ǧᵃ, doble convolucion
que alcanza |n-m|≤2 dado el soporte |n-m|≤1 de Ǧʳ) -- no crece con N.
Pero el indice de fila k tiene que caber en el rango fisico -N:N de la
matriz truncada, de ahi el min(2,p.N): con N=1 solo se pueden extraer
ρ₋₁,ρ₀,ρ₁ (ρ±₂ existen fisicamente pero exigen N≥2 para representarlos).
"""
function rho_harmonics(p::FloquetParams; η::Real, ωmin::Real, ωmax::Real,
                        Nω::Int, Kmax::Int = min(2, p.N))
    return Dict(k => rho_harmonic(p; η = η, ωmin = ωmin, ωmax = ωmax, Nω = Nω, n = k)
                for k in -Kmax:Kmax)
end

"ρ(t) = Σₖ e^{-ikΩt} ρₖ, reconstruida a partir de los armonicos `ρks`."
function rho_of_t(t::Real, ρks::Dict{Int, Matrix{ComplexF64}}, Ω::Real)
    ρt = zeros(ComplexF64, 2, 2)
    for (k, ρk) in ρks
        ρt .+= exp(-im * k * Ω * t) .* ρk
    end
    return ρt
end

spin(ρ) = (real(tr(σx * ρ)), real(tr(σy * ρ)), real(tr(σz * ρ)))
occ(ρ)  = real(tr(ρ))

"""
    spin_time_series(p; η, ωmin, ωmax, Nω, Kmax=min(2,p.N), Nper=3, Nt=601)

Matriz densidad ρ(t) y valores esperados de espin ⟨σx,σy,σz⟩(t) mas la
ocupacion n(t) del sitio manejado, sobre Nper periodos t∈[0,Nper·2π/Ω).
Devuelve un NamedTuple (t, ρ, sx, sy, sz, n), cada campo un vector de
longitud Nt (ρ es un vector de matrices 2x2).

ρ(t) es EXACTAMENTE T-periodica (es el estado estacionario de Floquet, no
hay transitorio), asi que Nper>1 solo repite copias identicas: no agrega
informacion, pero hace visible la periodicidad en la figura. El costo es
despreciable -- los armonicos ρₖ se calculan una sola vez y lo unico que
crece es el lazo de reconstruccion.
"""
function spin_time_series(p::FloquetParams; η::Real, ωmin::Real, ωmax::Real,
                           Nω::Int, Kmax::Int = min(2, p.N),
                           Nper::Int = 3, Nt::Int = 601)
    ρks = rho_harmonics(p; η = η, ωmin = ωmin, ωmax = ωmax, Nω = Nω, Kmax = Kmax)
    T   = 2π / p.Ω
    ts  = collect(range(0, Nper * T; length = Nt))
    ρts = [rho_of_t(t, ρks, p.Ω) for t in ts]
    return (t  = ts,
            ρ  = ρts,
            sx = [spin(ρt)[1] for ρt in ρts],
            sy = [spin(ρt)[2] for ρt in ρts],
            sz = [spin(ρt)[3] for ρt in ρts],
            n  = [occ(ρt)     for ρt in ρts])
end

#  validaciones
function validate(p::FloquetParams = FloquetParams(); η::Real = 2e-3)
    ok(b) = b ? "OK" : "FALLA"
    println("="^72)
    @printf("t=%.3f  λ=%.3f  J_sd=%.3f  θ=%.1f°  Ω=%.4f  β=%.1f  N=%d\n",
            p.t, p.λ, p.J_sd, rad2deg(p.θ), p.Ω, p.β, p.N)
    @printf("γ=%.6f   banda=[%.4f, %.4f]   Γ(0)=%.4f   Γ/Ω=%.1f\n",
            γ_band(p), -2γ_band(p), 2γ_band(p), Γ_lead(0.0, p; η = η),
            Γ_lead(0.0, p; η = η) / p.Ω)
    println("="^72)

    ωs = range(-6, 6; length = 4001)

    # 1. rama
    nbad = count(ω -> imag(Σr(ω, p; η = η)) > 1e-14, ωs)
    e1   = maximum(abs(Σr(ω, p; η = η) - Σr_fact(ω, p; η = η)) for ω in ωs)
    @printf("1. rama:  Im Σʳ>0 en %d/%d puntos [%s];  |criterio - factorizada| = %.2e [%s]\n",
            nbad, length(ωs), ok(nbad == 0), e1, ok(e1 < 1e-10))

    # 2. J_sd = 0
    p0 = FloquetParams(; J_sd = 0.0, N = p.N)
    ω  = 0.37
    G  = Gr_C(ω, p0; η = η)
    ed = maximum(abs.(fget(G, n, n, p0) - inv(ω + n*p0.Ω - Σr(ω + n*p0.Ω, p0; η=η)) * σ0) |> maximum
                 for n in -p0.N:p0.N)
    eo = maximum((n == m ? 0.0 : maximum(abs.(fget(G, n, m, p0))))
                 for n in -p0.N:p0.N, m in -p0.N:p0.N)
    @printf("2. J_sd=0:  |diag-analitico|=%.2e  |fuera diag|=%.2e  [%s]\n",
            ed, eo, ok(max(ed, eo) < 1e-12))

    # 3. θ = 0
    pz = FloquetParams(; θ = 0.0, N = p.N)
    G  = Gr_C(ω, pz; η = η)
    ed = maximum(maximum(abs.(fget(G, n, n, pz) -
            inv((ω + n*pz.Ω - Σr(ω + n*pz.Ω, pz; η=η)) * σ0 + pz.J_sd * σz)))
                 for n in -pz.N:pz.N)
    @printf("3. θ=0:     |diag - analitico 2x2| = %.2e  [%s]\n", ed, ok(ed < 1e-12))

    # 4. positividad
    worst = minimum(minimum(real(eigvals(Hermitian((ldos(ω, p; η=η) + adjoint(ldos(ω, p; η=η)))/2))))
                    for ω in range(-2.5, 2.5; length = 401))
    @printf("4. LDOS ⪰ 0:  autovalor minimo = %.3e  [%s]\n", worst, ok(worst > -1e-10))

    # 5. forma cerrada 2x2
    e5 = maximum(maximum(abs.(fget(Gr_C(ω, p; η=η), 0, 0, p) - G00_exact(ω, p; η=η)))
                 for ω in range(-2.5, 2.5; length = 501))
    @printf("5. Ǧʳ₀₀ vs forma cerrada 2x2:  %.2e  [%s]\n", e5, ok(e5 < 1e-11))

    # 6. soporte en el indice de Floquet
    println("6. soporte |n-m|:")
    Gr_ = Gr_C(0.31, p; η = η); Gl_ = Gless_C(0.31, p; η = η)
    for k in 0:min(3, 2p.N)
        a = maximum(maximum(abs.(fget(Gr_, n, n+k, p))) for n in -p.N:(p.N-k))
        b = maximum(maximum(abs.(fget(Gl_, n, n+k, p))) for n in -p.N:(p.N-k))
        @printf("     |n-m|=%d :  max|Ǧʳ|=%.3e   max|Ǧ<|=%.3e\n", k, a, b)
    end
    println("   (deben anularse para |n-m|≥2: simetria U(1) q = n - s/2)")

    # 7. convergencia en N
    # η atado a ESTA malla (η=2dω, igual que en sweep): con η < dω la
    # singularidad de van Hove del borde de banda (1D, ~1/sqrt) queda
    # sin muestrear y falta peso espectral en n.
    ω7min, ω7max, Nω7 = -3.0, 3.0, 801
    dω7 = (ω7max - ω7min) / (Nω7 - 1)
    η7  = 2 * dω7
    @printf("7. convergencia en N  (⟨σz⟩ DC;  Nω=%d  dω=%.3e  η=%.3e):\n", Nω7, dω7, η7)
    for N in (1, 2, 4, 8)
        pn = FloquetParams(; t=p.t, λ=p.λ, J_sd=p.J_sd, θ=p.θ, Ω=p.Ω, μ=p.μ, β=p.β, N=N)
        ρ  = rho_harmonic(pn; η = η7, ωmin = ω7min, ωmax = ω7max, Nω = Nω7)
        @printf("     N=%d  n=%.10f  ⟨σz⟩=%+.10f\n", N, occ(ρ), spin(ρ)[3])
    end
    println("="^72)
    return nothing
end

# ---------------------------------------------------------------- barrido
"""
Guarda dos CSV:
  floquet_ldos.csv  -- LDOS(ω) resuelta en espin (y densidad lesser cruda)
  floquet_rho_t.csv -- ρ(t) y ⟨σx,σy,σz⟩(t) sobre Nper periodos, en el sitio manejado
"""
function sweep(p::FloquetParams = FloquetParams();
               ωmin = -3.0, ωmax = 3.0, Nω = 6001, η = nothing,
               Nper = 3, Nt = 601)
    ωs = range(ωmin, ωmax; length = Nω)
    dω = step(ωs)
    ηe = isnothing(η) ? 2dω : η          # η = 2 * discretizacion, por defecto
    @printf("sweep: Nω=%d  dω=%.3e  η=%.3e  (η/Ω=%.2f)\n", Nω, dω, ηe, ηe/p.Ω)

    # --- 1. LDOS(ω) resuelta en espin -------------------------------------
    hdr1 = ["omega", "LDOS_up", "LDOS_dn", "LDOS_tot",
            "ReGless00_uu", "ReGless00_dd"]
    data1 = Matrix{Float64}(undef, Nω, length(hdr1))
    for (i, ω) in enumerate(ωs)
        A   = ldos(ω, p; η = ηe)
        Gl_ = fget(Gless_C(ω, p; η = ηe), 0, 0, p)
        data1[i, :] = [ω, real(A[1,1]), real(A[2,2]), real(A[1,1]) + real(A[2,2]),
                       real(-im*Gl_[1,1]), real(-im*Gl_[2,2])]
    end
    path1 = joinpath(OUT, "floquet_ldos.csv")
    writedlm(path1, vcat(permutedims(hdr1), data1), ",")
    println("  -> ", path1)

    # --- 2. ρ(t) y espin(t) sobre Nper periodos ---------------------------
    ts = spin_time_series(p; η = ηe, ωmin = ωmin, ωmax = ωmax, Nω = Nω,
                          Nper = Nper, Nt = Nt)
    hdr2 = ["t", "n_up", "n_dn", "Re_rho_updn", "Im_rho_updn", "sx", "sy", "sz", "n_tot"]
    data2 = Matrix{Float64}(undef, Nt, length(hdr2))
    for (i, ρt) in enumerate(ts.ρ)
        data2[i, :] = [ts.t[i], real(ρt[1,1]), real(ρt[2,2]),
                       real(ρt[1,2]), imag(ρt[1,2]),
                       ts.sx[i], ts.sy[i], ts.sz[i], ts.n[i]]
    end
    path2 = joinpath(OUT, "floquet_rho_t.csv")
    writedlm(path2, vcat(permutedims(hdr2), data2), ",")
    println("  -> ", path2)

    ρ0 = rho_harmonic(p; η = ηe, ωmin = ωmin, ωmax = ωmax, Nω = Nω, n = 0)
    ρ1 = rho_harmonic(p; η = ηe, ωmin = ωmin, ωmax = ωmax, Nω = Nω, n = 1)
    @printf("\n  ρ DC (k=0): n=%.8f   ⟨σx,σy,σz⟩ = (%+.3e, %+.3e, %+.3e)\n", occ(ρ0), spin(ρ0)...)
    @printf("  ρ k=1     :            ⟨σx,σy,σz⟩ = (%+.3e, %+.3e, %+.3e)\n", spin(ρ1)...)
    @printf("  ρ(t) instantanea en t=0:  n=%.8f  ⟨σx,σy,σz⟩ = (%+.3e, %+.3e, %+.3e)\n",
            ts.n[1], ts.sx[1], ts.sy[1], ts.sz[1])
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    p = FloquetParams()
    validate(p)
    println()
    sweep(p)
end
