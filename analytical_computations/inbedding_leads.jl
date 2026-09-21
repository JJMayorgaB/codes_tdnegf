#!/usr/bin/env julia

include(joinpath(@__DIR__, "floquet_gf.jl"))

#geometria
#T̂ = -t σ̂₀ - iλ σ̂_y = -γ exp(i θ_R σ̂_y)
T_hop(p::FloquetParams) = -p.t * σ0 - im * p.λ * σy
θ_R(p::FloquetParams)   = atan(p.λ, p.t)

function T_inout(lead::Symbol, p::FloquetParams)
    T = T_hop(p)
    return lead === :R ? (adjoint(T), T) : (T, adjoint(T))
end

T_depth(lead::Symbol, p::FloquetParams) =
    lead === :R ? (adjoint(T_hop(p)), T_hop(p)) : (T_hop(p), adjoint(T_hop(p)))

# funciones libres del lead (ǧ_αα)
g00_lead(ω::Real, p::FloquetParams; η::Real) = Σr(ω, p; η = η) / (2 * γ_band(p)^2)

#Parametro de la recursion: u = γ² G₀₀²
u_lead(ω::Real, p::FloquetParams; η::Real) = γ_band(p)^2 * g00_lead(ω, p; η = η)^2

"""
    g_lead_r(ω, n, m, lead, p; η): Componentes de la GF retardada del lead aislado
    G_{nn} = G₀₀ Σ_{p=0}^{n} uᵖ        
    G_{n0} = (T̂†)ⁿ G₀₀ⁿ⁺¹
    G_{0m} = T̂ᵐ  G₀₀ᵐ⁺¹
"""
function g_lead_r(ω::Real, n::Int, m::Int, lead::Symbol, p::FloquetParams; η::Real)
    a = g00_lead(ω, p; η = η)
    Tin_deep, Tout_deep = T_depth(lead, p)
    if n == m
        u   = u_lead(ω, p; η = η)
        acc = one(ComplexF64)
        s   = zero(ComplexF64)
        for _ in 0:n
            s += acc
            acc *= u
        end
        return (a * s) * σ0
    elseif m == 0
        return Tin_deep^n * (a^(n + 1) * σ0)
    elseif n == 0
        return Tout_deep^m * (a^(m + 1) * σ0)
    else
        error("g_lead_r: solo (n,n), (n,0) y (0,m); se pidio ($n,$m)")
    end
end

# gᵃ_{nm} = (gʳ_{mn})†
g_lead_a(ω::Real, n::Int, m::Int, lead::Symbol, p::FloquetParams; η::Real) =
    adjoint(g_lead_r(ω, m, n, lead, p; η = η))


#g<_{nm} = i f(ω) A_{nm},  A_{nm} = i(gʳ_{nm} - gᵃ_{nm}).
function g_lead_less(ω::Real, n::Int, m::Int, lead::Symbol, p::FloquetParams; η::Real)
    gr = g_lead_r(ω, n, m, lead, p; η = η)
    ga = g_lead_a(ω, n, m, lead, p; η = η)
    return im * fermi(ω, p) * (im * (gr - ga))
end

#Matriz 2N_F x 2N_F, diagonal en Floquet
function dress(ω::Real, n::Int, m::Int, lead::Symbol, p::FloquetParams;
               η::Real, comp::Symbol)
    nf = n_floquet(p)
    D  = zeros(ComplexF64, 2nf, 2nf)
    for k in -p.N:p.N
        r  = fblock(fidx(k, p))
        ωk = ω + k * p.Ω
        D[r, r] = comp === :r ? g_lead_r(ωk, n, m, lead, p; η = η)    :
                  comp === :a ? g_lead_a(ωk, n, m, lead, p; η = η)    :
                                g_lead_less(ωk, n, m, lead, p; η = η)
    end
    return D
end

#self-energia inbedding

struct LeadKernel
    Σr::Matrix{ComplexF64}
    Σa::Matrix{ComplexF64}
    Σl::Matrix{ComplexF64}
end

function lead_kernel(ω::Real, lead::Symbol, p::FloquetParams; η::Real)
    nf = n_floquet(p)
    Tin, Tout = T_inout(lead, p)
    Ti = kron(Matrix{ComplexF64}(I, nf, nf), Tin)
    To = kron(Matrix{ComplexF64}(I, nf, nf), Tout)
    Gr = Gr_C(ω, p; η = η)
    Gl = Gless_C(ω, p; η = η)
    return LeadKernel(Ti * Gr * To, Ti * adjoint(Gr) * To, Ti * Gl * To)
end

#GF del lead en el sitio n 
#Ǧʳ local del lead en el sitio n: ǧʳ_nn + ǧʳ_n0 Σ̌ʳ ǧʳ_0n."
function Gr_lead(ω::Real, n::Int, lead::Symbol, p::FloquetParams;
                 η::Real, K::Union{LeadKernel,Nothing} = nothing)
    Kr = isnothing(K) ? lead_kernel(ω, lead, p; η = η) : K
    return dress(ω, n, n, lead, p; η = η, comp = :r) +
           dress(ω, n, 0, lead, p; η = η, comp = :r) * Kr.Σr *
           dress(ω, 0, n, lead, p; η = η, comp = :r)
end


#Ǧ< local del lead en el sitio n: ǧ< + ǧ<Σ̌ᵃǧᵃ + ǧʳΣ̌<ǧᵃ + ǧʳΣ̌ʳǧ<
function Gless_lead(ω::Real, n::Int, lead::Symbol, p::FloquetParams;
                    η::Real, K::Union{LeadKernel,Nothing} = nothing)
    Kr = isnothing(K) ? lead_kernel(ω, lead, p; η = η) : K
    gl_r = dress(ω, n, 0, lead, p; η = η, comp = :r)
    gl_l = dress(ω, n, 0, lead, p; η = η, comp = :<)
    gr_a = dress(ω, 0, n, lead, p; η = η, comp = :a)
    gr_l = dress(ω, 0, n, lead, p; η = η, comp = :<)
    return dress(ω, n, n, lead, p; η = η, comp = :<) +
           gl_l * Kr.Σa * gr_a +
           gl_r * Kr.Σl * gr_a +
           gl_r * Kr.Σr * gr_l
end

# observables
#ldos_lead(ω, n, lead, p; η) -> Matrix{Float64} (2x2)

function ldos_lead(ω::Real, n::Int, lead::Symbol, p::FloquetParams;
                   η::Real, K::Union{LeadKernel,Nothing} = nothing)
    G = fget(Gr_lead(ω, n, lead, p; η = η, K = K), 0, 0, p)
    return @. -imag(G) / π
end

"""
rho_harmonic_lead(p; η, ωmin, ωmax, Nω, n, lead, k=0
Armonico k de la matriz densidad en el sitio n del lead,
ρₖ = -i ∫ dω/2π Ǧ<ₖ₀(ω)  (misma derivacion que en floquet_gf.jl: fila=k, col=0).
"""
function rho_harmonic_lead(p::FloquetParams; η::Real, ωmin::Real, ωmax::Real,
                           Nω::Int, n::Int, lead::Symbol, k::Int = 0)
    ωs  = range(ωmin, ωmax; length = Nω)
    dω  = step(ωs)
    acc = zeros(ComplexF64, 2, 2)
    for ω in ωs
        acc .+= fget(Gless_lead(ω, n, lead, p; η = η), k, 0, p)
    end
    return -im .* acc .* dω ./ (2π)
end

"""
depth_profile(p; η, ωmin, ωmax, Nω, sites, leads, Kmax)
Recorre ω por fuera y el sitio del lead por dentro, reusando el LeadKernel, y
acumula los armonicos ρₖ(n) de todos los sitios a la vez.
Devuelve Dict{(lead,n,k) => matriz 2x2}.
"""
function depth_profile(p::FloquetParams; η::Real, ωmin::Real, ωmax::Real, Nω::Int,
                       sites, leads = (:L, :R), Kmax::Int = min(2, p.N))
    ωs  = range(ωmin, ωmax; length = Nω)
    dω  = step(ωs)
    acc = Dict{Tuple{Symbol,Int,Int}, Matrix{ComplexF64}}()
    for α in leads, n in sites, k in -Kmax:Kmax
        acc[(α, n, k)] = zeros(ComplexF64, 2, 2)
    end
    for ω in ωs, α in leads
        K = lead_kernel(ω, α, p; η = η)          # una vez por (ω, lead)
        for n in sites
            Gl = Gless_lead(ω, n, α, p; η = η, K = K)
            for k in -Kmax:Kmax
                acc[(α, n, k)] .+= fget(Gl, k, 0, p)
            end
        end
    end
    return Dict(key => -im .* v .* dω ./ (2π) for (key, v) in acc)
end

# diagnosticos
"Longitud de decaimiento inducida por η: la correccion va como |u|ⁿ, ℓ = -1/ln|u|."
function decay_length(ω::Real, p::FloquetParams; η::Real)
    au = abs(u_lead(ω, p; η = η))
    return au >= 1 ? Inf : -1 / log(au)
end

spin_rot_per_site(p::FloquetParams) = 2 * θ_R(p)

"Periodo espacial de la espiral de Rashba, en sitios: 2π/(2θ_R) = π/θ_R."
spiral_period(p::FloquetParams) = θ_R(p) == 0 ? Inf : π / θ_R(p)

# validaciones
function validate_leads(p::FloquetParams = FloquetParams(); η::Real = 2e-3)
    ok(b) = b ? "OK" : "FALLA"
    γ = γ_band(p)
    println("="^72)
    @printf("INBEDDING LEADS   t=%.3f λ=%.3f J_sd=%.3f θ=%.1f° Ω=%.4f N=%d\n",
            p.t, p.λ, p.J_sd, rad2deg(p.θ), p.Ω, p.N)
    @printf("θ_R=%.4f rad (%.2f°)   el vector de espin rota %.2f°/sitio (=2θ_R)   periodo %.1f sitios\n",
            θ_R(p), rad2deg(θ_R(p)), rad2deg(spin_rot_per_site(p)), spiral_period(p))
    println("="^72)

    T = T_hop(p)

    # 1. amarre con floquet_gf.jl:  T̂† G₀₀ T̂ + T̂ G₀₀ T̂† = 2γ²G₀₀ = Σʳ
    e1 = 0.0
    for ω in range(-2.5, 2.5; length = 301)
        Semb = adjoint(T) * g_lead_r(ω, 0, 0, :L, p; η = η) * T +
               T * g_lead_r(ω, 0, 0, :R, p; η = η) * adjoint(T)
        e1 = max(e1, maximum(abs.(Semb - Σr(ω, p; η = η) * σ0)))
    end
    @printf("1. Σ_emb desde G₀₀ vs Σʳ de floquet_gf:  %.2e  [%s]\n", e1, ok(e1 < 1e-10))

    # 2. la recursion es consistente:  G_nn = G_{n-1,n-1} + G_{n-1,0} T̂† G_{0n}
    e2 = 0.0
    for α in (:L, :R), ω in (-1.7, -0.3, 0.44, 1.55), n in 1:8
        Tin, _ = T_depth(α, p)   # el T̂† del documento (T̂ para el lead izquierdo)
        lhs = g_lead_r(ω, n, n, α, p; η = η)
        rhs = g_lead_r(ω, n-1, n-1, α, p; η = η) +
              g_lead_r(ω, n-1, 0, α, p; η = η) * Tin * g_lead_r(ω, 0, n, α, p; η = η)
        e2 = max(e2, maximum(abs.(lhs - rhs)))
    end
    @printf("2. recursion G_nn = G_{n-1,n-1} + G_{n-1,0}T̂†G_{0n}:  %.2e  [%s]\n",
            e2, ok(e2 < 1e-10))

    # 3. la LDOS OSCILA alrededor del bulk, no converge a el.
    #    Dentro de la banda |u|~1, asi que la correccion G_n0 Σ G_0n no se
    #    atenua: oscila en el sitio con vector de onda 2k(ω) (tipo Friedel)
    #    modulada por la espiral de Rashba. Solo se apaga por η, en n ≳ ℓ.
    ωt = 0.53
    Abulk = 1 / (π * sqrt(4γ^2 - ωt^2))
    kω = acos(-ωt / (2γ))
    @printf("3. LDOS alrededor del bulk (ω=%.2f, bulk/espin=%.6f, periodo 2k: %.2f sitios):\n",
            ωt, Abulk, π / kω)
    for n in (0, 1, 4, 19, 79)
        A = ldos_lead(ωt, n, :R, p; η = η)
        @printf("     n=%3d :  A↑=%.6f  A↓=%.6f   A↑-bulk=%+.2e   |u|ⁿ=%.4f\n",
                n, A[1,1], A[2,2], A[1,1] - Abulk, abs(u_lead(ωt, p; η = η))^n)
    end
    println("   (NO debe converger: |u|≈1 en la banda, la correccion no se atenua)")

    # 4. positividad de la LDOS
    worst = Inf
    for α in (:L, :R), n in (0, 2, 9, 39), ω in range(-2.5, 2.5; length = 101)
        A = ldos_lead(ω, n, α, p; η = η)
        worst = min(worst, minimum(real(eigvals(Hermitian((A + A') / 2)))))
    end
    @printf("4. LDOS ⪰ 0 :  autovalor minimo = %.3e  [%s]\n", worst, ok(worst > -1e-10))

    # 5. soporte en el indice de Floquet (heredado de Ǧ_C)
    println("5. soporte |n-m| en Ǧ del lead (sitio 2, lead R):")
    Gr_ = Gr_lead(0.31, 2, :R, p; η = η)
    Gl_ = Gless_lead(0.31, 2, :R, p; η = η)
    for k in 0:min(3, 2p.N)
        a = maximum(maximum(abs.(fget(Gr_, n, n+k, p))) for n in -p.N:(p.N-k))
        b = maximum(maximum(abs.(fget(Gl_, n, n+k, p))) for n in -p.N:(p.N-k))
        @printf("     |n-m|=%d :  max|Ǧʳ|=%.3e   max|Ǧ<|=%.3e\n", k, a, b)
    end

    # 6. FDT en equilibrio: sin driving, Ǧ< = i f A
    pz = FloquetParams(; t=p.t, λ=p.λ, J_sd=p.J_sd, θ=0.0, Ω=p.Ω, μ=p.μ, β=p.β, N=p.N)
    e6 = 0.0
    for n in (0, 3, 14), ω in range(-1.8, 1.8; length = 121)
        Gr_ = fget(Gr_lead(ω, n, :R, pz; η = η), 0, 0, pz)
        Gl_ = fget(Gless_lead(ω, n, :R, pz; η = η), 0, 0, pz)
        A   = im * (Gr_ - adjoint(Gr_))
        e6  = max(e6, maximum(abs.(Gl_ - im * fermi(ω, pz) * A)))
    end
    @printf("6. FDT (θ=0, sin driving):  |Ǧ< - i f A| = %.2e  [%s]\n", e6, ok(e6 < 1e-10))

    # 7. longitud de decaimiento por η  -- NO es pass/fail, es una advertencia
    println("7. longitud de decaimiento inducida por η=$(η)  (ℓ = -1/ln|u|):")
    for ω in (0.0, 1.0, 1.9, 2.05)
        @printf("     ω=%+.2f :  |u|=%.8f   ℓ=%.1f sitios\n",
                ω, abs(u_lead(ω, p; η = η)), decay_length(ω, p; η = η))
    end
    println("   (no confiar en sitios n ≳ ℓ: ahi manda η, no la fisica)")
    println("="^72)
    return nothing
end

# barrido
"""
Los MISMOS observables de floquet_gf.jl, pero resueltos por sitio del lead.
Guarda dos CSV (formato largo, una fila por (ω,n,lead) o (t,n,lead)):
  inbedding_ldos.csv   -- LDOS(ω) resuelta en espin, para varios sitios
  inbedding_rho_t.csv  -- ρ(t), ⟨σ⟩(t) y ocupacion, para los mismos sitios
El sitio n=0 es la superficie (vecino del sitio manejado).
"""
function sweep_leads(p::FloquetParams = FloquetParams();
                     ωmin = -3.0, ωmax = 3.0, Nω = 6001, η = nothing,
                     sites = [0, 1, 4, 16],
                     leads = (:L, :R), Nper = 3, Nt = 601)
    ωs = range(ωmin, ωmax; length = Nω)
    dω = step(ωs)
    ηe = isnothing(η) ? 2dω : η
    @printf("sweep_leads: Nω=%d  dω=%.3e  η=%.3e   ℓ(ω=0)=%.1f sitios\n",
            Nω, dω, ηe, decay_length(0.0, p; η = ηe))

    # 1. LDOS(ω) por sitio 
    hdr1  = ["omega", "lead", "site", "LDOS_up", "LDOS_dn", "LDOS_tot"]
    data1 = Matrix{Any}(undef, Nω * length(leads) * length(sites), length(hdr1))
    i1 = 0
    for ω in ωs, α in leads
        K = lead_kernel(ω, α, p; η = ηe)      # una sola vez por (ω, lead)
        for n in sites
            A = ldos_lead(ω, n, α, p; η = ηe, K = K)
            i1 += 1
            data1[i1, :] = Any[ω, String(α), n, A[1,1], A[2,2], A[1,1] + A[2,2]]
        end
    end
    path1 = joinpath(OUT, "inbedding_ldos.csv")
    writedlm(path1, vcat(permutedims(hdr1), data1), ",")
    println("  -> ", path1)

    #2. ρ(t) y ⟨σ⟩(t): un solo barrido de ω para todos los armonicos 
    prof = depth_profile(p; η = ηe, ωmin = ωmin, ωmax = ωmax, Nω = Nω,
                         sites = sites, leads = leads)
    Kmax  = min(2, p.N)
    hdr2  = ["t", "lead", "site", "n_up", "n_dn", "Re_rho_updn", "Im_rho_updn",
             "sx", "sy", "sz", "n_tot"]
    tgrid = collect(range(0, Nper * 2π / p.Ω; length = Nt))
    data2 = Matrix{Any}(undef, length(leads) * length(sites) * Nt, length(hdr2))
    i2 = 0
    for α in leads, n in sites
        ρks = Dict(k => prof[(α, n, k)] for k in -Kmax:Kmax)
        for t in tgrid
            ρt = rho_of_t(t, ρks, p.Ω)
            sxv, syv, szv = spin(ρt)
            i2 += 1
            data2[i2, :] = Any[t, String(α), n, real(ρt[1,1]), real(ρt[2,2]),
                               real(ρt[1,2]), imag(ρt[1,2]), sxv, syv, szv, occ(ρt)]
        end
    end
    path2 = joinpath(OUT, "inbedding_rho_t.csv")
    writedlm(path2, vcat(permutedims(hdr2), data2), ",")
    println("  -> ", path2)

    println()
    for α in leads, n in sites
        ρ0 = prof[(α, n, 0)]
        @printf("  %s n=%2d :  ocup=%.6f   ⟨σx,σy,σz⟩ (prom. temporal) = (%+.3e, %+.3e, %+.3e)\n",
                α, n, occ(ρ0), spin(ρ0)...)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    p = FloquetParams()
    validate_leads(p)
    println()
    sweep_leads(p)
end
