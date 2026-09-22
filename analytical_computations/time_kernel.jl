#!/usr/bin/env julia
#=
time_kernel.jl -- kernel temporal de dos tiempos del sector electronico
inbebido en el lead,

  χ^{μν}_{ij}(t,t') = -i J_sd Tr_σ[ σ̂^μ ( G^<_{ij}(t,t') σ̂^ν G^a_{ji}(t',t)
                                        + G^r_{ij}(t,t') σ̂^ν G^<_{ji}(t',t) ) ]

para los pares (i,j) de sitios de UN lead, (t,t') ∈ [0, P·T]².

Metodologia:
  1. Ǧ_{ij}(ω) en espacio de Floquet: Ǧ_ij = g_ij + g_i0 Σ̌_inb g_0j, con la g_nm
     libre general (inbedding_leads.jl) y Σ̌_inb = T̂_in Ǧ_C T̂_out precomputada
     UNA vez por ω para todos los pares. Solo hace falta la columna de Floquet
     n=0: [ǧ Σ̌ ǧ]_{k0} = g(ω+kΩ) Σ̌_{k0} g(ω), porque ǧ es diagonal en Floquet.
  2. Armonicos   𝒢_{ij,k}(τ) = ∫dω/2π e^{-iωτ} Ǧ_{ij,k0}(ω),   |k| ≤ K.
  3. Reconstruccion   G_ij(t,t') = Σ_k e^{-ikΩt} 𝒢_{ij,k}(t-t').
  4. χ punto a punto en la malla (t_a, t'_b).

Que se usa y que NO se asume:
  - G^a_{ji}(t',t) = [G^r_{ij}(t,t')]†  (definicional: Ǧ^a = Ǧ^{r†}).
  - G^<_{ji}(t',t) se reconstruye DIRECTO de sus propios armonicos (el par (j,i)
    siempre se calcula). Si se usara el atajo -[G^<_{ij}(t,t')]† la parte
    imaginaria de χ saldria cero por construccion; asi Im χ es un test real.
    El atajo solo se compara y se reporta.
  - Causalidad: NO se impone. Se calcula la malla completa y se reporta
    max|χ| en t'>t frente a t'<t.
  - Truncacion de armonicos: se calcula hasta |k| = K (por defecto 4) con
    N = K + 2 replicas de Floquet, y se reporta el peso de cada |k| y la
    diferencia entre χ con |k|≤2 y con |k|≤K.

Cola en ω: cortar la integral en ±Λ deja un error ~ |Ǧ(Λ)|/(2πτ) en 𝒢(τ), que
incluye un falso 𝒢^r(τ<0). La parte libre del bloque k=0, g^r_ij(ω), decae
lento (~1/ω^{|i-j|+1}) y tiene transformada EXACTA conocida (cadena
semi-infinita, metodo de imagenes):
  g^r_ij(τ) = T̂^{|i-j|}_± (-γ)^{-|i-j|} (-i)θ(τ) e^{-ητ}
              [ i^{|i-j|} J_{|i-j|}(2γτ) - i^{i+j+2} J_{i+j+2}(2γτ) ]
Se resta g^r_ij(ω) antes de integrar y se suma g^r_ij(τ). Es una identidad
(se resta y se suma la misma funcion), no una hipotesis sobre G: lo que queda
en la integral numerica es g_i0 Σ̌ g_0j, que decae como ~1/ω^{i+j+3}.
Las J_ν de orden entero se calculan con la regla del trapecio periodica
(exponencialmente exacta), sin paquetes externos.
Por defecto ω ∈ (-15, 15) con malla uniforme fina (dω = η/2). Con --tail none
se integra Ǧ^r completa con corte limpio, para comparar.

Uso:
  julia -t 32 time_kernel.jl --sites 1,2,3,4 --lead R            # los 9 (μ,ν)
  julia -t 32 time_kernel.jl --sites 1,7,13,20 --mu x --nu y     # solo uno
  julia time_kernel.jl --help

Salida (en --outdir, por defecto output/time_kernel/):
  time_kernel_{μ}{ν}_{sitios}_i{I}_j{J}.npy   χ complejo (Nt,Nt), [a,b] = χ(t_a, t'_b)
  time_kernel_{sitios}_meta.json               malla, parametros, convenciones
  time_kernel_{sitios}_checks_{tag}.txt        todos los chequeos numericos
  harmonics_{hash}.npy/.txt                    cache de los armonicos 𝒢_{ij,k}(τ)
Convencion de sitios: la etiqueta i=1 es el sitio n=0 (interfaz con el sitio
manejado), como en el resto de figuras.
=#

include(joinpath(@__DIR__, "inbedding_leads.jl"))   # trae floquet_gf.jl

using LinearAlgebra
using Printf


struct M2
    a::ComplexF64   # (1,1)
    b::ComplexF64   # (2,1)
    c::ComplexF64   # (1,2)
    d::ComplexF64   # (2,2)
end
M2(A::AbstractMatrix) = M2(A[1, 1], A[2, 1], A[1, 2], A[2, 2])
const Z2 = M2(0, 0, 0, 0)

Base.:+(X::M2, Y::M2) = M2(X.a + Y.a, X.b + Y.b, X.c + Y.c, X.d + Y.d)
Base.:-(X::M2, Y::M2) = M2(X.a - Y.a, X.b - Y.b, X.c - Y.c, X.d - Y.d)
Base.:-(X::M2)        = M2(-X.a, -X.b, -X.c, -X.d)
Base.:*(s::Number, X::M2) = M2(s * X.a, s * X.b, s * X.c, s * X.d)
Base.:*(X::M2, Y::M2) = M2(X.a * Y.a + X.c * Y.b, X.b * Y.a + X.d * Y.b,
                           X.a * Y.c + X.c * Y.d, X.b * Y.c + X.d * Y.d)
Base.adjoint(X::M2)   = M2(conj(X.a), conj(X.c), conj(X.b), conj(X.d))
tr2(X::M2)    = X.a + X.d
maxabs(X::M2) = max(abs(X.a), abs(X.b), abs(X.c), abs(X.d))
Base.Matrix(X::M2) = ComplexF64[X.a X.c; X.b X.d]

const PAULI = Dict("x" => M2(σx), "y" => M2(σy), "z" => M2(σz))
const P0    = M2(σ0)


# GF libre del lead en forma rapida: potencias de T̂ precalculadas
"Tin^d y Tout^d (d = 0..dmax) del lead, con la MISMA convencion que T_depth."
function tpowers(lead::Symbol, p::FloquetParams, dmax::Int)
    Tin, Tout = T_depth(lead, p)
    return (Tin  = [M2(Tin^d)  for d in 0:dmax],
            Tout = [M2(Tout^d) for d in 0:dmax])
end

"""
    besselj_int(ν, x): J_ν(x) de orden entero,
        J_ν(x) = (1/2π) ∫₀^{2π} cos(νθ - x sinθ) dθ
    con la regla del trapecio en un periodo: el integrando es periodico y
    analitico, asi que el error es exponencialmente chico en cuanto M supera
    |x| + |ν| con un margen de unas x^{1/3} (zona de transicion de J_ν).
"""
function besselj_int(ν::Int, x::Float64)
    M = ceil(Int, abs(x) + abs(ν) + 12 * abs(x)^(1 / 3) + 32)
    s = 0.0
    @inbounds for q in 0:M-1
        θ = 2π * q / M
        s += cos(ν * θ - x * sin(θ))
    end
    return s / M
end

"i^n exacto como ComplexF64."
ipow(n::Int) = (1.0 + 0.0im, 0.0 + 1.0im, -1.0 + 0.0im, 0.0 - 1.0im)[mod(n, 4) + 1]

"""
Transformada exacta de la parte escalar de la g^r libre del lead,
    s_nm(τ) = ∫dω/2π e^{-iωτ} g₀₀^{|d|+1} Σ_{p≤min(n,m)} uᵖ,   d = n - m,
    s_nm(τ) = (-γ)^{-|d|} (-i)θ(τ) e^{-ητ} [ i^{|d|} J_{|d|}(2γτ) - i^{n+m+2} J_{n+m+2}(2γτ) ]
(g_nm(τ) = T̂^{|d|}_± s_nm(τ)). θ(0) = 1/2, igual que la inversion de Fourier.
"""
function free_ft_scalar(n::Int, m::Int, τ::Float64, γ::Float64, η::Float64)
    τ < 0 && return zero(ComplexF64)
    d  = abs(n - m)
    th = τ == 0 ? 0.5 : 1.0
    x  = 2γ * τ
    br = ipow(d) * besselj_int(d, x) - ipow(n + m + 2) * besselj_int(n + m + 2, x)
    return -im * th * exp(-η * τ) * br / (-γ)^d
end

"g^r_nm libre a partir de g₀₀ = a y u = γ²a² (mismo resultado que g_lead_r)."
@inline function gr_free(a::ComplexF64, u::ComplexF64, n::Int, m::Int, TP)
    s   = zero(ComplexF64)
    acc = one(ComplexF64)
    for _ in 0:min(n, m)
        s   += acc
        acc *= u
    end
    d = n - m
    return d >= 0 ? (a^(d + 1) * s) * TP.Tin[d + 1] : (a^(1 - d) * s) * TP.Tout[1 - d]
end


# Ǧ_ij de matriz completa (solo para validar la version rapida)
function Gr_lead_ij(ω::Real, i::Int, j::Int, lead::Symbol, p::FloquetParams;
                    η::Real, K::Union{LeadKernel,Nothing} = nothing)
    Kr = isnothing(K) ? lead_kernel(ω, lead, p; η = η) : K
    return dress(ω, i, j, lead, p; η = η, comp = :r) +
           dress(ω, i, 0, lead, p; η = η, comp = :r) * Kr.Σr *
           dress(ω, 0, j, lead, p; η = η, comp = :r)
end

function Gless_lead_ij(ω::Real, i::Int, j::Int, lead::Symbol, p::FloquetParams;
                       η::Real, K::Union{LeadKernel,Nothing} = nothing)
    Kr = isnothing(K) ? lead_kernel(ω, lead, p; η = η) : K
    gl_r = dress(ω, i, 0, lead, p; η = η, comp = :r)
    gl_l = dress(ω, i, 0, lead, p; η = η, comp = :<)
    gr_a = dress(ω, 0, j, lead, p; η = η, comp = :a)
    gr_l = dress(ω, 0, j, lead, p; η = η, comp = :<)
    return dress(ω, i, j, lead, p; η = η, comp = :<) +
           gl_l * Kr.Σa * gr_a + gl_r * Kr.Σl * gr_a + gl_r * Kr.Σr * gr_l
end

# Columna de armonicos a un ω: bloques Ǧ^r_{ij,k0}, Ǧ^<_{ij,k0}, |k| ≤ K
@inline colbase(kk, c, hp, nk) = 4 * ((kk - 1) + nk * ((c - 1) + 2 * (hp - 1)))

@inline function putblock!(col, fm, base, X::M2, Xs::M2)
    # X  : bloque fisico (para el maximo en ω)
    # Xs : lo que entra a la integral (X menos la cola analitica, si aplica)
    col[base + 1] = Xs.a; col[base + 2] = Xs.b
    col[base + 3] = Xs.c; col[base + 4] = Xs.d
    fm[base + 1] = max(fm[base + 1], abs(X.a)); fm[base + 2] = max(fm[base + 2], abs(X.b))
    fm[base + 3] = max(fm[base + 3], abs(X.c)); fm[base + 4] = max(fm[base + 4], abs(X.d))
    return nothing
end

function fill_column!(col, fm, ω::Float64, p::FloquetParams, hpairs, sites, sidx,
                      K::Int, TP, Tio; η::Float64, tail::Bool = true)
    nk = 2K + 1
    Ω  = p.Ω
    γ2 = γ_band(p)^2
    Ti, To = Tio

    # sitio manejado: Ǧ^r_C, Ǧ^a_C = Ǧ^{r†}_C, Ǧ^<_C = Ǧ^r Σ̌^< Ǧ^a
    G  = Gr_C(ω, p; η = η)
    Gl = G * Σless_floquet(ω, p; η = η) * adjoint(G)

    # columna 0 de Σ̌_inb = T̂_in Ǧ_C T̂_out (T̂ diagonal en Floquet)
    Sr = Vector{M2}(undef, nk); Sa = Vector{M2}(undef, nk); Sl = Vector{M2}(undef, nk)
    for kk in 1:nk
        k = kk - 1 - K
        Sr[kk] = Ti * M2(fget(G, k, 0, p)) * To
        Sa[kk] = Ti * adjoint(M2(fget(G, 0, k, p))) * To   # [Ǧ^{r†}]_{k0} = (Ǧ^r_{0k})†
        Sl[kk] = Ti * M2(fget(Gl, k, 0, p)) * To
    end

    # lead libre en las frecuencias corridas ω+kΩ
    ak = [g00_lead(ω + (kk - 1 - K) * Ω, p; η = η) for kk in 1:nk]
    uk = [γ2 * a^2 for a in ak]
    fk = [fermi(ω + (kk - 1 - K) * Ω, p) for kk in 1:nk]
    k0 = K + 1
    a0, u0, f0 = ak[k0], uk[k0], fk[k0]

    ns = length(sites)
    gr_s0 = Matrix{M2}(undef, nk, ns)     # g^r_{s0}(ω+kΩ)
    gl_s0 = Matrix{M2}(undef, nk, ns)     # g^<_{s0}(ω+kΩ)
    gr_0s = Vector{M2}(undef, ns)         # g^r_{0s}(ω)
    ga_0s = Vector{M2}(undef, ns)         # g^a_{0s}(ω) = (g^r_{s0}(ω))†
    gl_0s = Vector{M2}(undef, ns)         # g^<_{0s}(ω)
    for (is, n) in enumerate(sites)
        for kk in 1:nk
            grn0 = gr_free(ak[kk], uk[kk], n, 0, TP)
            gan0 = adjoint(gr_free(ak[kk], uk[kk], 0, n, TP))
            gr_s0[kk, is] = grn0
            gl_s0[kk, is] = (-fk[kk]) * (grn0 - gan0)     # g^< = i f · i(g^r - g^a)
        end
        gr_0s[is] = gr_free(a0, u0, 0, n, TP)
        ga_0s[is] = adjoint(gr_s0[k0, is])
        gl_0s[is] = (-f0) * (gr_0s[is] - ga_0s[is])
    end

    for (hp, (ni, nj)) in enumerate(hpairs)
        ii, jj = sidx[ni], sidx[nj]
        gr_ij = gr_free(a0, u0, ni, nj, TP)
        ga_ij = adjoint(gr_free(a0, u0, nj, ni, TP))
        gl_ij = (-f0) * (gr_ij - ga_ij)
        for kk in 1:nk
            Xr = gr_s0[kk, ii] * Sr[kk] * gr_0s[jj]
            Xl = gl_s0[kk, ii] * Sa[kk] * ga_0s[jj] +
                 gr_s0[kk, ii] * Sl[kk] * ga_0s[jj] +
                 gr_s0[kk, ii] * Sr[kk] * gl_0s[jj]
            if kk == k0
                Xr = Xr + gr_ij
                Xl = Xl + gl_ij
            end
            # a la integral numerica entra Ǧ^r - g^r_ij (k=0); g^r_ij(τ) se suma exacta
            Xrs = (tail && kk == k0) ? Xr - gr_ij : Xr
            putblock!(col, fm, colbase(kk, 1, hp, nk), Xr, Xrs)
            putblock!(col, fm, colbase(kk, 2, hp, nk), Xl, Xl)
        end
    end
    return nothing
end

"Lee de una columna el bloque (hp, c, kk) como M2."
getblock(col, kk, c, hp, nk) = (b = colbase(kk, c, hp, nk);
                                M2(col[b + 1], col[b + 2], col[b + 3], col[b + 4]))

# Armonicos 𝒢_{ij,k}(τ): cuadratura en ω como producto de matrices por bloques
function build_harmonics(p, hpairs, sites, sidx, K, TP, Tio, ωs, wts, τs;
                         η::Float64, chunk::Int, lg, tail::Bool = true)
    nk = 2K + 1
    nh = length(hpairs)
    M  = 4 * nk * 2 * nh
    Nτ = length(τs)
    Nω = length(ωs)
    H    = zeros(ComplexF64, Nτ, M)
    fmax = zeros(Float64, M)
    nthr  = Threads.nthreads()
    nblas = BLAS.get_num_threads()
    t0 = time()
    nchunks = cld(Nω, chunk)
    for (ic, c0) in enumerate(1:chunk:Nω)
        ws = c0:min(c0 + chunk - 1, Nω)
        nc = length(ws)
        Vt = Matrix{ComplexF64}(undef, M, nc)      # una columna por ω
        P  = Matrix{ComplexF64}(undef, Nτ, nc)     # w(ω)/2π · e^{-iωτ}
        parts = collect(Iterators.partition(1:nc, cld(nc, nthr)))
        fms   = [zeros(Float64, M) for _ in parts]
        BLAS.set_num_threads(1)                    # los hilos de Julia hacen las inv
        @sync for (ip, part) in enumerate(parts)
            Threads.@spawn begin
                for l in part
                    w = ws[l]
                    ω = ωs[w]
                    fill_column!(view(Vt, :, l), fms[ip], ω, p, hpairs, sites, sidx,
                                 K, TP, Tio; η = η, tail = tail)
                    cw = wts[w] / (2π)
                    @inbounds for it in 1:Nτ
                        P[it, l] = cw * cis(-ω * τs[it])
                    end
                end
            end
        end
        for f in fms
            fmax .= max.(fmax, f)
        end
        BLAS.set_num_threads(nblas)
        mul!(H, P, transpose(Vt), true, true)      # H += P Vᵀ
        if ic == 1 || ic % max(1, nchunks ÷ 10) == 0 || ic == nchunks
            el = time() - t0
            lg(@sprintf("   bloque %d/%d de ω  (%.1f s, faltan ~%.1f s)",
                        ic, nchunks, el, el / ic * (nchunks - ic)))
        end
    end
    # se suma la transformada EXACTA de la parte libre g^r_ij restada en k=0
    tail && add_free_exact!(H, p, hpairs, K, TP, τs; η = η)
    return H, fmax
end

"H[:, bloque (r, k=0, par)] += g^r_ij(τ) = T̂^{|d|}_± s_ij(τ)."
function add_free_exact!(H, p, hpairs, K, TP, τs; η::Float64)
    nk = 2K + 1
    k0 = K + 1
    γ  = γ_band(p)
    Threads.@threads for hp in eachindex(hpairs)
        ni, nj = hpairs[hp]
        d  = ni - nj
        Td = d >= 0 ? TP.Tin[d + 1] : TP.Tout[1 - d]
        b  = colbase(k0, 1, hp, nk)
        for (it, τ) in enumerate(τs)
            s = free_ft_scalar(ni, nj, τ, γ, η)
            s == 0 && continue
            H[it, b + 1] += s * Td.a
            H[it, b + 2] += s * Td.b
            H[it, b + 3] += s * Td.c
            H[it, b + 4] += s * Td.d
        end
    end
    return H
end


# E/S: .npy sin dependencias externas (numpy lo lee con np.load)
function write_npy(path::AbstractString, A::Array{T}) where {T}
    descr = T === ComplexF64 ? "<c16" : T === Float64 ? "<f8" :
            error("write_npy: tipo no soportado $T")
    shp = size(A)
    shs = length(shp) == 1 ? "($(shp[1]),)" : "(" * join(shp, ", ") * ")"
    hdr = "{'descr': '$descr', 'fortran_order': True, 'shape': $shs, }"
    pad = mod(64 - mod(10 + length(hdr) + 1, 64), 64)
    hdr = hdr * " "^pad * "\n"
    open(path, "w") do io
        write(io, UInt8(0x93))
        write(io, "NUMPY")
        write(io, UInt8(1))
        write(io, UInt8(0))
        write(io, htol(UInt16(length(hdr))))
        write(io, hdr)
        write(io, A)                     # orden de columnas = fortran_order
    end
    return path
end

function read_npy_c16(path::AbstractString)
    open(path, "r") do io
        magic = read(io, 6)
        magic == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
            error("no es un .npy: $path")
        major = read(io, UInt8)
        read(io, UInt8)
        hlen = major == 1 ? Int(ltoh(read(io, UInt16))) : Int(ltoh(read(io, UInt32)))
        hdr  = String(read(io, hlen))
        (occursin("'<c16'", hdr) && occursin("'fortran_order': True", hdr)) ||
            error("formato inesperado en $path: $hdr")
        m   = match(r"'shape':\s*\(([^)]*)\)", hdr)
        shp = Tuple(parse.(Int, filter(!isempty, strip.(split(m[1], ",")))))
        A   = Array{ComplexF64}(undef, shp...)
        read!(io, A)
        return A
    end
end

jsonval(v::Bool) = v ? "true" : "false"
jsonval(v::Integer) = string(v)
jsonval(v::AbstractFloat) = isfinite(v) ? @sprintf("%.17g", v) : "null"
jsonval(v::AbstractString) = "\"" * replace(v, "\\" => "\\\\", "\"" => "\\\"") * "\""
jsonval(v::Symbol) = jsonval(String(v))
jsonval(v::AbstractVector) = "[" * join(jsonval.(v), ", ") * "]"
jsonval(v::Tuple) = jsonval(collect(v))

function write_json(path::AbstractString, d::Vector{<:Pair})
    open(path, "w") do io
        println(io, "{")
        for (q, (k, v)) in enumerate(d)
            print(io, "  ", jsonval(String(k)), ": ", jsonval(v))
            println(io, q < length(d) ? "," : "")
        end
        println(io, "}")
    end
    return path
end


# Validaciones de las piezas nuevas
function validations(p, lead, hpairs, sites, sidx, K, TP, Tio; η, lg)
    ok(b) = b ? "OK" : "FALLA"
    nk = 2K + 1
    Tin, _ = T_depth(lead, p)
    ωt = (-1.37, -0.21, 0.013, 0.58, 1.66, 2.31)
    nmax = max(maximum(sites), 12)

    lg("-"^78)
    lg("VALIDACIONES")

    # 1. g_nm general reproduce las formulas especiales que ya estaban validadas
    e1 = 0.0
    for ω in ωt, n in 0:nmax
        a  = g00_lead(ω, p; η = η); u = u_lead(ω, p; η = η)
        Ti_, To_ = T_depth(lead, p)
        e1 = max(e1, maximum(abs.(g_lead_r(ω, n, n, lead, p; η = η) -
                                  a * sum(u^q for q in 0:n) * σ0)))
        e1 = max(e1, maximum(abs.(g_lead_r(ω, n, 0, lead, p; η = η) - Ti_^n * (a^(n + 1) * σ0))))
        e1 = max(e1, maximum(abs.(g_lead_r(ω, 0, n, lead, p; η = η) - To_^n * (a^(n + 1) * σ0))))
    end
    lg(@sprintf("1. g_nm general vs casos (n,n),(n,0),(0,m):            %.2e  [%s]", e1, ok(e1 < 1e-10)))

    # 2. recursion de auto-similaridad g_nm = g_{n-1,m-1} + g_{n-1,0} T̂† g_{0m}
    e2 = 0.0; s2 = 0.0
    for ω in ωt, n in 1:nmax, m in 1:nmax
        lhs = g_lead_r(ω, n, m, lead, p; η = η)
        rhs = g_lead_r(ω, n - 1, m - 1, lead, p; η = η) +
              g_lead_r(ω, n - 1, 0, lead, p; η = η) * Tin * g_lead_r(ω, 0, m, lead, p; η = η)
        e2 = max(e2, maximum(abs.(lhs - rhs))); s2 = max(s2, maximum(abs.(lhs)))
    end
    lg(@sprintf("2. recursion g_nm = g_{n-1,m-1} + g_{n-1,0}T̂†g_{0m}:     %.2e  (rel %.2e) [%s]",
                e2, e2 / s2, ok(e2 / s2 < 1e-10)))

    # 3. version rapida (M2) de g_nm vs matricial
    e3 = 0.0
    for ω in ωt, n in 0:nmax, m in 0:nmax
        a = g00_lead(ω, p; η = η); u = u_lead(ω, p; η = η)
        e3 = max(e3, maximum(abs.(Matrix(gr_free(a, u, n, m, TP)) - g_lead_r(ω, n, m, lead, p; η = η))))
    end
    lg(@sprintf("3. g_nm rapida (2x2) vs g_lead_r:                         %.2e  [%s]", e3, ok(e3 < 1e-12)))

    # 4. columna 0 rapida vs Ǧ_ij de matriz completa (dress + lead_kernel)
    M = 4 * nk * 2 * length(hpairs)
    col = zeros(ComplexF64, M); fm = zeros(M)
    e4r = 0.0; e4l = 0.0; s4 = 0.0
    for ω in ωt
        fill_column!(col, fm, ω, p, hpairs, sites, sidx, K, TP, Tio; η = η, tail = false)
        Kl = lead_kernel(ω, lead, p; η = η)
        for (hp, (ni, nj)) in enumerate(hpairs)
            Gf = Gr_lead_ij(ω, ni, nj, lead, p; η = η, K = Kl)
            Lf = Gless_lead_ij(ω, ni, nj, lead, p; η = η, K = Kl)
            for kk in 1:nk
                k = kk - 1 - K
                e4r = max(e4r, maximum(abs.(Matrix(getblock(col, kk, 1, hp, nk)) - fget(Gf, k, 0, p))))
                e4l = max(e4l, maximum(abs.(Matrix(getblock(col, kk, 2, hp, nk)) - fget(Lf, k, 0, p))))
                s4  = max(s4, maximum(abs.(fget(Gf, k, 0, p))))
            end
        end
    end
    lg(@sprintf("4. columna 0 rapida vs matriz completa:  r %.2e   < %.2e  (escala %.2e) [%s]",
                e4r, e4l, s4, ok(max(e4r, e4l) < 1e-10 * max(1.0, s4))))

    # 5. antihermiticidad de Ǧ^< en espacio de Floquet: (Ǧ^<_ij)† = -Ǧ^<_ji
    e5 = 0.0; s5 = 0.0
    for ω in ωt
        Kl = lead_kernel(ω, lead, p; η = η)
        for (ni, nj) in hpairs
            A = Gless_lead_ij(ω, ni, nj, lead, p; η = η, K = Kl)
            B = Gless_lead_ij(ω, nj, ni, lead, p; η = η, K = Kl)
            e5 = max(e5, maximum(abs.(adjoint(A) + B))); s5 = max(s5, maximum(abs.(A)))
        end
    end
    lg(@sprintf("5. (Ǧ^<_ij)† + Ǧ^<_ji = 0 (matriz de Floquet completa):  %.2e  (rel %.2e) [%s]",
                e5, e5 / s5, ok(e5 / s5 < 1e-10)))

    # 6. J_ν por trapecio periodico vs valores de referencia (scipy.special.jv)
    refs = ((0, 1.0, 0.7651976865579666), (1, 1.0, 0.44005058574493355),
            (5, 10.0, -0.2340615281867936), (20, 100.0, 0.06221745849833951),
            (42, 2000.0, 0.0005678797465551087), (3, 7539.8, 0.006637499320661047))
    e6 = maximum(abs(besselj_int(ν, x) - v) for (ν, x, v) in refs)
    lg(@sprintf("6. J_ν(x) por trapecio periodico vs referencia:           %.2e  [%s]", e6, ok(e6 < 1e-12)))

    # 7. transformada exacta de la g libre vs cuadratura numerica con banda muy
    #    ancha (±400). Para d=0 la cola 1/ω se trata aparte con σ̂₀/(ω+i), cuya
    #    transformada es -iθ(τ)e^{-τ}; el error que queda es ~1/(2π·400·τ).
    ηv = 1e-3; W = 400.0
    Nv = ceil(Int, 2W / (ηv / 4)) + 1
    ωv = collect(range(-W, W; length = Nv)); dωv = ωv[2] - ωv[1]
    av = [g00_lead(ω, p; η = ηv) for ω in ωv]
    uv = γ_band(p)^2 .* av .^ 2
    γ  = γ_band(p)
    e7 = 0.0; s7 = 0.0
    for (n, m) in ((0, 0), (2, 0), (0, 3), (3, 1), (2, 2)), τ in (-30.0, -5.0, 0.7, 3.3, 12.0, 40.0)
        d  = n - m
        acc = zero(ComplexF64)
        for q in 1:Nv
            S = zero(ComplexF64); pw = one(ComplexF64)
            for _ in 0:min(n, m)
                S += pw; pw *= uv[q]
            end
            f = av[q]^(abs(d) + 1) * S - (d == 0 ? 1 / complex(ωv[q], 1.0) : 0.0)
            wq = (q == 1 || q == Nv) ? dωv / 2 : dωv
            acc += wq * f * cis(-ωv[q] * τ)
        end
        num = acc / (2π)
        d == 0 && τ >= 0 && (num += -im * (τ == 0 ? 0.5 : 1.0) * exp(-τ))
        ex = free_ft_scalar(n, m, τ, γ, ηv)
        e7 = max(e7, abs(num - ex)); s7 = max(s7, abs(ex))
    end
    lg(@sprintf("7. g^r libre: FT exacta (Bessel) vs cuadratura ±400:     %.2e  (max|g| %.2e) [%s]",
                e7, s7, ok(e7 < 1e-4 * s7)))
    lg("-"^78)
    return nothing
end


const USAGE = """
julia -t NHILOS time_kernel.jl [opciones]

  --sites 1,2,3,4      etiquetas de sitio (i=1 es la interfaz n=0)
  --lead R|L           lead (R por defecto)
  --mu x --nu y        calcula solo ese (μ,ν); sin ellos, los 9
  --pairs 1:1,2:1      subconjunto de pares i:j (por defecto todos los de --sites)
  --periods 3          ventana (t,t') ∈ [0, P·T]²
  --Nt 601             puntos por eje de tiempo
  --kmax 4             armonico maximo |k|; se usa N = kmax + 2
  --eta 2.65e-6        ensanchamiento (defecto: el de la corrida validada η6b;
                       auto = 0.01/(P·T))
  --Nomega auto        puntos en ω; auto = dω = η/2
  --omega-min -15 --omega-max 15   malla uniforme fina (dω = η/2) en todo el intervalo
  --tail free|none     free: resta g^r_ij(ω) y suma su FT exacta (defecto)
                       none: corte limpio de la integral, para comparar
  --chunk 4096         frecuencias por bloque del producto de matrices
  --outdir DIR         por defecto output/time_kernel
  --no-cache           no leer/escribir la cache de armonicos
  --lambda --Jsd --theta(grados) --Omega --EF --beta   parametros fisicos
"""

const VALKEYS = Set(["sites", "lead", "mu", "nu", "pairs", "periods", "Nt", "kmax", "eta",
                     "Nomega", "omega-min", "omega-max", "tail", "chunk", "outdir",
                     "lambda", "Jsd", "theta", "Omega", "EF", "beta"])
const FLAGKEYS = Set(["no-cache", "help"])

function parse_cli(argv)
    o = Dict{String,String}()
    q = 1
    while q <= length(argv)
        a = argv[q]
        startswith(a, "--") || error("argumento inesperado: $a\n$USAGE")
        key = a[3:end]
        if key in FLAGKEYS
            o[key] = "true"; q += 1
        elseif key in VALKEYS
            q < length(argv) || error("falta el valor de --$key")
            o[key] = argv[q + 1]; q += 2
        else
            error("opcion desconocida --$key\n$USAGE")
        end
    end
    return o
end

# main
function main(argv = ARGS)
    o = parse_cli(argv)
    if haskey(o, "help")
        print(USAGE)
        return nothing
    end
    getf(k, d) = parse(Float64, get(o, k, string(d)))
    geti(k, d) = parse(Int, get(o, k, string(d)))

    # sitios, pares, (μ,ν)
    labels = parse.(Int, split(get(o, "sites", "1,2,3,4"), ","))
    all(>=(1), labels) || error("--sites: las etiquetas empiezan en 1 (i=1 es n=0)")
    tagS = join(labels, "-")
    lead = Symbol(get(o, "lead", "R"))
    lead in (:L, :R) || error("--lead debe ser L o R")

    req = if haskey(o, "pairs")
        [Tuple(parse.(Int, split(s, ":"))) for s in split(o["pairs"], ",")]
    else
        [(I, J) for J in labels for I in labels]
    end
    hl = unique(vcat(req, [(J, I) for (I, J) in req]))      # siempre (i,j) y (j,i)
    hpairs = [(I - 1, J - 1) for (I, J) in hl]               # indices internos n
    hidx   = Dict(hl[q] => q for q in eachindex(hl))
    sites  = sort(unique(vcat(first.(hpairs), last.(hpairs))))
    sidx   = Dict(n => q for (q, n) in enumerate(sites))

    comps = ["x", "y", "z"]
    mns = if haskey(o, "mu") || haskey(o, "nu")
        (haskey(o, "mu") && haskey(o, "nu")) || error("--mu y --nu van juntos")
        (o["mu"] in comps && o["nu"] in comps) || error("--mu/--nu deben ser x, y o z")
        [(o["mu"], o["nu"])]
    else
        [(μ, ν) for μ in comps for ν in comps]
    end
    tagMN = length(mns) == 1 ? mns[1][1] * mns[1][2] : "all"

    # parametros
    K = geti("kmax", 4)
    K >= 2 || error("--kmax debe ser ≥ 2 (se compara contra |k| ≤ 2)")
    p0 = FloquetParams()
    λ  = getf("lambda", p0.λ)
    p  = FloquetParams(; λ = λ, t = sqrt(1 - λ^2), J_sd = getf("Jsd", p0.J_sd),
                       θ = deg2rad(getf("theta", rad2deg(p0.θ))), Ω = getf("Omega", p0.Ω),
                       μ = getf("EF", p0.μ), β = getf("beta", p0.β), N = K + 2)

    T       = 2π / p.Ω
    periods = getf("periods", 3.0)
    Nt      = geti("Nt", 601)
    τmax    = periods * T
    dt      = τmax / (Nt - 1)
    ts      = [(a - 1) * dt for a in 1:Nt]
    Nτ      = 2Nt - 1
    τs      = [(it - Nt) * dt for it in 1:Nτ]           # τ = t_a - t'_b, it = a - b + Nt

    # defecto = corrida validada "η6b": η = 2.65e-6, dω = η/2 (Nω auto), ω ∈ ±15.
    # η·τmax = 0.01 (≈2 % de amortiguamiento en χ a τ = 3T) y aliasing ~e^{-4π}.
    ηarg = get(o, "eta", "2.65e-6")
    η  = ηarg == "auto" ? 0.01 / τmax : parse(Float64, ηarg)
    ωmin = getf("omega-min", -15.0); ωmax = getf("omega-max", 15.0)
    tailmode = get(o, "tail", "free")
    tailmode in ("free", "none") || error("--tail debe ser free o none")
    usetail = tailmode == "free"
    Nω = get(o, "Nomega", "auto") == "auto" ? ceil(Int, (ωmax - ωmin) / (η / 2)) + 1 :
         parse(Int, o["Nomega"])
    ωs  = collect(range(ωmin, ωmax; length = Nω))
    dω  = ωs[2] - ωs[1]
    wts = fill(dω, Nω); wts[1] = wts[end] = dω / 2           # trapecio
    chunk = geti("chunk", 4096)
    outdir = get(o, "outdir", joinpath(@__DIR__, "output", "time_kernel"))
    mkpath(outdir)
    usecache = !haskey(o, "no-cache")

    logbuf = String[]
    lg(s) = (println(s); push!(logbuf, s); flush(stdout))

    lg("="^78)
    lg("TIME KERNEL  χ^{μν}_{ij}(t,t')")
    lg("="^78)
    lg(@sprintf("t=%.4f λ=%.4f J_sd=%.3f θ=%.2f° Ω=%.4f E_F=%.3f β=%.1f", p.t, p.λ, p.J_sd,
                rad2deg(p.θ), p.Ω, p.μ, p.β))
    lg(@sprintf("lead %s   sitios %s  (internos n = %s)", lead, string(labels), string(labels .- 1)))
    lg(@sprintf("pares pedidos %d, con transpuestos %d   (μ,ν): %s", length(req), length(hl),
                join([μ * ν for (μ, ν) in mns], " ")))
    lg(@sprintf("K = %d  ->  N = %d replicas de Floquet (%dx%d)", K, p.N, 2 * (2p.N + 1), 2 * (2p.N + 1)))
    lg(@sprintf("T = %.3f   ventana [0, %.1f T] = [0, %.1f]   Nt = %d   dt = %.4f", T, periods, τmax, Nt, dt))
    lg(@sprintf("ω ∈ [%.2f, %.2f]  Nω = %d  dω = %.3e   η = %.3e   cola: %s", ωmin, ωmax, Nω, dω, η,
                usetail ? "resta exacta de g^r_ij" : "corte limpio"))
    lg(@sprintf("hilos Julia = %d   hilos BLAS = %d", Threads.nthreads(), BLAS.get_num_threads()))
    fη = η * τmax
    lg(@sprintf("η·τmax = %.3f  ->  amortiguamiento artificial e^{-ητ} al final de la ventana = %.4f",
                fη, exp(-fη)))
    fη > 0.1 && lg("   AVISO: η·τmax > 0.1, los tiempos largos quedan afectados por η")
    dω > η / 2 * (1 + 1e-9) &&
        lg("   AVISO: dω > η/2, los lorentzianos de ancho η quedan submuestreados")
    dω * τmax > π && lg("   AVISO: dω·τmax > π, e^{-iωτ} submuestreada en ω a τ grandes")

    # potencias hasta max(sitio, 12): las validaciones 1-3 recorren n,m ≤ 12
    TP  = tpowers(lead, p, max(maximum(sites), 12))
    Tio = Tuple(M2.(T_inout(lead, p)))

    validations(p, lead, hpairs, sites, sidx, K, TP, Tio; η = η, lg = lg)

    #armonicos (con cache)
    pstr = join(["lead=$lead", "hpairs=$hpairs", "lambda=$(p.λ)", "t=$(p.t)", "Jsd=$(p.J_sd)",
                 "theta=$(p.θ)", "Omega=$(p.Ω)", "mu=$(p.μ)", "beta=$(p.β)", "N=$(p.N)", "K=$K",
                 "eta=$η", "Nomega=$Nω", "wmin=$ωmin", "wmax=$ωmax", "Nt=$Nt",
                 "periods=$periods", "tail=$tailmode"], ";")
    htag  = string(hash(pstr); base = 16)
    hfile = joinpath(outdir, "harmonics_$(htag).npy")
    hmeta = joinpath(outdir, "harmonics_$(htag).txt")
    nk = 2K + 1; nh = length(hpairs)
    H = nothing; fmax = nothing
    if usecache && isfile(hfile) && isfile(hmeta) && strip(read(hmeta, String)) == pstr
        lg("armonicos: leidos de la cache $(basename(hfile))")
        H = read_npy_c16(hfile)
        fmax = vec(read_npy_c16(replace(hfile, ".npy" => "_fmax.npy")) .|> real)
    else
        lg("armonicos: calculando $(nk) armonicos × $(nh) pares × {r,<} ...")
        t0 = time()
        H, fmax = build_harmonics(p, hpairs, sites, sidx, K, TP, Tio, ωs, wts, τs;
                                  η = η, chunk = chunk, lg = lg, tail = usetail)
        lg(@sprintf("armonicos listos en %.1f s", time() - t0))
        if usecache
            write_npy(hfile, H)
            write_npy(replace(hfile, ".npy" => "_fmax.npy"), complex.(fmax))
            write(hmeta, pstr)
            lg("   cache -> $(basename(hfile))")
        end
    end
    H4 = reshape(H, Nτ, 4, nk, 2, nh)
    F4 = reshape(fmax, 4, nk, 2, nh)

    # chequeo de soporte en k
    lg("-"^78)
    lg("SOPORTE EN k: max|bloque_k| / max|bloque_0|   (w = en ω sobre la malla, t = en τ)")
    lg(@sprintf("%-11s %-3s", "(i,j)", "") * join([@sprintf("   |k|=%d: w        t      ", kab) for kab in 1:K]))
    worst = zeros(K, 2, 2)      # (|k|, c, dominio)
    for (hp, (ni, nj)) in enumerate(hpairs), c in 1:2
        w0 = maximum(F4[:, K + 1, c, hp]); t0 = maximum(abs, view(H4, :, :, K + 1, c, hp))
        line = @sprintf("(%2d,%2d)     %-3s", ni + 1, nj + 1, c == 1 ? "r" : "<")
        for kab in 1:K
            kks = (K + 1 - kab, K + 1 + kab)
            rw = maximum(maximum(F4[:, kk, c, hp]) for kk in kks) / w0
            rt = maximum(maximum(abs, view(H4, :, :, kk, c, hp)) for kk in kks) / t0
            worst[kab, c, 1] = max(worst[kab, c, 1], rw)
            worst[kab, c, 2] = max(worst[kab, c, 2], rt)
            line *= @sprintf("   %9.2e %9.2e  ", rw, rt)
        end
        lg(line)
    end
    lg("maximo sobre pares:")
    for c in 1:2, kab in 1:K
        lg(@sprintf("   %s  |k|=%d :  en ω %.2e   en τ %.2e", c == 1 ? "r" : "<", kab,
                    worst[kab, c, 1], worst[kab, c, 2]))
    end

    # χ(t,t') por par
    lg("-"^78)
    lg("KERNEL")
    E    = [cis(-(kk - 1 - K) * p.Ω * ts[a]) for a in 1:Nt, kk in 1:nk]
    kin  = collect((K + 1 - 2):(K + 1 + 2))
    kout = [kk for kk in 1:nk if !(kk in kin)]
    Smu  = [PAULI[μ] for (μ, _) in mns]
    Snu  = [PAULI[ν] for (_, ν) in mns]
    nmn  = length(mns)
    Jsd  = p.J_sd
    # stats[q, :] = max|χ|, max|χ-χ_{|k|≤2}|, max|χ| (t'>t), max|χ| (t'<t), max|Imχ|, max|Reχ|
    stats_by_pair = Dict{Tuple{Int,Int},Matrix{Float64}}()
    shortcut = Dict{Tuple{Int,Int},NTuple{2,Float64}}()

    @inline function hsum(Hc, ti, a, ks)
        s1 = zero(ComplexF64); s2 = zero(ComplexF64); s3 = zero(ComplexF64); s4 = zero(ComplexF64)
        @inbounds for kk in ks
            e = E[a, kk]
            s1 += e * Hc[ti, 1, kk]; s2 += e * Hc[ti, 2, kk]
            s3 += e * Hc[ti, 3, kk]; s4 += e * Hc[ti, 4, kk]
        end
        return M2(s1, s2, s3, s4)
    end

    tk = time()
    for (I, J) in req
        hp, hq = hidx[(I, J)], hidx[(J, I)]
        Hr  = H4[:, :, :, 1, hp]
        Hl  = H4[:, :, :, 2, hp]
        Hlq = H4[:, :, :, 2, hq]
        χ   = Array{ComplexF64}(undef, Nt, Nt, nmn)
        rows = collect(Iterators.partition(1:Nt, cld(Nt, 4 * Threads.nthreads())))
        tasks = map(rows) do rr
            Threads.@spawn begin
                st = zeros(nmn, 6); dsh = 0.0; msh = 0.0
                for a in rr, b in 1:Nt
                    ti = a - b + Nt                  # τ = t_a - t'_b
                    tq = b - a + Nt                  # τ = t'_b - t_a
                    Gr2  = hsum(Hr, ti, a, kin);  Gr  = Gr2  + hsum(Hr, ti, a, kout)
                    Gl2  = hsum(Hl, ti, a, kin);  Gl  = Gl2  + hsum(Hl, ti, a, kout)
                    Glq2 = hsum(Hlq, tq, b, kin); Glq = Glq2 + hsum(Hlq, tq, b, kout)  # G^<_ji(t',t)
                    Ga  = adjoint(Gr)                # G^a_ji(t',t)
                    Ga2 = adjoint(Gr2)
                    dsh = max(dsh, maxabs(Glq + adjoint(Gl)))
                    msh = max(msh, maxabs(Gl))
                    for q in 1:nmn
                        Sm, Sn = Smu[q], Snu[q]
                        x  = -im * Jsd * tr2(Sm * (Gl * Sn * Ga + Gr * Sn * Glq))
                        x2 = -im * Jsd * tr2(Sm * (Gl2 * Sn * Ga2 + Gr2 * Sn * Glq2))
                        χ[a, b, q] = x
                        ax = abs(x)
                        st[q, 1] = max(st[q, 1], ax)
                        st[q, 2] = max(st[q, 2], abs(x - x2))
                        b > a && (st[q, 3] = max(st[q, 3], ax))
                        b < a && (st[q, 4] = max(st[q, 4], ax))
                        st[q, 5] = max(st[q, 5], abs(imag(x)))
                        st[q, 6] = max(st[q, 6], abs(real(x)))
                    end
                end
                (st, dsh, msh)
            end
        end
        res = fetch.(tasks)
        # OJO: no llamar `st` a esto. Un nombre asignado aqui Y dentro de la
        # tarea de arriba lo comparten todas las tareas (Julia lo captura), y
        # las estadisticas se pisan entre hilos. χ no se afecta, solo los chequeos.
        stp = reduce((x, y) -> max.(x, y), first.(res))
        stats_by_pair[(I, J)] = stp
        shortcut[(I, J)] = (maximum(r[2] for r in res), maximum(r[3] for r in res))
        for (q, (μ, ν)) in enumerate(mns)
            write_npy(joinpath(outdir, "time_kernel_$(μ)$(ν)_$(tagS)_i$(I)_j$(J).npy"), χ[:, :, q])
        end
        lg(@sprintf("   (i,j)=(%2d,%2d) listo  (%.1f s acumulado)", I, J, time() - tk))
    end

    # resumen
    lg("-"^78)
    lg("RESUMEN POR (μ,ν)  (maximo sobre pares; relativos a max|χ| del par)")
    lg("  μν   max|χ|      |χ_K-χ_2|/|χ|  |χ(t'>t)|/|χ|  |χ(t'<t)|/|χ|  max|Imχ|/max|Reχ|")
    for (q, (μ, ν)) in enumerate(mns)
        m1 = 0.0; r2 = 0.0; ru = 0.0; rl = 0.0; ri = 0.0
        for pr in req
            s = stats_by_pair[pr][q, :]
            m1 = max(m1, s[1])
            s[1] > 0 || continue
            r2 = max(r2, s[2] / s[1]); ru = max(ru, s[3] / s[1]); rl = max(rl, s[4] / s[1])
            s[6] > 0 && (ri = max(ri, s[5] / s[6]))
        end
        lg(@sprintf("  %s%s  %.3e   %.3e      %.3e      %.3e      %.3e", μ, ν, m1, r2, ru, rl, ri))
    end
    lg("detalle por par (μ,ν) = " * join([μ * ν for (μ, ν) in mns], ","))
    for pr in req
        s = stats_by_pair[pr]
        lg(@sprintf("  (%2d,%2d)  max|χ| = %s", pr[1], pr[2], join([@sprintf("%.2e", s[q, 1]) for q in 1:nmn], " ")))
        lg(@sprintf("           t'>t    = %s", join([@sprintf("%.2e", s[q, 3]) for q in 1:nmn], " ")))
        lg(@sprintf("           |Imχ|   = %s", join([@sprintf("%.2e", s[q, 5]) for q in 1:nmn], " ")))
        lg(@sprintf("           Δ|k|≤2  = %s", join([@sprintf("%.2e", s[q, 2]) for q in 1:nmn], " ")))
    end
    lg("atajo G^<_ji(t',t) = -[G^<_ij(t,t')]† (solo se compara, no se usa):")
    for pr in req
        d, m = shortcut[pr]
        lg(@sprintf("  (%2d,%2d)  max|directo - atajo| = %.2e   (max|G^<| = %.2e, rel %.2e)",
                    pr[1], pr[2], d, m, m > 0 ? d / m : 0.0))
    end
    lg("="^78)

    # metadatos y chequeos
    meta = [
        "sites" => labels, "site_convention" => "i = n + 1 (i=1 es la interfaz n=0)",
        "lead" => lead, "pairs" => [[I, J] for (I, J) in req], "mu_nu" => [μ * ν for (μ, ν) in mns],
        "Nt" => Nt, "periods" => periods, "T" => T, "dt" => dt, "Omega" => p.Ω,
        "index_convention" => "chi[a,b] = chi(t_a, t'_b), t_a = (a-1)*dt",
        "file_pattern" => "time_kernel_{mu}{nu}_$(tagS)_i{I}_j{J}.npy",
        "lambda" => p.λ, "t_hop" => p.t, "J_sd" => p.J_sd, "theta_deg" => rad2deg(p.θ),
        "E_F" => p.μ, "beta" => p.β, "kmax" => K, "N_floquet" => p.N,
        "eta" => η, "Nomega" => Nω, "omega_min" => ωmin, "omega_max" => ωmax, "tail" => tailmode,
    ]
    write_json(joinpath(outdir, "time_kernel_$(tagS)_meta.json"), meta)
    open(joinpath(outdir, "time_kernel_$(tagS)_checks_$(tagMN).txt"), "w") do io
        foreach(s -> println(io, s), logbuf)
    end
    println("salida en ", outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
