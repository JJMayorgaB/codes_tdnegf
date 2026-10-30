#!/usr/bin/env julia
#=
bond_current_llg.jl -- matriz densidad de los F (enlace a -> b y sitios a, b) a partir de la
G lesser vestida con los momentos de la LLG efectiva en el estado estacionario, calculada en
espacio de Floquet.

Funciones de Green del lead derecho con el driver por embedding (kernel_elements.jl, k = 0, ±1):

    G^μ_{mn}(t,t') = Σ_k e^{-ikΩt} ∫dω/2π e^{-iω(t-t')} 𝒢^μ_{mn,k}(ω)

con t el tiempo de Floquet (t_lab - t_on). Matriz de Floquet de una funcion de dos tiempos:
Ǎ_{pq}(ω) = A(ω+pΩ, ω+qΩ), es decir Ǧ^μ_{mn,pq}(ω) = 𝒢^μ_{mn,p-q}(ω+qΩ).

Momentos de los F en el estado estacionario (punto fijo de la LLG en el marco rotante):
M_j(t) = R_y(Ωt) M̄_j, de modo que

    σ·M_j(t) = Σ_{q=0,±1} e^{-iqΩt} V̌_{j,q},
    V̌_{j,0} = M̄^y_j σy,   V̌_{j,+1} = (M̄^z_j - i M̄^x_j) τ+,   V̌_{j,-1} = (M̄^z_j + i M̄^x_j) τ-,

y su matriz de Floquet es V̌_{j,pq} = V̌_{j,p-q}. Con H^sd_F = -J_sd Σ_{j=a,b} σ·M_j, la expansion
de Dyson 𝙂 = G + G H^sd_F G + G H^sd_F G H^sd_F G + ... da, por Langreth (V = -J_sd σ·M_j en el
sitio j, j = a, b):

    orden 0:  G^<
    orden 1:  G^r V G^<  +  G^< V G^a
    orden 2:  G^r V G^r V G^<  +  G^r V G^< V G^a  +  G^< V G^a V G^a

donde los productos en el tiempo son productos de matrices de Floquet (sitios a, b x armonicos
p = -P..P x espin). La matriz densidad a tiempos iguales:

    ρ_{mn}(t) = -i 𝙂^<_{mn}(t,t) = Σ_k e^{-ikΩt} ρ_{mn,k},   ρ_{mn,k} = -i ∫dω/2π [𝙂̌^<_{mn}]_{k0}(ω)

Cada factor cambia el indice de Floquet en a lo sumo 1, asi que con P ≥ 2·orden + 1 los
productos son exactos (sin error de truncamiento). La integral en ω se hace en la malla
ω = s dω, dω = Ω/nΩ, η = 5 dω (como s̄ en kernel_elements.jl --sbar), con nΩ y 2nΩ, y se
extrapola a η -> 0 (Richardson): ρ = 2 ρ(η/2) - ρ(η).

Salida, con el mismo formato que bond_rho_t.csv de TDNEGF (sitios de TDNEGF, lead = C,
rho_nm = ρ_{nm}(t), rho_mn = ρ_{mn}(t)), ordenes 0 .. --order sumados:
    output/<tag>/F<a>-<b>/bond_rho_llg.csv     pares (n,m) = (a,b), (a,a), (b,b), tiempos de la
                                              LLG en sus ultimos --last + 1 periodos
La corriente y la figura las hace plot_bond_current_llg.py (con bond_currents, como TDNEGF).

Uso (desde LLG_equation):
    julia -t auto bond_current_llg.jl --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005 --sites 1,2
        [--order 2] [--nomega 40] [--nfloq 6] [--margin 2.0] [--last 5]
=#

include(joinpath(@__DIR__, "..", "kernel_elements.jl"))

const HERE_B   = @__DIR__
const TDROOT_B = joinpath(HERE_B, "..", "..", "two_oscillators_setup", "y_precession", "output", "DFF")
const SIGS = (σx, σy, σz)

# ---------------------------------------------------------------- bloques 2x2 de 𝒢^μ_{mn,k}(ω)
"""
Suma en G[r0+1:r0+2, c0+1:c0+2] el bloque 2x2 del armonico k de 𝒢^μ_{mn,k}(ω), con h la fila de
12 escalares (orden COEFS) e i0 = 0 (r), 4 (a), 8 (<):
    k = 0: α σ0 + β σy,   k = +1: γ+ τ+,   k = -1: γ- τ-.
"""
@inline function putblk!(G::Matrix{ComplexF64}, r0::Int, c0::Int, h, i0::Int, k::Int)
    if k == 0
        α = h[i0+1]; β = h[i0+2]
        G[r0+1, c0+1] += α;       G[r0+2, c0+2] += α
        G[r0+1, c0+2] += -im * β; G[r0+2, c0+1] += im * β
    elseif k == 1                                   # τ+ = (σz + iσx)/2
        g = h[i0+3] / 2
        G[r0+1, c0+1] += g;      G[r0+1, c0+2] += im * g
        G[r0+2, c0+1] += im * g; G[r0+2, c0+2] += -g
    else                                            # τ- = (σz - iσx)/2
        g = h[i0+4] / 2
        G[r0+1, c0+1] += g;       G[r0+1, c0+2] += -im * g
        G[r0+2, c0+1] += -im * g; G[r0+2, c0+2] += -g
    end
    return nothing
end

# ---------------------------------------------------------------- ρ_{mn,k} en Floquet
"""
    floquet_rho(p, sites, Mb, nΩ, P, margin, order) -> (ρ, ks, η, Nω)

ρ[ik, m, n, σ, σ', o+1] = contribucion de orden o (0..order) a ρ_{mn,k} (m, n = posiciones 1, 2
de sites; k = ks[ik]), con un solo nΩ (sin extrapolar).
"""
function floquet_rho(p::KParams, sites, Mb, nΩ::Int, P::Int, margin::Real, order::Int)
    dω = p.Ω / nΩ; η = 5 * dω
    Lmin = -(sqrt(4γt(p)^2 + p.J_sd^2) + margin) - (P + 1) * p.Ω           # banda + estados ligados
    Lmax = p.μ + (isfinite(p.β) ? 40 / p.β : 0.0) + (P + 1) * p.Ω           # Fermi + replicas
    s1 = floor(Int, Lmin / dω); s2 = ceil(Int, Lmax / dω)
    sh = P * nΩ                                                            # ω + qΩ, |q| ≤ P
    ωs = collect((s1 - sh):(s2 + sh)) .* dω
    pairs = [(sites[m], sites[n]) for m in 1:2 for n in 1:2]
    pos   = [(m, n) for m in 1:2 for n in 1:2]
    H = harmonics_grid(ωs, pairs, p; η = η)                                # (Nω, 4, 12)

    nF = 2P + 1; N = 4nF
    lin0(s, q) = ((s - 1) * nF + (q + P)) * 2                              # (sitio, armonico) -> fila - 1
    V = zeros(ComplexF64, N, N)
    for j in 1:2
        Mx, My, Mz = Mb[j]
        Vh = Dict(0 => My * σy, 1 => (Mz - im * Mx) * τp, -1 => (Mz + im * Mx) * τm)
        for q in -P:P, k in -1:1
            pq = q + k
            -P ≤ pq ≤ P || continue
            V[lin0(j, pq)+1:lin0(j, pq)+2, lin0(j, q)+1:lin0(j, q)+2] .= -p.J_sd .* Vh[k]
        end
    end
    cols = [lin0(n, 0) + b for n in 1:2 for b in 1:2]                      # columna q = 0
    K = 2order + 1
    ks = collect(-K:K)
    nint = s2 - s1 + 1
    # bloques de ω, uno por tarea (no se usa threadid(): en Julia ≥ 1.12 puede superar nthreads())
    chunks = collect(Iterators.partition(1:nint, cld(nint, nthreads())))
    acc = [zeros(ComplexF64, length(ks), 2, 2, 2, 2, order + 1) for _ in chunks]
    @threads for ic in eachindex(chunks)
      Gr = zeros(ComplexF64, N, N); Ga = zeros(ComplexF64, N, N); Gl = zeros(ComplexF64, N, N)
      A = acc[ic]
      for ii in chunks[ic]
        fill!(Gr, 0); fill!(Ga, 0); fill!(Gl, 0)
        i0 = ii + sh                                                       # ω de integracion en ωs
        for (ip, (m, n)) in enumerate(pos), q in -P:P
            h = view(H, i0 + q * nΩ, ip, :)                                # 𝒢(ω + qΩ)
            for k in -1:1
                pq = q + k
                -P ≤ pq ≤ P || continue
                r0 = lin0(m, pq); c0 = lin0(n, q)
                putblk!(Gr, r0, c0, h, 0, k); putblk!(Ga, r0, c0, h, 4, k); putblk!(Gl, r0, c0, h, 8, k)
            end
        end
        X0 = Gl[:, cols]
        Xs = Matrix{ComplexF64}[X0]
        if order ≥ 1
            VX0 = V * X0; VGa = V * Ga[:, cols]
            push!(Xs, Gr * VX0 + Gl * VGa)                                 # G^r V G^< + G^< V G^a
            if order ≥ 2                                                   # los tres de orden 2
                push!(Xs, Gr * (V * (Gr * VX0)) + Gr * (V * (Gl * VGa)) + Gl * (V * (Ga * VGa)))
            end
        end
        for (o, X) in enumerate(Xs), (ik, k) in enumerate(ks), m in 1:2
            r0 = lin0(m, k)
            for n in 1:2, a in 1:2, b in 1:2
                A[ik, m, n, a, b, o] += X[r0+a, (n-1)*2+b]
            end
        end
      end
    end
    ρ = sum(acc) .* (-im * dω / (2π))
    return ρ, ks, η, nint
end

"ρ_{mn}(t) (2x2) a partir de los armonicos, con los ordenes os sumados."
function rho_t(ρ, ks, m::Int, n::Int, tF::Real, Ω::Real, os)
    r = zeros(ComplexF64, 2, 2)
    for o in os, (ik, k) in enumerate(ks)
        r .+= cis(-k * Ω * tF) .* ρ[ik, m, n, :, :, o+1]
    end
    return r
end

"Promedio en un periodo, en el marco rotante, de Tr[σ ρ_{mn}(t)] (vector complejo de 3)."
function rot_mean(ρ, ks, m, n, Ω, os; nt = 400)
    T = 2π / Ω; v = zeros(ComplexF64, 3)
    for it in 0:nt-1
        t = it * T / nt
        r = rho_t(ρ, ks, m, n, t, Ω, os)
        w = [tr(s * r) for s in SIGS]
        c, s = cos(Ω * t), sin(Ω * t)                                     # R_y(-Ωt) w
        v .+= [c * w[1] - s * w[3], w[2], s * w[1] + c * w[3]]
    end
    return v ./ nt
end

# ---------------------------------------------------------------- main
function main_bond(args = ARGS)
    opt = Dict("--tag" => "setup2_gso0p1_jsd0p2_th10deg_Om0p005", "--sites" => "1,2", "--last" => "5",
               "--order" => "2", "--nomega" => "40", "--nfloq" => "", "--margin" => "2.0", "--tdnegf" => "")
    i = 1
    while i ≤ length(args)
        haskey(opt, args[i]) && i < length(args) || error("argumento no reconocido: $(args[i])")
        opt[args[i]] = args[i+1]; i += 2
    end
    tag = opt["--tag"]
    a, b = parse.(Int, split(opt["--sites"], ','))
    order = parse(Int, opt["--order"])
    0 ≤ order ≤ 2 || error("--order: 0, 1 o 2")
    P = isempty(opt["--nfloq"]) ? 2order + 2 : parse(Int, opt["--nfloq"])
    P ≥ 2order + 1 || error("--nfloq tiene que ser ≥ 2·orden + 1 = $(2order + 1)")
    nΩ = parse(Int, opt["--nomega"]); margin = parse(Float64, opt["--margin"])
    BLAS.set_num_threads(1)                         # matrices pequeñas: el paralelismo va en ω

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
    T = 2π / p.Ω

    # ---- punto fijo de la LLG en el marco rotante: promedio del ultimo periodo
    Rr = readdlm(joinpath(llgdir, "spins_rot_llg.csv"), ',', Float64; skipstart = 1)
    w = findall(≥(Rr[end, 1] - T), Rr[:, 1])
    Mb = [vec(sum(Rr[w, c0:c0+2]; dims = 1)) ./ length(w) for c0 in (5, 8)]   # M2 = F1, M3 = F2
    amp = [maximum(sqrt.(sum((Rr[w, c0:c0+2] .- Mb[k]') .^ 2; dims = 2))) for (k, c0) in enumerate((5, 8))]
    @printf("output/%s/F%d-%d   orden %d   P = %d   nΩ = %d y %d   hilos = %d\n", tag, a, b, order, P, nΩ, 2nΩ, nthreads())
    for (k, s) in enumerate((a, b))
        @printf("  M̄_%d = (%+.6f, %+.6f, %+.6f)   max|M̄(t) - M̄| en el ultimo periodo = %.1e\n", s, Mb[k]..., amp[k])
    end

    # ---- ρ_{mn,k} con nΩ y 2nΩ, extrapolado a η -> 0
    t0 = time()
    ρ1, ks, η1, N1 = floquet_rho(p, (a, b), Mb, nΩ, P, margin, order)
    ρ2, _,  η2, N2 = floquet_rho(p, (a, b), Mb, 2nΩ, P, margin, order)
    ρ = 2 .* ρ2 .- ρ1
    @printf("  ρ_(mn,k): η = %.3e y %.3e (%d y %d puntos en ω), extrapolado a η -> 0 (%.1f s)\n",
            η1, η2, N1, N2, time() - t0)
    @printf("  cambio η -> η/2 (relativo): %.1e\n", maximum(abs.(ρ2 .- ρ1)) / maximum(abs.(ρ)))
    for m in 1:2                                                       # control: orden cero en t = 0
        r0 = rho_t(ρ, ks, m, m, 0.0, p.Ω, 0:0)
        @printf("  orden cero, s̄_%d = (%+.6e, %+.6e, %+.6e)\n", (a, b)[m], (real(tr(s * r0)) for s in SIGS)...)
    end
    println("  promedio en el marco rotante de Tr[σ ρ^ab] (ρ^ab = ρ_(ba)), <σ_a> y <σ_b>, por orden:")
    for o in 0:order
        v = rot_mean(ρ, ks, 2, 1, p.Ω, o:o)
        sa = real.(rot_mean(ρ, ks, 1, 1, p.Ω, o:o)); sb = real.(rot_mean(ρ, ks, 2, 2, p.Ω, o:o))
        @printf("    orden %d: Tr[σ ρ^ab] = (%+.3e%+.3ei, %+.3e%+.3ei, %+.3e%+.3ei)   <σ_a> = (%+.5f, %+.5f, %+.5f)   <σ_b> = (%+.5f, %+.5f, %+.5f)\n",
                o, real(v[1]), imag(v[1]), real(v[2]), imag(v[2]), real(v[3]), imag(v[3]), sa..., sb...)
    end

    # ---- serie temporal en los tiempos de la LLG (ultimos --last + 1 periodos)
    L = readdlm(joinpath(llgdir, "spins_t_llg.csv"), ',', Float64; skipstart = 1)
    tL = L[:, 1]
    t_out = tL[findall(≥(tL[end] - (parse(Float64, opt["--last"]) + 1) * T - 1e-9), tL)]
    geo = readdlm(joinpath(tdir, "geometry.csv"), ',', String; skipstart = 1)
    sD = parse(Int, geo[findfirst(==("D"), geo[:, 3]), 2])
    path = joinpath(llgdir, "bond_rho_llg.csv")
    ent = ((1, 1), (1, 2), (2, 1), (2, 2))                             # uu, ud, du, dd
    os = 0:order
    open(path, "w") do io
        print(io, "t,lead,n,m")
        for blk in ("nm", "mn"), s in ("uu", "ud", "du", "dd")
            print(io, ",Re_rho_", blk, "_", s, ",Im_rho_", blk, "_", s)
        end
        println(io)
        for (m, n) in ((1, 2), (1, 1), (2, 2))
            for t in t_out
                tF = t - t_on
                @printf(io, "%.10g,C,%d,%d", t, sD + (a, b)[m], sD + (a, b)[n])
                for X in (rho_t(ρ, ks, m, n, tF, p.Ω, os), rho_t(ρ, ks, n, m, tF, p.Ω, os)), (r, s) in ent
                    @printf(io, ",%.15e,%.15e", real(X[r, s]), imag(X[r, s]))
                end
                println(io)
            end
        end
    end
    println("  -> ", path, "   (ordenes 0..", order, " sumados, ", length(t_out), " tiempos)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_bond()
end
