#!/usr/bin/env julia
#=
kernel_harmonics.jl -- armonicos de Floquet del kernel temporal,

  χ^{μν}_{ij}(t,t') = Σ_k e^{-ikΩt} χ^{μν}_{ij,k}(τ),      τ = t - t',

en una malla 1D de τ (fina si se quiere), SIN pasar por la malla (t,t').

Con G_ab(t,t') = Σ_k e^{-ikΩt} 𝒢^{ab}_k(τ) (lo mismo que usa time_kernel.jl),
cada termino de χ es un producto de dos sumas finitas en el mismo punto:

  G^<_ij(t,t') σ^ν G^a_ji(t',t) = Σ_{k,k'} e^{-i(k-k')Ωt} 𝒢^<_k(τ) σ^ν [𝒢^r_{k'}(τ)]†
  G^r_ij(t,t') σ^ν G^<_ji(t',t) = Σ_{k,k'} e^{-i(k+k')Ωt} e^{ik'Ωτ} 𝒢^r_k(τ) σ^ν 𝒢^{<,ji}_{k'}(-τ)

(la fase e^{ik'Ωτ} sale de que G^<_ji(t',t) tiene su primer argumento en
t' = t - τ). Agrupando por la potencia de e^{-iΩt}:

  χ_k(τ) = -i J_sd Tr_σ σ^μ [ Σ_{k1-k2=k} 𝒢^<_{k1} σ^ν 𝒢^{r†}_{k2}
                            + Σ_{k1+k2=k} e^{ik2Ωτ} 𝒢^r_{k1} σ^ν 𝒢^{<,ji}_{k2}(-τ) ]

Es exactamente la cuenta de time_kernel.jl en otro orden (primero se
multiplica y despues se agrupan las fases), asi que no hay aproximacion nueva.
El modo --verify lo comprueba contra los .npy de χ(t,t') ya calculados.

Uso:
  julia -t 64 kernel_harmonics.jl --sites 1,2,3,4 --dtau 0.5 --taumax 200 --outdir output/kh_dt0.5
  julia -t 64 kernel_harmonics.jl --verify output/time_kernel_eta6b      # compara contra time_kernel

Salida (en --outdir):
  kh_{μ}{ν}_{sitios}.npy     χ_k complejo, forma (Nτ, 5, n, n):
                             [τ, k+2 (k=-2..2, indices de numpy), indice de i, indice de j]
  kh_{sitios}_tau.npy        malla τ ∈ [-τmax, τmax] (se guarda τ<0 tambien)
  kh_{sitios}_meta.json      parametros y convenciones
  kh_{sitios}_checks.txt     chequeos
  harmonics_{hash}.npy/.txt  cache de los armonicos de G (formato de time_kernel; no se comparte, la clave incluye dtau)
=#

include(joinpath(@__DIR__, "time_kernel.jl"))    # trae M2, build_harmonics, validaciones, E/S

using LinearAlgebra
using Printf

"""
    chi_harmonics(H4, τs, p, pairs, hidx, mns, K) -> Dict((I,J) => Array (Nτ, 4K+1, nmn))

H4[it, e, kk, c, hp]: armonicos de G (e = entrada 2x2, kk = k+K+1, c = 1 r / 2 <).
La malla τs tiene que ser simetrica, τs[it] = -τs[Nτ+1-it], para leer 𝒢^{<,ji}(-τ).
Devuelve χ_k para k = -2K..2K (el producto de dos armonicos |k| ≤ K).
"""
function chi_harmonics(H4, τs, p, pairs, hidx, mns, K)
    Nτ  = length(τs)
    nk  = 2K + 1
    KK  = 2K
    nkc = 2KK + 1
    Ω   = p.Ω
    Jsd = p.J_sd
    Smu = [PAULI[μ] for (μ, _) in mns]
    Snu = [PAULI[ν] for (_, ν) in mns]
    nmn = length(mns)
    out = Dict{Tuple{Int,Int},Array{ComplexF64,3}}()
    for pr in pairs
        out[pr] = zeros(ComplexF64, Nτ, nkc, nmn)
    end
    blk(it, kk, c, hp) = M2(H4[it, 1, kk, c, hp], H4[it, 2, kk, c, hp],
                            H4[it, 3, kk, c, hp], H4[it, 4, kk, c, hp])
    Threads.@threads for pr in pairs
        I, J = pr
        hp, hq = hidx[(I, J)], hidx[(J, I)]
        X = out[pr]
        acc = Vector{M2}(undef, nkc)
        for it in 1:Nτ
            iq  = Nτ + 1 - it                    # -τ
            τ   = τs[it]
            Gr  = [blk(it, kk, 1, hp) for kk in 1:nk]
            Gl  = [blk(it, kk, 2, hp) for kk in 1:nk]
            Glq = [blk(iq, kk, 2, hq) for kk in 1:nk]
            Gra = [adjoint(g) for g in Gr]
            for q in 1:nmn
                Sm, Sn = Smu[q], Snu[q]
                fill!(acc, Z2)
                for a in 1:nk, b in 1:nk
                    k1 = a - 1 - K
                    k2 = b - 1 - K
                    acc[k1 - k2 + KK + 1] += Gl[a] * Sn * Gra[b]
                    acc[k1 + k2 + KK + 1] += cis(k2 * Ω * τ) * (Gr[a] * Sn * Glq[b])
                end
                for kc in 1:nkc
                    X[it, kc, q] = -im * Jsd * tr2(Sm * acc[kc])
                end
            end
        end
    end
    return out
end

# ======================================================================
# modo --verify: reconstruye χ(t,t') desde la cache de time_kernel.jl
# ======================================================================
function kh_verify(dir::AbstractString)
    metaf = filter(f -> endswith(f, "_meta.json") && startswith(f, "time_kernel_"), readdir(dir))
    isempty(metaf) && error("no hay time_kernel_*_meta.json en $dir")
    txt = read(joinpath(dir, metaf[1]), String)
    getnum(k) = parse(Float64, match(Regex("\"$k\":\\s*([-0-9.eE+]+)"), txt)[1])
    Nt = Int(getnum("Nt")); dt = getnum("dt"); Ω = getnum("Omega"); Jsd = getnum("J_sd")
    labs = parse.(Int, split(match(r"\"sites\":\s*\[([^\]]*)\]", txt)[1], ","))
    tagS = join(labs, "-")
    println("verificando contra $(metaf[1])  (sitios $tagS, Nt=$Nt, dt=$dt)")

    # cache de armonicos: el .txt tiene el pstr con hpairs y K
    cands = filter(f -> startswith(f, "harmonics_") && endswith(f, ".txt"), readdir(dir))
    isempty(cands) && error("no hay cache harmonics_*.txt en $dir")
    pstr = strip(read(joinpath(dir, cands[1]), String))
    length(cands) > 1 && println("  (hay $(length(cands)) caches; uso $(cands[1]))")
    K  = parse(Int, match(r"(?:^|;)K=(\d+)", pstr)[1])
    hp = [(parse(Int, m[1]), parse(Int, m[2])) for m in eachmatch(r"\((\d+),\s*(\d+)\)",
          match(r"hpairs=\[(.*?)\]", pstr)[1])]
    H  = read_npy_c16(joinpath(dir, replace(cands[1], ".txt" => ".npy")))
    Nτ = 2Nt - 1
    nk = 2K + 1
    H4 = reshape(H, Nτ, 4, nk, 2, length(hp))
    τs = [(it - Nt) * dt for it in 1:Nτ]
    hl   = [(a + 1, b + 1) for (a, b) in hp]
    hidx = Dict(hl[q] => q for q in eachindex(hl))
    p = FloquetParams(; Ω = Ω, J_sd = Jsd, N = K + 2)
    comps = ["x", "y", "z"]
    mns = [(μ, ν) for μ in comps for ν in comps]
    pairs = [(I, J) for J in labs for I in labs]
    ch = chi_harmonics(H4, τs, p, pairs, hidx, mns, K)
    KK = 2K
    ts = [(a - 1) * dt for a in 1:Nt]
    worst = 0.0; esc = 0.0; nfiles = 0
    for pr in pairs, (q, (μ, ν)) in enumerate(mns)
        f = joinpath(dir, "time_kernel_$(μ)$(ν)_$(tagS)_i$(pr[1])_j$(pr[2]).npy")
        isfile(f) || continue
        ref = read_npy_c16(f)
        X = ch[pr]
        for a in 1:Nt, b in 1:Nt
            ti = a - b + Nt
            v = zero(ComplexF64)
            for kc in 1:(2KK + 1)
                v += cis(-(kc - 1 - KK) * Ω * ts[a]) * X[ti, kc, q]
            end
            worst = max(worst, abs(v - ref[a, b])); esc = max(esc, abs(ref[a, b]))
        end
        nfiles += 1
    end
    @printf("archivos comparados: %d\nmax|Σ_k e^{-ikΩt} χ_k(τ) - χ(t,t')| = %.3e   (escala %.3e, rel %.3e)\n",
            nfiles, worst, esc, worst / esc)
    println(worst / esc < 1e-10 ? "VERIFICACION OK" : "VERIFICACION FALLA")
    return nothing
end

# ======================================================================
# CLI
# ======================================================================
const KH_USAGE = """
julia -t NHILOS kernel_harmonics.jl [opciones]

  --sites 1,2,3,4      etiquetas de sitio (i=1 es la interfaz n=0); pares = todos los (i,j)
  --lead R|L
  --mu x --nu y        solo ese (μ,ν); sin ellos, los 9
  --dtau 0.5           paso de la malla en τ
  --taumax 200         τ ∈ [0, taumax]
  --kmax 2             armonico maximo de G (N = kmax + 2); χ sale hasta |k| = 2 kmax
  --eta auto           auto = 0.01/taumax
  --Nomega auto        auto: dω = η/2
  --omega-min -15 --omega-max 15
  --tail free|none     como en time_kernel.jl
  --chunk 4096
  --outdir DIR         por defecto output/kernel_harmonics
  --no-cache
  --lambda --Jsd --theta(grados) --Omega --EF --beta
  --verify DIR         compara contra los .npy de time_kernel.jl en DIR (usa su cache)
"""
const KH_VALKEYS = Set(["sites", "lead", "mu", "nu", "dtau", "taumax", "kmax", "eta", "Nomega",
                        "omega-min", "omega-max", "tail", "chunk", "outdir", "lambda", "Jsd",
                        "theta", "Omega", "EF", "beta", "verify"])
const KH_FLAGKEYS = Set(["no-cache", "help"])

function kh_parse(argv)
    o = Dict{String,String}()
    q = 1
    while q <= length(argv)
        a = argv[q]
        startswith(a, "--") || error("argumento inesperado: $a\n$KH_USAGE")
        key = a[3:end]
        if key in KH_FLAGKEYS
            o[key] = "true"; q += 1
        elseif key in KH_VALKEYS
            q < length(argv) || error("falta el valor de --$key")
            o[key] = argv[q + 1]; q += 2
        else
            error("opcion desconocida --$key\n$KH_USAGE")
        end
    end
    return o
end

function kh_main(argv = ARGS)
    o = kh_parse(argv)
    haskey(o, "help") && (print(KH_USAGE); return nothing)
    haskey(o, "verify") && return kh_verify(o["verify"])
    getf(k, d) = parse(Float64, get(o, k, string(d)))
    geti(k, d) = parse(Int, get(o, k, string(d)))

    labels = parse.(Int, split(get(o, "sites", "1,2,3,4"), ","))
    all(>=(1), labels) || error("--sites: las etiquetas empiezan en 1")
    tagS = join(labels, "-")
    lead = Symbol(get(o, "lead", "R"))
    lead in (:L, :R) || error("--lead debe ser L o R")      # T_depth trata cualquier otro como L
    pairs = [(I, J) for J in labels for I in labels]
    hl = unique(vcat(pairs, [(J, I) for (I, J) in pairs]))
    hpairs = [(I - 1, J - 1) for (I, J) in hl]
    hidx = Dict(hl[q] => q for q in eachindex(hl))
    sites = sort(unique(vcat(first.(hpairs), last.(hpairs))))
    sidx = Dict(n => q for (q, n) in enumerate(sites))
    comps = ["x", "y", "z"]
    mns = if haskey(o, "mu") || haskey(o, "nu")
        (haskey(o, "mu") && haskey(o, "nu")) || error("--mu y --nu van juntos")
        (o["mu"] in comps && o["nu"] in comps) || error("--mu/--nu deben ser x, y o z")
        [(o["mu"], o["nu"])]
    else
        [(μ, ν) for μ in comps for ν in comps]
    end

    K = geti("kmax", 2)
    p0 = FloquetParams()
    λ  = getf("lambda", p0.λ)
    p  = FloquetParams(; λ = λ, t = sqrt(1 - λ^2), J_sd = getf("Jsd", p0.J_sd),
                       θ = deg2rad(getf("theta", rad2deg(p0.θ))), Ω = getf("Omega", p0.Ω),
                       μ = getf("EF", p0.μ), β = getf("beta", p0.β), N = K + 2)

    dτ   = getf("dtau", 0.5)
    τmax = getf("taumax", 200.0)
    nτ   = floor(Int, τmax / dτ + 1e-9)
    τs   = [q * dτ for q in -nτ:nτ]                 # simetrica: hace falta -τ para G^<_ji
    Nτ   = length(τs)
    ηarg = get(o, "eta", "auto")
    η    = ηarg == "auto" ? 0.01 / (nτ * dτ) : parse(Float64, ηarg)
    ωmin = getf("omega-min", -15.0); ωmax = getf("omega-max", 15.0)
    Nω   = get(o, "Nomega", "auto") == "auto" ? ceil(Int, (ωmax - ωmin) / (η / 2)) + 1 :
           parse(Int, o["Nomega"])
    ωs   = collect(range(ωmin, ωmax; length = Nω))
    dω   = ωs[2] - ωs[1]
    wts  = fill(dω, Nω); wts[1] = wts[end] = dω / 2
    tailmode = get(o, "tail", "free")
    tailmode in ("free", "none") || error("--tail debe ser free o none")
    chunk  = geti("chunk", 4096)
    outdir = get(o, "outdir", joinpath(@__DIR__, "output", "kernel_harmonics"))
    mkpath(outdir)
    usecache = !haskey(o, "no-cache")

    logbuf = String[]
    lg(s) = (println(s); push!(logbuf, s); flush(stdout))
    lg("="^78)
    lg("KERNEL HARMONICS  χ^{μν}_{ij,k}(τ)")
    lg("="^78)
    lg(@sprintf("t=%.4f λ=%.4f J_sd=%.3f θ=%.2f° Ω=%.4f E_F=%.3f β=%.1f", p.t, p.λ, p.J_sd,
                rad2deg(p.θ), p.Ω, p.μ, p.β))
    lg(@sprintf("lead %s  sitios %s   kmax(G) = %d -> N = %d, χ hasta |k| = %d", lead,
                string(labels), K, p.N, 2K))
    lg(@sprintf("τ ∈ [0, %.2f]  dτ = %.4f  (%d puntos, malla simetrica de %d)", nτ * dτ, dτ, nτ + 1, Nτ))
    lg(@sprintf("frecuencia maxima sin aliasing π/dτ = %.3f", π / dτ))
    lg(@sprintf("ω ∈ [%.1f, %.1f]  Nω = %d  dω = %.3e  η = %.3e  (η·τmax = %.3f)  cola: %s",
                ωmin, ωmax, Nω, dω, η, η * nτ * dτ, tailmode))
    dω > η / 2 * (1 + 1e-9) && lg("   AVISO: dω > η/2")
    ωχ = 4γ_band(p) + 2p.J_sd + 4p.Ω                # frecuencia maxima de χ_k(τ)
    π / dτ < ωχ && lg(@sprintf("   AVISO: π/dτ = %.3f < %.3f: χ_k(τ) queda con ALIASING; use dτ < %.3f",
                               π / dτ, ωχ, π / ωχ))

    TP  = tpowers(lead, p, max(maximum(sites), 12))
    Tio = Tuple(M2.(T_inout(lead, p)))
    validations(p, lead, hpairs, sites, sidx, K, TP, Tio; η = η, lg = lg)

    pstr = join(["kh", "lead=$lead", "hpairs=$hpairs", "lambda=$(p.λ)", "t=$(p.t)",
                 "Jsd=$(p.J_sd)", "theta=$(p.θ)", "Omega=$(p.Ω)", "mu=$(p.μ)", "beta=$(p.β)",
                 "N=$(p.N)", "K=$K", "eta=$η", "Nomega=$Nω", "wmin=$ωmin", "wmax=$ωmax",
                 "dtau=$dτ", "ntau=$nτ", "tail=$tailmode"], ";")
    htag  = string(hash(pstr); base = 16)
    hfile = joinpath(outdir, "harmonics_$(htag).npy")
    hmeta = joinpath(outdir, "harmonics_$(htag).txt")
    nk = 2K + 1; nh = length(hpairs)
    if usecache && isfile(hfile) && isfile(hmeta) && strip(read(hmeta, String)) == pstr
        lg("armonicos de G: leidos de la cache")
        H = read_npy_c16(hfile)
    else
        lg("armonicos de G: calculando ...")
        t0 = time()
        H, _ = build_harmonics(p, hpairs, sites, sidx, K, TP, Tio, ωs, wts, τs;
                               η = η, chunk = chunk, lg = lg, tail = tailmode == "free")
        lg(@sprintf("listos en %.1f s", time() - t0))
        if usecache
            write_npy(hfile, H); write(hmeta, pstr)
        end
    end
    H4 = reshape(H, Nτ, 4, nk, 2, nh)

    ch = chi_harmonics(H4, τs, p, pairs, hidx, mns, K)

    # chequeos
    KK = 2K
    i0 = nτ + 1                                     # τ = 0
    lg("-"^78)
    lg("CHEQUEOS (maximo sobre pares, relativo a max|χ_k| del par)")
    lg("  μν   causal τ<0    |χ_{-k}-χ_k*|   peso |k|>2   max|χ_0|")
    for (q, (μ, ν)) in enumerate(mns)
        rc = 0.0; rr = 0.0; rk = 0.0; m0 = 0.0
        for pr in pairs
            X = ch[pr]
            s = maximum(abs, X[:, :, q])
            s > 0 || continue
            rc = max(rc, maximum(abs, X[1:i0-1, :, q]) / s)
            rr = max(rr, maximum(maximum(abs.(X[:, KK + 1 - m, q] .- conj.(X[:, KK + 1 + m, q])))
                                 for m in 1:KK) / s)
            if KK > 2
                rk = max(rk, maximum(abs, X[:, [1:KK-2; KK+4:2KK+1], q]) / s)
            end
            m0 = max(m0, maximum(abs, X[:, KK + 1, q]))
        end
        lg(@sprintf("  %s%s  %.2e      %.2e       %.2e     %.3e", μ, ν, rc, rr, rk, m0))
    end
    lg("  causal τ<0: χ_k(τ<0) deberia ser ~0;  χ_{-k} = χ_k* porque χ(t,t') es real")
    lg("="^78)

    # salida: malla τ COMPLETA (τ < 0 incluido: no se impone causalidad al guardar), k = -2..2
    n = length(labels)
    ks = (KK + 1 - 2):(KK + 1 + 2)
    for (q, (μ, ν)) in enumerate(mns)
        A = zeros(ComplexF64, Nτ, 5, n, n)
        for (c, I) in enumerate(labels), (r, J) in enumerate(labels)
            A[:, :, c, r] = ch[(I, J)][:, ks, q]
        end
        write_npy(joinpath(outdir, "kh_$(μ)$(ν)_$(tagS).npy"), A)
    end
    write_npy(joinpath(outdir, "kh_$(tagS)_tau.npy"), collect(τs))
    meta = [
        "sites" => labels, "lead" => lead, "dtau" => dτ, "taumax" => nτ * dτ, "ntau" => Nτ,
        "Omega" => p.Ω, "k" => collect(-2:2), "kmax_G" => K, "N_floquet" => p.N,
        "index_convention" => "numpy (0-based): A[tau, k+2, i_idx, j_idx], k = -2..2, tau in [-taumax, taumax]; chi(t,t') = sum_k exp(-i k Omega t) chi_k(t-t')",
        "eta" => η, "Nomega" => Nω, "omega_min" => ωmin, "omega_max" => ωmax, "tail" => tailmode,
        "lambda" => p.λ, "t_hop" => p.t, "J_sd" => p.J_sd, "theta_deg" => rad2deg(p.θ),
        "E_F" => p.μ, "beta" => p.β, "mu_nu" => [μ * ν for (μ, ν) in mns],
    ]
    write_json(joinpath(outdir, "kh_$(tagS)_meta.json"), meta)
    open(joinpath(outdir, "kh_$(tagS)_checks.txt"), "w") do io
        foreach(s -> println(io, s), logbuf)
    end
    println("salida en ", outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    kh_main(ARGS)
end
