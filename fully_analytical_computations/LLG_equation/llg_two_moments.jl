#!/usr/bin/env julia
#=
llg_two_moments.jl -- LLG efectiva (Floquet, marco rotante) de dos momentos libres
F1, F2 en los sitios a < b del lead derecho, con el driver D en el sitio 0:

    ∂_t M̄_i = +J_sd M̄_i × [ s̄_i + (Ω/J_sd) ŷ + Σ_{j=1,2} ( 𝒜̄_ij M̄_j + 𝒟̄_ij D_t M̄_j ) ],
    D_t M̄_j = ∂_t M̄_j + Ω ŷ × M̄_j,
    𝒜̄_ij v = J_ij v + Γ_ij v + v × t_ij,    𝒟̄_ij v = α_ij v + Λ_ij v + v × r_ij,

y la trayectoria fisica M_i(t) = R_y(Ω (t - t_on)) M̄_i(t)  (t_on: fase cero del driver,
la de common.jl). El driver en el marco rotante es fijo, M̄_D = (0, cos θ, sin θ).

Signo: el de TDNEGF (Sunny) y el de Heisenberg para Ĥ_sd = -J_sd σ̂·M, es decir
∂_t M = +J_sd M × ⟨σ̂⟩ en el laboratorio; al pasar al marco rotante el termino
cinematico da +(Ω/J_sd) ŷ dentro del corchete.

Como ∂_t M̄ aparece a los dos lados (D_t), en cada evaluacion se resuelve el sistema
lineal 6x6
    ∂_t M̄_i - J_sd M̄_i × Σ_j 𝒟̄_ij ∂_t M̄_j = J_sd M̄_i × B_i,
    B_i = s̄_i + (Ω/J_sd) ŷ + Σ_j ( 𝒜̄_ij M̄_j + Ω 𝒟̄_ij (ŷ × M̄_j) ).

Duracion: por defecto 100 periodos desde t0 (--periods N o --tfinal T), independiente
de la duracion de TDNEGF. Convergencia al atractor (punto fijo o ciclo limite): al final
se comparan ventanas consecutivas de --diag-window periodos (defecto 4): promedio de M̄_i
y amplitud RMS de su oscilacion alrededor del promedio. Si dejan de cambiar, llego.
Integrador: RK4 de paso fijo; |M̄_i| se renormaliza tras cada paso (la deriva antes
de renormalizar se guarda como diagnostico del paso).

Entradas (sitios a, b del analitico; el indice 1 de la LLG es a y el 2 es b):
    ../output/<tag>/data/llg_elements_sites1-10.csv   J, Γ, t, α, Λ, r de (a,a) (a,b) (b,a) (b,b)
    ../output/<tag>/data/sbar_sites1-10.csv           s̄_a, s̄_b (y J_sd, Ω, θ del encabezado)
    TDNEGF (opcional): ../../two_oscillators_setup/y_precession/output/DFF/F<a>-<b>/<tag sin setup2_>_buf4_adyn0p0
        params.txt  -> t_on, t_rise, Δt_out (y control de J_sd, Ω, θ, γ_SO)
        spins_t.csv -> condicion inicial en t0 = primera muestra con t ≥ t_on + t_rise

Salidas en output/<tag>/F<a>-<b>[_ic<etiqueta>]/ (junto a este script):
    spins_t_llg.csv    t, M1_x..M3_z: M1 = D (analitico), M2 = F1, M3 = F2 en el laboratorio
                       (mismo formato que spins_t.csv de TDNEGF)
    spins_rot_llg.csv  lo mismo en el marco rotante (M̄)
    params_llg.txt     todo lo usado: parametros, elementos, s̄, condicion inicial, diagnosticos

Uso:
    julia llg_two_moments.jl --tag setup2_gso0p1_jsd0p2_th10deg_Om0p005 --sites 3,6
        [--tdnegf RUTA|none]                 # por defecto se deduce del tag y los sitios
        [--ic tdnegf|+y|-y|"x,y,z;x,y,z"]    # condicion inicial (defecto: tdnegf, o +y sin TDNEGF)
        [--t0 T] [--ton T]                   # sin TDNEGF: t0 = 0, t_on = 0
        [--periods 100 | --tfinal T]         # duracion desde t0 (defecto 100 periodos)
        [--dt 1.0] [--dtout 2.0] [--diag-window 4] [--outdir RUTA] [--overwrite]
        [--cutoff-coef] [--cutoff-value 1e-5]  # o CUTOFF_COEF=true: anula los coeficientes de
                                             # intercambio, damping y s̄ con |valor| < cutoff
                                             # (pares por partes S y A); la salida va a la misma carpeta
    Para todos los conjuntos y configuraciones DFF: bash run_llg_all.sh
=#

using LinearAlgebra
using Printf
using DelimitedFiles

const HERE   = @__DIR__
const ANA    = joinpath(HERE, "..", "output")
const TDROOT = joinpath(HERE, "..", "..", "two_oscillators_setup", "y_precession", "output", "DFF")
const YHAT   = [0.0, 1.0, 0.0]

const NAMES_A = ("J", "Γxx", "Γyy", "Γzz", "Γxz", "Γxy", "Γyz", "tx", "ty", "tz")
const NAMES_D = ("α", "Λxx", "Λyy", "Λzz", "Λxz", "Λxy", "Λyz", "rx", "ry", "rz")

# ---------------------------------------------------------------- lectura
"Encabezado '# clave=valor ...' de un CSV."
function read_meta(path::AbstractString)
    head = first(eachline(path))
    startswith(head, "#") || error("$path: sin encabezado de parametros")
    return Dict(String(a) => String(b) for (a, b) in
                (split(kv, '=') for kv in split(lstrip(head, '#')) if occursin('=', kv)))
end

"Dict (i, j, elemento) => valor de llg_elements_sites1-10.csv."
function read_elements(path::AbstractString)
    d = Dict{Tuple{Int,Int,String},Float64}()
    for (k, line) in enumerate(eachline(path))
        k == 1 && continue                                   # i,j,tipo,elemento,valor,escala
        f = split(strip(line), ',')
        length(f) ≥ 5 || continue
        d[(parse(Int, f[1]), parse(Int, f[2]), String(f[4]))] = parse(Float64, f[5])
    end
    return d
end

"Matriz 3x3 de X v = c v + Γ v + v × t (mismo orden que tensor() de llg_elements.jl)."
function tensor3(el, i::Int, j::Int, names)
    g(n) = haskey(el, (i, j, n)) ? el[(i, j, n)] : error("falta el elemento $n del par ($i,$j)")
    c, xx, yy, zz, xz, xy, yz, tx, ty, tz = map(g, names)
    return [c+xx  xy+tz  xz-ty;
            xy-tz c+yy   yz+tx;
            xz+ty yz-tx  c+zz]
end

"""
Cutoff de coeficientes (CUTOFF_COEF): anula en el Dict de elementos todos los coeficientes
con |valor| < cut de los sitios (a, b), los mismos que entran en la LLG:
  on-site (a,a) y (b,b): cada elemento por separado;
  par (a,b)/(b,a): partes simetrica S = (X_ab + X_ba)/2 y antisimetrica A = (X_ab - X_ba)/2
  por separado, y se reconstruye X_ab = S + A, X_ba = S - A.
Devuelve la lista de los coeficientes anulados (nombre y valor).
"""
function cutoff_elements!(el, a::Int, b::Int, cut::Real)
    quitados = Tuple{String,Float64}[]
    for n in (NAMES_A..., NAMES_D...)
        for i in (a, b)
            v = el[(i, i, n)]
            if abs(v) < cut
                el[(i, i, n)] = 0.0
                push!(quitados, ("$(n)_$(i)$(i)", v))
            end
        end
        S = (el[(a, b, n)] + el[(b, a, n)]) / 2
        A = (el[(a, b, n)] - el[(b, a, n)]) / 2
        abs(S) < cut && (push!(quitados, ("$(n)^S_$(a)$(b)", S)); S = 0.0)
        abs(A) < cut && (push!(quitados, ("$(n)^A_$(a)$(b)", A)); A = 0.0)
        el[(a, b, n)] = S + A
        el[(b, a, n)] = S - A
    end
    return quitados
end

"s̄_m de sbar_sites1-N.csv (columnas m, sx, sy, sz, n)."
function read_sbar(path::AbstractString, m::Int)
    for (k, line) in enumerate(eachline(path))
        (k == 1 || startswith(line, "#") || startswith(line, "m,")) && continue
        f = split(strip(line), ',')
        parse(Int, f[1]) == m && return [parse(Float64, f[2]), parse(Float64, f[3]), parse(Float64, f[4])]
    end
    error("$path: no esta el sitio $m")
end

"Pares 'clave = numero' de params.txt de TDNEGF."
function read_tdnegf_params(path::AbstractString)
    txt = read(path, String)
    d = Dict{String,Float64}()
    for m in eachmatch(r"(\S+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", txt)
        d[lstrip(m.captures[1], '(')] = parse(Float64, m.captures[2])
    end
    ax = match(r"eje de precesion\s*=\s*(\w)", txt)
    (ax === nothing || ax.captures[1] == "y") || error("$path: la LLG es para precesion alrededor de y")
    get(d, "PREC_SIGN", 1.0) == 1.0 || error("$path: PREC_SIGN distinto de 1 no soportado")
    return d
end

# ---------------------------------------------------------------- geometria
Ry(φ) = [cos(φ) 0.0 sin(φ); 0.0 1.0 0.0; -sin(φ) 0.0 cos(φ)]
cx(v) = [0.0 -v[3] v[2]; v[3] 0.0 -v[1]; -v[2] v[1] 0.0]       # cx(v) w = v × w

# ---------------------------------------------------------------- modelo
struct Model
    Jsd::Float64; Ω::Float64; θ::Float64
    A::Matrix{Matrix{Float64}}          # A[i,j] = 𝒜̄_ij (3x3)
    D::Matrix{Matrix{Float64}}          # D[i,j] = 𝒟̄_ij (3x3)
    s::Vector{Vector{Float64}}          # s̄_i
end

"∂_t M̄ (vector de 6: M̄_1, M̄_2) resolviendo el sistema lineal 6x6."
function rhs(m::Model, M::Vector{Float64})
    Mv = (M[1:3], M[4:6])
    L = Matrix{Float64}(I, 6, 6)
    r = zeros(6)
    for i in 1:2
        Bi = m.s[i] + (m.Ω / m.Jsd) * YHAT
        for j in 1:2
            Bi += m.A[i, j] * Mv[j] + m.Ω * (m.D[i, j] * cross(YHAT, Mv[j]))
            L[3i-2:3i, 3j-2:3j] .-= m.Jsd .* (cx(Mv[i]) * m.D[i, j])
        end
        r[3i-2:3i] = m.Jsd * cross(Mv[i], Bi)
    end
    return L \ r
end

function rk4_step(m::Model, M::Vector{Float64}, h::Float64)
    k1 = rhs(m, M)
    k2 = rhs(m, M .+ (h / 2) .* k1)
    k3 = rhs(m, M .+ (h / 2) .* k2)
    k4 = rhs(m, M .+ h .* k3)
    return M .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

# ---------------------------------------------------------------- CLI
function parse_vec(s::AbstractString)
    v = parse.(Float64, split(strip(s), ','))
    length(v) == 3 || error("vector mal formado: '$s'")
    return v ./ norm(v)
end

function ic_label(ic::AbstractString)
    ic == "tdnegf" && return ""
    ic == "+y" && return "_icpy"
    ic == "-y" && return "_icmy"
    return "_iccustom"
end

function main(args = ARGS)
    opt = Dict("--tag" => "", "--sites" => "", "--tdnegf" => "", "--ic" => "", "--t0" => "",
               "--tfinal" => "", "--periods" => "100", "--ton" => "", "--dt" => "1.0", "--dtout" => "2.0",
               "--diag-window" => "4", "--outdir" => "", "--cutoff-value" => "1e-5")
    overwrite = false
    cutoff_coef = lowercase(get(ENV, "CUTOFF_COEF", "false")) in ("true", "1", "yes")
    i = 1
    while i ≤ length(args)
        a = args[i]
        if a == "--overwrite"
            overwrite = true; i += 1
        elseif a == "--cutoff-coef"
            cutoff_coef = true; i += 1
        elseif haskey(opt, a) && i < length(args)
            opt[a] = args[i+1]; i += 2
        else
            error("argumento no reconocido: $a")
        end
    end
    tag = opt["--tag"]; isempty(tag) && error("falta --tag")
    isempty(opt["--sites"]) && error("falta --sites a,b")
    a, b = parse.(Int, split(opt["--sites"], ','))
    (1 ≤ a < b) || error("--sites: se espera 1 ≤ a < b (los F a la derecha del driver)")

    # ---- analitico
    ddir = joinpath(ANA, tag, "data")
    fel = joinpath(ddir, "llg_elements_sites1-10.csv")
    fsb = joinpath(ddir, "sbar_sites1-10.csv")
    isfile(fel) || error("no encuentro $fel")
    isfile(fsb) || error("no encuentro $fsb (corre antes kernel_elements.jl --sbar / run_sbar.sh)")
    # los llg_elements no tienen encabezado: el signo del kernel se lee de su origen (kernel_chi0)
    fk0 = joinpath(ddir, "kernel_chi0_sites1-10.csv")
    get(read_meta(fk0), "chi_sign", "0") == "1" ||
        error("$fk0 no tiene chi_sign=1: kernel con el signo viejo (-i J_sd). " *
              "Corrige antes con  python fix_kernel_sign.py --tag $tag  (desde fully_analytical_computations)")
    meta = read_meta(fsb)
    Jsd = parse(Float64, meta["J_sd"]); Ω = parse(Float64, meta["Omega"]); θ = parse(Float64, meta["theta"])
    γso = parse(Float64, meta["gammaSO"])
    el = read_elements(fel)
    s = [read_sbar(fsb, a), read_sbar(fsb, b)]
    cutv = parse(Float64, opt["--cutoff-value"])
    quitados = Tuple{String,Float64}[]
    if cutoff_coef
        quitados = cutoff_elements!(el, a, b, cutv)
        for (p, m) in ((1, a), (2, b)), c in 1:3
            if abs(s[p][c]) < cutv
                push!(quitados, ("sbar$("xyz"[c])_$(m)", s[p][c])); s[p][c] = 0.0
            end
        end
        @printf("CUTOFF_COEF: se anulan %d coeficientes con |valor| < %.1e\n", length(quitados), cutv)
    end
    sites = (a, b)
    A = [tensor3(el, sites[p], sites[q], NAMES_A) for p in 1:2, q in 1:2]
    D = [tensor3(el, sites[p], sites[q], NAMES_D) for p in 1:2, q in 1:2]
    model = Model(Jsd, Ω, θ, A, D, s)
    T = 2π / Ω

    # ---- TDNEGF
    tdir = opt["--tdnegf"]
    if isempty(tdir)
        rest = replace(tag, r"^setup\d+_" => "")
        tdir = joinpath(TDROOT, "F$a-$b", rest * "_buf4_adyn0p0")
        isdir(tdir) || (tdir = "none")
    end
    has_td = tdir != "none"
    ic = isempty(opt["--ic"]) ? (has_td ? "tdnegf" : "+y") : opt["--ic"]
    (ic == "tdnegf" && !has_td) && error("--ic tdnegf necesita la corrida de TDNEGF (--tdnegf RUTA)")

    dtout = parse(Float64, opt["--dtout"])
    if has_td
        tp = read_tdnegf_params(joinpath(tdir, "params.txt"))
        for (k, v, w) in (("j_sd", tp["j_sd"], Jsd), ("Ω", tp["Ω"], Ω), ("θ_max", deg2rad(tp["θ_max"]), θ), ("γso", tp["γso"], γso))
            abs(v - w) ≤ 1e-9 * max(1.0, abs(w)) || error("parametro $k distinto en TDNEGF ($v) y en el analitico ($w)")
        end
        t_on = tp["t_relax"]                                   # t_on = t_relax en params.txt
        dtout = get(tp, "Δt_out", dtout)
        geo = readdlm(joinpath(tdir, "geometry.csv"), ',', String; skipstart = 1)
        sD = parse(Int, geo[findfirst(==("D"), geo[:, 3]), 2])
        col(site) = (r = findfirst(==(string(site)), geo[:, 2]); r === nothing && error("TDNEGF: no hay LMM en el sitio $site");
                     geo[r, 3] == "F" || error("TDNEGF: el sitio $site no es un F"); parse(Int, geo[r, 1]))
        mF = (col(sD + a), col(sD + b))
        dat = readdlm(joinpath(tdir, "spins_t.csv"), ',', Float64; skipstart = 1)
        target = isempty(opt["--t0"]) ? t_on + tp["t_rise"] : parse(Float64, opt["--t0"])
        r0 = findfirst(≥(target - 1e-9), dat[:, 1]); r0 === nothing && error("TDNEGF no llega a t0 = $target")
        t0 = dat[r0, 1]
        M0_td = [dat[r0, 3mF[k]-1:3mF[k]+1] for k in 1:2]       # columnas M<m>_x,y,z
    else
        t_on = isempty(opt["--ton"]) ? 0.0 : parse(Float64, opt["--ton"])
        t0 = isempty(opt["--t0"]) ? 0.0 : parse(Float64, opt["--t0"])
        M0_td = nothing
    end
    tf = isempty(opt["--tfinal"]) ? t0 + parse(Float64, opt["--periods"]) * T : parse(Float64, opt["--tfinal"])

    # ---- condicion inicial en el marco rotante
    Mb0 = if ic == "tdnegf"
        [Ry(-Ω * (t0 - t_on)) * normalize(M0_td[k]) for k in 1:2]
    elseif ic == "+y"
        [copy(YHAT), copy(YHAT)]
    elseif ic == "-y"
        [-YHAT, -YHAT]
    else
        v = split(ic, ';'); length(v) == 2 || error("--ic: formato 'x,y,z;x,y,z'")
        [parse_vec(v[1]), parse_vec(v[2])]
    end

    outdir = isempty(opt["--outdir"]) ? joinpath(HERE, "output", tag, "F$a-$b" * ic_label(ic)) : opt["--outdir"]
    f_lab = joinpath(outdir, "spins_t_llg.csv"); f_rot = joinpath(outdir, "spins_rot_llg.csv")
    if isfile(f_lab) && !overwrite
        println("ya existe $f_lab (usa --overwrite para recalcular)")
        return nothing
    end
    mkpath(outdir)

    # ---- integracion
    dt = parse(Float64, opt["--dt"])
    nsub = max(1, round(Int, dtout / dt)); h = dtout / nsub
    nout = floor(Int, (tf - t0) / dtout + 1e-9)
    @printf("LLG F%d-%d  %s\n  J_sd=%.3g Ω=%.4g θ=%.1f°  t0=%.2f  t_final=%.2f (%.2f T)  h=%.3g  ic=%s\n",
            a, b, tag, Jsd, Ω, rad2deg(θ), t0, t0 + nout * dtout, nout * dtout / T, h, ic)
    MbD = [0.0, cos(θ), sin(θ)]
    M = vcat(Mb0[1], Mb0[2])
    traj = zeros(nout + 1, 7); traj[1, 1] = t0; traj[1, 2:7] = M
    drift = 0.0; t1 = time()
    for k in 1:nout
        for _ in 1:nsub
            M = rk4_step(model, M, h)
            for p in 0:1
                nr = norm(M[3p+1:3p+3]); drift = max(drift, abs(nr - 1))
                M[3p+1:3p+3] ./= nr
            end
        end
        traj[k+1, 1] = t0 + k * dtout; traj[k+1, 2:7] = M
    end
    @printf("  %.1f s   deriva de |M̄| por paso (antes de renormalizar): %.1e\n", time() - t1, drift)

    # ---- convergencia al atractor: ventanas consecutivas de W periodos al final
    W = parse(Float64, opt["--diag-window"])
    kW = round(Int, W * T / dtout)
    nwin = min(3, nout ÷ max(kW, 1))
    diaglines = String[]
    if nwin ≥ 2
        push!(diaglines, @sprintf("convergencia: ventanas de %g periodos al final (la ultima es [t_f - %gT, t_f])", W, W))
        for p in 1:2
            cols = 3p-1:3p+1
            st = Tuple{Vector{Float64},Float64}[]
            for w in nwin:-1:1                                          # de la mas vieja a la ultima
                seg = traj[end-w*kW+1:end-(w-1)*kW, cols]
                μw = vec(sum(seg; dims = 1)) ./ size(seg, 1)
                amp = sqrt(sum(abs2, seg .- μw') / size(seg, 1))       # RMS de la oscilacion de M̄
                push!(st, (μw, amp))
            end
            # (no reusar A ni D: son los tensores que se escriben en params_llg.txt)
            for (w, (μw, amp)) in zip(nwin:-1:1, st)
                push!(diaglines, @sprintf("  F%d  ventana -%d: media M̄ = (%+.5f, %+.5f, %+.5f)   amplitud RMS = %.5f",
                                     p, w, μw..., amp))
            end
            dμ = norm(st[end][1] - st[end-1][1]); dA = abs(st[end][2] - st[end-1][2])
            push!(diaglines, @sprintf("  F%d  ultimas dos ventanas: |Δ media| = %.2e   |Δ amplitud| = %.2e", p, dμ, dA))
        end
    else
        push!(diaglines, "convergencia: corrida demasiado corta para comparar ventanas de $W periodos")
    end
    foreach(l -> println("  ", l), diaglines)

    # ---- salida
    hdr = "t,M1_x,M1_y,M1_z,M2_x,M2_y,M2_z,M3_x,M3_y,M3_z"
    open(f_lab, "w") do io
        println(io, hdr)
        for k in axes(traj, 1)
            t = traj[k, 1]; R = Ry(Ω * (t - t_on))
            v = vcat(R * MbD, R * traj[k, 2:4], R * traj[k, 5:7])
            println(io, join([@sprintf("%.10g", t); [@sprintf("%.15e", x) for x in v]], ','))
        end
    end
    open(f_rot, "w") do io
        println(io, hdr)
        for k in axes(traj, 1)
            v = vcat(MbD, traj[k, 2:7])
            println(io, join([@sprintf("%.10g", traj[k, 1]); [@sprintf("%.15e", x) for x in v]], ','))
        end
    end
    open(joinpath(outdir, "params_llg.txt"), "w") do io
        println(io, "LLG efectiva de dos momentos (Floquet, marco rotante), RK4")
        println(io, "tag = ", tag, "   sitios del analitico: F1 = ", a, ", F2 = ", b)
        @printf(io, "J_sd = %.15g   Omega = %.15g   theta = %.15g rad (%.6g deg)   gammaSO = %.15g\n",
                Jsd, Ω, θ, rad2deg(θ), γso)
        println(io, "tdnegf = ", tdir)
        @printf(io, "t_on = %.15g   t0 = %.15g   t_final = %.15g   T = %.15g\n", t_on, t0, t0 + nout * dtout, T)
        @printf(io, "dt = %.15g   dtout = %.15g   ic = %s\n", h, dtout, ic)
        @printf(io, "Mbar1(t0) = %s\nMbar2(t0) = %s\n", string(Mb0[1]), string(Mb0[2]))
        @printf(io, "sbar1 = %s\nsbar2 = %s\n", string(s[1]), string(s[2]))
        for p in 1:2, q in 1:2
            println(io, "A_", p, q, " = ", string(A[p, q]))
            println(io, "D_", p, q, " = ", string(D[p, q]))
        end
        println(io, "signo: dM/dt = +J_sd M x <sigma> (convencion de TDNEGF/Sunny)")
        if cutoff_coef
            @printf(io, "CUTOFF_COEF = true: anulados %d coeficientes con |valor| < %.1e:\n", length(quitados), cutv)
            for (n, v) in quitados
                @printf(io, "  %s = %+.4e\n", n, v)
            end
        else
            println(io, "CUTOFF_COEF = false (todos los coeficientes)")
        end
        @printf(io, "deriva max de |Mbar| por paso = %.3e\n", drift)
        foreach(l -> println(io, l), diaglines)
    end
    println("  -> ", f_lab, "\n  -> ", f_rot)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
