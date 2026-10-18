#!/usr/bin/env julia
#=
llg_elements.jl -- elementos de intercambio y damping de la LLG efectiva en el
marco rotante a partir de χ̌_{ij,k}(0) e i∂_ωχ̌_{ij,k}(0) (kernel_chi0_<pares>.csv,
escrito por kernel_elements.jl --chi0; tambien sirve el resumen
kernel_chi0_kernel_<pares>.csv de --kernel, y entonces la salida lleva _kernel_ en el nombre).

Usa χ̌_{-k}(0) = [χ̌_k(0)]* e i∂_ωχ̌_{-k}(0) = [i∂_ωχ̌_k(0)]*, con lo que las
formulas de la seccion quedan:

  intercambio (con χ̌ = χ̌_k(0)):
    J      = (2 xx0 + yy0)/3
    Γ^xx   = (xx0 - yy0)/3 + 2 Re xx2      Γ^yy = -2(xx0 - yy0)/3
    Γ^zz   = (xx0 - yy0)/3 - 2 Re xx2      Γ^xz = 2 Im xx2
    Γ^xy   = Re xy1 + Re yx1               Γ^yz = Im xy1 + Im yx1
    t^x    = Im yx1 - Im xy1               t^y  = -xz0        t^z = Re xy1 - Re yx1
  damping: lo mismo con χ̌ -> i∂_ωχ̌ (α, Λ, r en lugar de J, Γ, t).

Imprime, por par, los 9 + 9 elementos; para (1,2) y (2,1) su parte simetrica y
antisimetrica en sitio; y la lista de todos ordenada por magnitud, con su escala
en la LLG (J_sd y Ω se leen del encabezado del CSV):
    intercambio (reactivo):  g J_sd |X| / Ω   (frecuencia frente a la del marco rotante)
    damping (multiplica D_t m, adimensional como un Gilbert):  g J_sd |X|
Escribe llg_elements_<pares>.csv junto al archivo de entrada.

Cualquier conjunto de pares: los elementos se calculan para todos los pares del
CSV; se imprimen en detalle solo las configuraciones D=0, F=a, F=b pedidas con
--configs (por defecto "1,2;3,6;5,9", las que existan en el archivo).
Escribe en la carpeta del CSV:
    llg_elements_<label>.csv     i, j, tipo, elemento, valor, escala (todos los pares)
    llg_elements_SA_<label>.csv  i<j: partes simetrica (S) y antisimetrica (A) en sitio

Uso:
    julia llg_elements.jl                                   # output/setup2_gso0p1_jsd0p2_th10deg_Om0p005/data:
                                                            # sites1-10 si existe, si no 1,1;1,2;2,1;2,2
    julia llg_elements.jl RUTA/kernel_chi0_sites1-10.csv [--g 1.0] [--configs "1,2;3,6;5,9"]
=#

using Printf

const DEFAULT_DIR  = joinpath(@__DIR__, "output", "setup2_gso0p1_jsd0p2_th10deg_Om0p005", "data")
const DEFAULT_FILE = let f = joinpath(DEFAULT_DIR, "kernel_chi0_sites1-10.csv")
    isfile(f) ? f : joinpath(DEFAULT_DIR, "kernel_chi0_1-1_1-2_2-1_2-2.csv")
end

const NAMES_A = ("J", "Γxx", "Γyy", "Γzz", "Γxz", "Γxy", "Γyz", "tx", "ty", "tz")
const NAMES_D = ("α", "Λxx", "Λyy", "Λzz", "Λxz", "Λxy", "Λyz", "rx", "ry", "rz")

"Lee el CSV: Dict((m,n) => Dict(comp => (χ̌(0), i∂χ̌(0))))."
function read_chi0(path::AbstractString)
    data = Dict{Tuple{Int,Int},Dict{String,Tuple{ComplexF64,ComplexF64}}}()
    pairs = Tuple{Int,Int}[]
    for line in eachline(path)
        (startswith(line, "#") || startswith(line, "m,")) && continue
        f = split(strip(line), ',')
        length(f) == 7 || continue
        m, n = parse(Int, f[1]), parse(Int, f[2])
        v = parse.(Float64, f[4:7])
        haskey(data, (m, n)) || (data[(m, n)] = Dict{String,Tuple{ComplexF64,ComplexF64}}(); push!(pairs, (m, n)))
        data[(m, n)][String(f[3])] = (complex(v[1], v[2]), complex(v[3], v[4]))
    end
    return data, pairs
end

"""
Los 10 elementos (J o α, Γ o Λ (6), t o r (3)) a partir de los armonicos k ≥ 0
x = Dict(comp => valor) (χ̌_k(0) o i∂χ̌_k(0)). Devuelve tambien la parte
imaginaria de los k = 0 (deberia ser ≈ 0).
"""
function elements(x)
    xx0, yy0, xz0 = x["xx0"], x["yy0"], x["xz0"]
    xy1, yx1, xx2 = x["xy1"], x["yx1"], x["xx2"]
    a, b, c = real(xx0), real(yy0), real(xz0)
    J   = (2a + b) / 3
    Γxx = (a - b) / 3 + 2real(xx2)
    Γyy = -2(a - b) / 3
    Γzz = (a - b) / 3 - 2real(xx2)
    Γxz = 2imag(xx2)
    Γxy = real(xy1) + real(yx1)
    Γyz = imag(xy1) + imag(yx1)
    tx  = imag(yx1) - imag(xy1)
    ty  = -c
    tz  = real(xy1) - real(yx1)
    imk0 = maximum(abs.(imag.((xx0, yy0, xz0))))
    return (J, Γxx, Γyy, Γzz, Γxz, Γxy, Γyz, tx, ty, tz), imk0
end

"Matriz 3x3 en el marco rotante a partir de los 10 elementos (J δ + Γ + ε t)."
function tensor(e)
    J, Γxx, Γyy, Γzz, Γxz, Γxy, Γyz, tx, ty, tz = e
    return [J+Γxx  Γxy+tz  Γxz-ty;
            Γxy-tz J+Γyy   Γyz+tx;
            Γxz+ty Γyz-tx  J+Γzz]
end

function show_block(title, names, e)
    println("    ", title)
    for (nm, v) in zip(names, e)
        @printf("      %-4s = %+.6e\n", nm, v)
    end
end

function show_matrix(title, M)
    println("    ", title)
    for r in 1:3
        @printf("      [% .6e  % .6e  % .6e]\n", M[r, 1], M[r, 2], M[r, 3])
    end
end

"J_sd y Ω del encabezado (# clave=valor ...) del CSV."
function read_meta(path::AbstractString)
    head = first(eachline(path))
    d = Dict(String(a) => parse(Float64, b) for (a, b) in
             (split(kv, '=') for kv in split(lstrip(head, '#')) if occursin('=', kv)))
    # chi_sign=1: prefactor +i J_sd. Sin esa clave el CSV es del prefactor viejo -i J_sd
    get(d, "chi_sign", 0.0) == 1.0 ||
        error("$path: kernel con el signo viejo (-i J_sd); corrigelo antes con " *
              "python fix_kernel_sign.py --tag <tag>")
    return d["J_sd"], d["Omega"]
end

"""
Configuracion D=0, F=a, F=b: imprime sus 4 pares (a,a), (b,b), (a,b), (b,a) con
matrices, la parte S/A en sitio de (a,b) y la lista de sus elementos ordenada por
magnitud, con la escala en la LLG.
"""
function show_config(a, b, EA, ED, IA, ID, scale)
    prs = [(a, a), (b, b), (a, b), (b, a)]
    all(haskey(EA, pr) for pr in prs) || (println("\n(configuracion F=$a, F=$b: faltan pares, se omite)"); return)
    println("\n", "#"^78, "\n# configuracion D=0, F=$a, F=$b\n", "#"^78)
    for pr in prs
        i, j = pr
        @printf("\npar (i,j) = (%d,%d)%s   [comprobacion: max|Im k=0|: χ̌ %.1e, i∂χ̌ %.1e]\n",
                i, j, i == j ? " (on-site: J no da torque)" : "", IA[pr], ID[pr])
        show_block("intercambio", NAMES_A, EA[pr])
        show_block("damping", NAMES_D, ED[pr])
        show_matrix("Ā_ij", tensor(EA[pr]))
        show_matrix("D̄_ij", tensor(ED[pr]))
    end
    @printf("\nsitios (%d,%d) y (%d,%d): X^S = (X_ij + X_ji)/2,  X^A = (X_ij - X_ji)/2\n", a, b, b, a)
    for (names, E) in ((NAMES_A, EA), (NAMES_D, ED)), (k, nm) in enumerate(names)
        @printf("      %-4s   S = %+.6e   A = %+.6e\n", nm,
                (E[(a, b)][k] + E[(b, a)][k]) / 2, (E[(a, b)][k] - E[(b, a)][k]) / 2)
    end
    rows = Tuple{Float64,String,Tuple{Int,Int},String}[]
    for pr in prs, (names, E, kind) in ((NAMES_A, EA, "intercambio"), (NAMES_D, ED, "damping"))
        for (k, nm) in enumerate(names)
            (nm == "J" && pr[1] == pr[2]) && continue
            push!(rows, (E[pr][k], nm, pr, kind))
        end
    end
    sort!(rows; by = r -> -abs(r[1]))
    vmax = abs(rows[1][1])
    @printf("\n%d elementos (sin J_ii), ordenados por |valor|:\n", length(rows))
    println("   escala: intercambio -> g J_sd |X|/Ω ;  damping -> g J_sd |X|")
    @printf("   %4s  %-12s %-6s %-7s %14s %10s %12s\n", "#", "tipo", "elem", "(i,j)", "valor", "/max", "escala")
    for (r, (v, nm, pr, kind)) in enumerate(rows)
        @printf("   %4d  %-12s %-6s (%d,%d)  %+.6e  %9.1e  %10.2e%s\n", r, kind, nm, pr..., v, abs(v) / vmax,
                scale(v, kind), kind == "intercambio" ? " Ω" : "")
    end
end

function main(args = ARGS)
    path = DEFAULT_FILE; g = 1.0; cfgs = "1,2;3,6;5,9"
    i = 1
    while i ≤ length(args)
        if args[i] == "--g" && i < length(args)
            g = parse(Float64, args[i+1]); i += 2
        elseif args[i] == "--configs" && i < length(args)
            cfgs = args[i+1]; i += 2
        else
            path = args[i]; i += 1
        end
    end
    isfile(path) || error("no encuentro $path (corre antes kernel_elements.jl --chi0 o --kernel)")
    data, pairs = read_chi0(path)
    Jsd, Ω = read_meta(path)
    scale(v, kind) = kind == "intercambio" ? g * Jsd * abs(v) / Ω : g * Jsd * abs(v)
    println("="^78)
    println("Elementos de la LLG efectiva (marco rotante) desde ", basename(path))
    @printf("J_sd = %g   Ω = %g   g = %g   |   %d pares\n", Jsd, Ω, g, length(pairs))
    println("="^78)

    EA = Dict{Tuple{Int,Int},NTuple{10,Float64}}(); ED = Dict{Tuple{Int,Int},NTuple{10,Float64}}()
    IA = Dict{Tuple{Int,Int},Float64}();            ID = Dict{Tuple{Int,Int},Float64}()
    for pr in pairs
        x = data[pr]
        EA[pr], IA[pr] = elements(Dict(k => v[1] for (k, v) in x))
        ED[pr], ID[pr] = elements(Dict(k => v[2] for (k, v) in x))
    end
    @printf("comprobacion (todos los pares): max|Im k=0|  χ̌ %.1e   i∂χ̌ %.1e\n",
            maximum(values(IA)), maximum(values(ID)))

    # configuraciones D-F-F pedidas
    for tok in split(cfgs, ';'; keepempty = false)
        a, b = parse.(Int, split(strip(tok), ','))
        show_config(a, b, EA, ED, IA, ID, scale)
    end

    stem = replace(basename(path), "kernel_chi0_" => "", ".csv" => "")
    dir  = dirname(abspath(path))
    # tabla larga: todos los pares y elementos (incluye J_ii, marcado sin torque por la escala)
    out = joinpath(dir, "llg_elements_" * stem * ".csv")
    open(out, "w") do io
        println(io, "i,j,tipo,elemento,valor,escala")
        for pr in pairs, (names, E, kind) in ((NAMES_A, EA, "intercambio"), (NAMES_D, ED, "damping"))
            for (k, nm) in enumerate(names)
                @printf(io, "%d,%d,%s,%s,%.15e,%.6e\n", pr..., kind, nm, E[pr][k], scale(E[pr][k], kind))
            end
        end
    end
    # parte simetrica / antisimetrica en sitio de todos los pares i < j con (j,i) disponible
    outSA = joinpath(dir, "llg_elements_SA_" * stem * ".csv")
    open(outSA, "w") do io
        println(io, "i,j,tipo,elemento,S,A")
        for (i, j) in pairs
            (i < j && haskey(EA, (j, i))) || continue
            for (names, E, kind) in ((NAMES_A, EA, "intercambio"), (NAMES_D, ED, "damping")), (k, nm) in enumerate(names)
                @printf(io, "%d,%d,%s,%s,%.15e,%.15e\n", i, j, kind, nm,
                        (E[(i, j)][k] + E[(j, i)][k]) / 2, (E[(i, j)][k] - E[(j, i)][k]) / 2)
            end
        end
    end
    println("\n  -> ", out, "\n  -> ", outSA)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
