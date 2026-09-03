#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra, Printf
using Plots, LaTeXStrings

pgfplotsx()

const sigma0 = ComplexF64[1 0; 0 1]
const sigmay = ComplexF64[0 -im; im 0]

# Parámetros del electrodo
const t = 1.0/sqrt(2)
const λ = 1.0/sqrt(2)
const γ² = t^2 + λ^2

const ETA = 1e-5

const AX = Dict(:axis => Dict("tick style" => "{line width=1.2pt, color=black}"))
base(; kw...) = (framestyle = :box, grid = false, dpi = 300,
                 background_color_legend = :transparent,
                 foreground_color_legend = :transparent,
                 extra_kwargs = AX, kw...)


function lead_G(E, eta, steps, H_1, H_0; verbose_small = false, verbose_large = false)

    Id = Matrix{ComplexF64}(I, size(H_0, 1), size(H_0, 1))

    A = H_1
    B = adjoint(H_1)
    C = (E + eta*im) * Id - H_0
    D = (E + eta*im) * Id - H_0

    k_used = 0

    for k in 1:steps

        Dinv = inv(D)
        A_new = A * Dinv * A
        B_new = B * Dinv * B
        C_new = C - A * Dinv * B
        D_new = D - A * Dinv * B - B * Dinv * A
        A, B, C, D = A_new, B_new, C_new, D_new

        k_used = k

        if verbose_small
            @printf("Iteration %d: norm(A) = %.2e, norm(B) = %.2e\n", k, norm(A), norm(B))
        end

        if norm(A) + norm(B) < 1e-12
            if verbose_large
                @printf("Iteration %d: norm(A) = %.2e\n", k, norm(A))
            end
            break
        end
    end

    return inv(C), k_used
end


# Función de Green de superficie en forma cerrada.
function g_exact(z::ComplexF64)
    s = sqrt(z^2 - 4γ²)
    imag(s) < 0 && (s = -s)
    return (z - s) / (2γ²)
end

function main()
    H_1 = -t*sigma0 - im*λ*sigmay      # H_{i,i+1}
    H_0 = zeros(ComplexF64, 2, 2)      # sin término de sitio

    W  = 2*sqrt(γ²)                    # borde de banda
    ωs = collect(range(-1.6W, 1.6W; length = 1200))

    reLS = similar(ωs); imLS = similar(ωs)
    reEX = similar(ωs); imEX = similar(ωs)
    offd = similar(ωs); iters = zeros(Int, length(ωs))

    for (k, ω) in enumerate(ωs)
        g_surf, it = lead_G(ω, ETA, 400, H_1, H_0)

        Σ_LS = H_1 * g_surf * adjoint(H_1)          # un solo electrodo
        Σ_EX = γ² * g_exact(ComplexF64(ω, ETA)) * sigma0

        reLS[k] = real(Σ_LS[1,1]); imLS[k] = imag(Σ_LS[1,1])
        reEX[k] = real(Σ_EX[1,1]); imEX[k] = imag(Σ_EX[1,1])
        offd[k] = abs(Σ_LS[1,2]) + abs(Σ_LS[2,1]) + abs(Σ_LS[1,1] - Σ_LS[2,2])
        iters[k] = it
    end

    err = maximum(@. sqrt((reLS - reEX)^2 + (imLS - imEX)^2))

    @printf("\nElectrodo 1D con Rashba:  t = %.2f   λ = %.2f   γ² = %.4f\n", t, λ, γ²)
    @printf("Borde de banda 2γ = %.4f      eta = %.0e\n", W, ETA)
    @printf("Iteraciones de lead_G: %d–%d\n", minimum(iters), maximum(iters))
    @printf("\nError máximo |Σ_LS - Σ_exacta| = %.3e\n", err)
    @printf("Desviación máxima de la forma ∝ σ_0 = %.3e\n", maximum(offd))

    # ── figura ───────────────────────────────────────────────────────────────
    pr = plot(; ylabel = L"\mathrm{Re}\,\Sigma(\omega)",
              legend = :topleft, legendfontsize = 12, base()...)
    plot!(pr, ωs, reEX; lc = :black, lw = 1.6, label = "Exact")
    plot!(pr, ωs, reLS; lc = :red, ls = :dash, lw = 1.2, label = "López Sancho")

    pim = plot(; xlabel = L"\omega / \gamma", ylabel = L"\mathrm{Im}\,\Sigma(\omega)",  legend = false, base()...)
    plot!(pim, ωs, imEX; lc = :black, lw = 1.6)
    plot!(pim, ωs, imLS; lc = :red, ls = :dash, lw = 1.2)

    p = plot(pr, pim; layout = (2, 1), size = (560, 560), link = :x)
    savefig(p, joinpath(@__DIR__, "fig_rashba_lead_sigma.png"))
    savefig(p, joinpath(@__DIR__, "fig_rashba_lead_sigma.svg"))
    println("\n  → fig_rashba_lead_sigma  (punteadas: bordes de banda ±2γ)")

end

main()
