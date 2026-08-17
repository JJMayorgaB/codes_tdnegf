#!/usr/bin/env julia
# Fig. 1 — AIM no interactuante: n(t),  V = 0,  Δ/Γ = 8

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TDNEGF, DifferentialEquations, LinearAlgebra
using Printf, DelimitedFiles, SpecialFunctions
using Plots, LaTeXStrings
pgfplotsx()

# ── parámetros en unidades de Γ ───────────────────────────────────────────────
const εc_over_Γ = 20.0     # ancho de banda relativo; fija Γ en unidades de código
const Δ_over_Γ  = 8.0
const V_over_Γ  = 0.0
const T_over_Γ  = 0.4      # el paper no lo da para la Fig. 1

const Γ     = 2.0 / εc_over_Γ      # ε_c = 2 con γ_lead = 1   ⇒  Γ = 0.1
const γc2   = Γ / 2                # acoplamiento² por lead:  Σ^r(0) = -i γc²
const Δ_dot = Δ_over_Γ * Γ
const Vbias = V_over_Γ * Γ
const β     = 1.0 / (T_over_Γ * Γ)

const t_end = 2.0 / Γ              # Γt hasta 2
const dt    = t_end / 200

const N_λ1, N_λ2 = 49, 20
const Nx, Ny, Nσ, N_orb = 1, 1, 2, 1
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# ── coeficientes del lead: γc² reescala los residuos de banda ─────────────────
function lead_coeffs(Rλ, zλ, μ)
    Rs = copy(Rλ)
    Rs[1:N_λ1] .*= γc2
    ΣL = build_Σᴸ_nλ(Rs, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=1.0, μ=μ)
    ΣG = build_Σᴳ_nλ(Rs, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=1.0, μ=μ)
    χ  = build_χ_nλ(zλ,      1, Nσ, 1, N_λ1, N_λ2; β=β, γ=1.0, μ=μ)
    return ΣL, ΣG, χ
end

# ── modelo y propagación ──────────────────────────────────────────────────────
Rλ, zλ = load_poles_square(N_λ1, N_λ2)
ΣL_L, ΣG_L, χ_L = lead_coeffs(Rλ, zλ, +Vbias/2)
ΣL_R, ΣG_R, χ_R = lead_coeffs(Rλ, zλ, -Vbias/2)

p = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
H = ComplexF64.(Δ_dot * Matrix{Float64}(I, p.Ns, p.Ns))
p.H_ab .= H; p.H0_ab .= H

ξ = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=1, y_coup=1:Ny)   # ambos leads al sitio 1
blocks = [SelfEnergyBlock(:left,  p.Nc, N_λ1, N_λ2, ΣL_L, ΣG_L, χ_L, ξ),
          SelfEnergyBlock(:right, p.Nc, N_λ1, N_λ2, ΣL_R, ΣG_R, χ_R, ξ)]

pb = ExperimentalBlockRHSParams(p.H_ab, blocks, ComplexF64[0.0, 0.0], p)
u0 = zeros(ComplexF64, pb.dims_ρ_ab[1]^2 + pb.aux_layout.total_size)

@printf("Γ = %.4f   Δ = %.4f   β = %.2f   t_end = %.1f   |u| = %d\n",
        Γ, Δ_dot, β, t_end, length(u0))

@time sol = solve(ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_end), pb), Vern7();
                  saveat=dt, adaptive=true, dense=false, reltol=1e-8, abstol=1e-10)

obs = ObservablesTDNEGF(p; N_tmax=length(sol.t), N_leads=2)
obs.t = sol.t
for (it, ut) in enumerate(sol.u)
    obs.idx = it
    obs_n_i!(pointer_blocks(ut, pb.dims_ρ_ab, pb.aux_layout), p, obs)
end

Γt = obs.t .* Γ
n  = obs.n_i[1, :] ./ 2                     # obs.n_i suma ↑+↓ → por spin

# ── referencia analítica WFB (Γ = 1, suma de Riemann: robusta) ────────────────
n_stat_wfb() = 0.5 + sum(imag(digamma(0.5 + 1/(2π*T_over_Γ) +
                        im*(s*V_over_Γ/2 - Δ_over_Γ)/(2π*T_over_Γ)))
                        for s in (+1,-1)) / (2π)

function n_wfb(t; ωmax=200.0, Nω=40000)
    f(ω, μ) = 1 / (1 + exp((ω - μ)/T_over_Γ))
    ωs = range(-ωmax, ωmax; length=Nω); dω = step(ωs)
    acc = 0.0
    for ω in ωs
        acc += (f(ω, V_over_Γ/2) + f(ω, -V_over_Γ/2)) *
               (1 + exp(-2t) - 2exp(-t)*cos((ω - Δ_over_Γ)*t)) /
               (1 + (ω - Δ_over_Γ)^2)
    end
    return acc * dω / (2π)
end

τ  = collect(range(0.0, 2.0; length=201))
na = [n_wfb(t) for t in τ]

# ── referencia analítica CORTE SUAVE (ε_c' = -ε_c), ruta con el kernel K ──────
function n_soft(εc; Δ=Δ_over_Γ, V=V_over_Γ, T=T_over_Γ, h=0.002, N=1000, stride=5)
    Γa  = 1.0                     # unidades analíticas
    εcp = -εc
    η   = T

    # Σ^<(s): coeficientes y frecuencias
    b = exp((εcp - εc)/T)
    coef = Float64[]; freq = Float64[]
    for μ in (V/2, -V/2)
        m = exp((μ - εc)/T)
        append!(coef, [m/((b-1)*(m-1)), m/((1-b)*(m-b)), m/((1-m)*(b-m))])
        append!(freq, [εc, εcp, μ])
        @assert abs(coef[end-2] + coef[end-1] + coef[end]) < 1e-10   # regla de suma
    end
    Σ0 = -im*(Γa/(2π))*sum(coef .* freq)
    ΣLs(s) = abs(s) < 1e-7 ? Σ0 :
             (Γa*T/(2*sinh(π*T*s))) * sum(coef .* exp.(-im .* freq .* s))

    # etapa 1: D^r por la ecuación de Volterra con el kernel K
    φ0 = -im*(εcp - εc)/(π*η)
    φ(τ) = τ < 1e-9 ? φ0 : exp(im*Δ*τ)*(exp(-im*εcp*τ) - exp(-im*εc*τ))/sinh(π*η*τ)
    Ck = -im*Γa*η / (exp((εcp - εc)/η) - 1)

    φv = [φ(j*h) for j in 0:N]
    J  = zeros(ComplexF64, N+1)
    for m in 1:N
        J[m+1] = J[m] + 0.5h*(φv[m] + φv[m+1])
    end
    k = [Ck*exp(-im*Δ*j*h)*J[j+1] for j in 0:N]

    d = zeros(ComplexF64, N+1)
    for m in 0:N
        acc = 0.5*k[m+1]*(-im)
        for j in 1:m-1
            acc += k[m-j+1]*d[j+1]
        end
        d[m+1] = -im*exp(-im*Δ*m*h) + h*acc
    end

    # etapa 2: n(t) = -i ∫∫ D^r(u) Σ^<(v-u) [D^r(v)]*
    sl = [ΣLs(m*h) for m in -N:N]
    Gm = [d[j]*conj(d[kk])*sl[N+1 + (kk-j)] for j in 1:N+1, kk in 1:N+1]

    idx = 0:stride:N
    ts  = collect(idx) .* h
    nt  = zeros(length(idx))
    for (pp, m) in enumerate(idx)
        w = ones(m+1); w[1] = 0.5; w[end] = 0.5
        nt[pp] = real(-im * h^2 * dot(w, view(Gm, 1:m+1, 1:m+1), w))
    end
    return ts, nt
end

ts_s, n_s = n_soft(εc_over_Γ)

@printf("n(0): TDNEGF %.2e | soft %.2e\n", n[1], n_s[1])
@printf("máximo:  TDNEGF %.5f en Γt=%.3f | soft %.5f en Γt=%.3f | WFB %.5f en Γt=%.3f\n",
        maximum(n), Γt[argmax(n)], maximum(n_s), ts_s[argmax(n_s)],
        maximum(na), τ[argmax(na)])
@printf("n(Γt=2): TDNEGF %.5f | soft %.5f | WFB %.5f | n_stat %.5f\n",
        n[end], n_s[end], na[end], n_stat_wfb())

# ── figura y datos ────────────────────────────────────────────────────────────
plot(ts_s, n_s;
    xlabel = L"\Gamma t",
    ylabel = L"n(t)",
    label  = L"\mathrm{soft\ cutoff}",
    lw = 1.0,
    lc = :blue,
    ls = :dash,
    framestyle = :box,
    legend = :topright,
    size = (300, 200),
    dpi = 300,
    xlims = (0, 2),
    ylims = (0, 0.1),
    xticks = 0:0.5:2,
    yticks = 0:0.025:0.1,
    grid = false,
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    extra_kwargs = Dict(
        :axis => Dict(
            "tick style" => "{line width=1.5pt, color=black}"
        )
    )
)
plot!(Γt, n; lw = 1.0, lc = :red, ls = :solid, label = L"\mathrm{TDNEGF}")

savefig(joinpath(OUT, "anderson_fig1.svg"))
savefig(joinpath(OUT, "anderson_fig1.png"))
println(joinpath(OUT, "anderson_fig1.png"))
