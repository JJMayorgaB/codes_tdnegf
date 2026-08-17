#!/usr/bin/env julia
# Fig. 2 — AIM no interactuante: I_L(t), I_R(t), I(t),  Δ = 0,  V/Γ = 20,  T/Γ = 0.4

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TDNEGF, DifferentialEquations, LinearAlgebra
using Printf, DelimitedFiles, QuadGK, SpecialFunctions
using Plots, LaTeXStrings
pgfplotsx()

# ── parámetros en unidades de Γ ───────────────────────────────────────────────
const εc_over_Γ = 20.0
const Δ_over_Γ  = 0.0
const V_over_Γ  = 20.0
const T_over_Γ  = 0.4

const Γ     = 2.0 / εc_over_Γ      # ε_c = 2 con γ_lead = 1   ⇒  Γ = 0.1
const γc2   = Γ / 2                # acoplamiento² por lead:  Σ^r(0) = -i γc²
const Δ_dot = Δ_over_Γ * Γ
const Vbias = V_over_Γ * Γ         # = 2.0 ⇒ μ_α = ±1.0, dentro de la banda [-2,2]
const β     = 1.0 / (T_over_Γ * Γ)

const t_end = 2.0 / Γ
const dt    = t_end / 200

const N_λ1, N_λ2 = 49, 20
const Nx, Ny, Nσ, N_orb = 1, 1, 2, 1
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

#  coeficientes del lead
function lead_coeffs(Rλ, zλ, μ)
    Rs = copy(Rλ)
    Rs[1:N_λ1] .*= γc2
    ΣL = build_Σᴸ_nλ(Rs, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=1.0, μ=μ)
    ΣG = build_Σᴳ_nλ(Rs, zλ, 1, Nσ, 1, N_λ1, N_λ2; β=β, γ=1.0, μ=μ)
    χ  = build_χ_nλ(zλ,      1, Nσ, 1, N_λ1, N_λ2; β=β, γ=1.0, μ=μ)
    return ΣL, ΣG, χ
end

#modelo y propagación
Rλ, zλ = load_poles_square(N_λ1, N_λ2)
ΣL_L, ΣG_L, χ_L = lead_coeffs(Rλ, zλ, +Vbias/2)
ΣL_R, ΣG_R, χ_R = lead_coeffs(Rλ, zλ, -Vbias/2)

p = ModelParamsTDNEGF(Nx=Nx, Ny=Ny, Nσ=Nσ, N_orb=N_orb, Nα=2, N_λ1=N_λ1, N_λ2=N_λ2)
H = ComplexF64.(Δ_dot * Matrix{Float64}(I, p.Ns, p.Ns))
p.H_ab .= H; p.H0_ab .= H

ξ = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=1, y_coup=1:Ny)
blocks = [SelfEnergyBlock(:left,  p.Nc, N_λ1, N_λ2, ΣL_L, ΣG_L, χ_L, ξ),
          SelfEnergyBlock(:right, p.Nc, N_λ1, N_λ2, ΣL_R, ΣG_R, χ_R, ξ)]


pb = ExperimentalBlockRHSParams(p.H_ab, blocks, ComplexF64[0.0, 0.0], p)
u0 = zeros(ComplexF64, pb.dims_ρ_ab[1]^2 + pb.aux_layout.total_size)

@printf("Γ = %.4f   V = %.4f   β = %.2f   t_end = %.1f   |u| = %d\n",
        Γ, Vbias, β, t_end, length(u0))

@time sol = solve(ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_end), pb), Vern7();
                  saveat=dt, adaptive=true, dense=false, reltol=1e-8, abstol=1e-10)

obs = ObservablesTDNEGF(p; N_tmax=length(sol.t), N_leads=2)
obs.t = sol.t
for (it, ut) in enumerate(sol.u)
    obs.idx = it
    ptr = pointer_blocks(ut, pb.dims_ρ_ab, pb.aux_layout)
    obs_n_i!(ptr, p, obs)
    obs_Ixα!(ptr, pb, obs)
end

Γt = obs.t .* Γ
n  = obs.n_i[1, :] ./ 2

IL = obs.Iα[1, :] ./ Γ ./ 2                 # índice 1 = bloque :left
IR = -obs.Iα[2, :] ./ Γ ./ 2                # signo volteado → convención analítica
It = (IL .+ IR) ./ 2

# corriente de disipación 
# Todo por spin y con IR volteado ⇒ la conservación es  I_L - I_R = dn/dΓt.
Id = similar(n)
Id[1] = (n[2]-n[1])/(Γt[2]-Γt[1]); Id[end] = (n[end]-n[end-1])/(Γt[end]-Γt[end-1])
for i in 2:length(n)-1; Id[i] = (n[i+1]-n[i-1])/(Γt[i+1]-Γt[i-1]); end

#referencia analítica WFB (Γ = 1), Ecs. (31)-(32)
fermi_(ω,μ)   = 1/(1 + exp(clamp((ω-μ)/T_over_Γ, -500, 500)))
gsmooth_(ω,μ) = fermi_(ω,μ) - (ω < μ ? 1.0 : 0.0)

function Chigh(c, t)
    t ≤ 0 && return π/2 - atan(c)
    c == 0 && return (π/2)*exp(-t)
    I, _ = quadgk(y -> exp(-y*t)/(1 + (c+im*y)^2), 0.0, 1.0, Inf; rtol=1e-11)
    return real(im*exp(im*c*t)*I)
end
function Shigh(c, t)
    t ≤ 0 && return 0.0
    c == 0 && return (π/2)*exp(-t)
    I, _ = quadgk(y -> exp(-y*t)*(c+im*y)/(1 + (c+im*y)^2), 0.0, 1.0, Inf; rtol=1e-11)
    return imag(im*exp(im*c*t)*I)
end
Clow(b,t) = b ≤ 0 ? Chigh(-b,t) : π*exp(-t) - Chigh(b,t)
Slow(b,t) = b ≤ 0 ? Shigh(-b,t) : π*exp(-t) - Shigh(b,t)
A0(b)     = atan(b) + π/2

IL_stat_wfb(V) = sum(s*imag(digamma(0.5 + 1/(2π*T_over_Γ) +
                     im*(s*V/2 - Δ_over_Γ)/(2π*T_over_Γ))) for s in (+1,-1)) / (2π)

function IL_wfb(t, V)
    t ≤ 0 && return 0.5
    μL, μR = V/2, -V/2
    bL, bR = μL - Δ_over_Γ, μR - Δ_over_Γ
    step = exp(-t)*(A0(bL) + A0(bR)) - (2Clow(bR,t) + π*exp(-t)) -
           (2Slow(bL,t) - π*exp(-t))
    function sm(ω)
        x = ω - Δ_over_Γ
        (exp(-t)*(gsmooth_(ω,μL) + gsmooth_(ω,μR))
         - 2cos(x*t)*gsmooth_(ω,μR) - 2x*sin(x*t)*gsmooth_(ω,μL)) / (1 + x^2)
    end
    lo, hi = min(μL,μR) - 60T_over_Γ, max(μL,μR) + 60T_over_Γ
    pts = sort(unique([μR, μL]))
    smooth, _ = quadgk(sm, lo, pts..., hi; rtol=1e-10)
    return IL_stat_wfb(V) - exp(-t)*(step + smooth)/(2π)
end
IR_wfb(t, V) = -IL_wfb(t, -V)

τ   = collect(range(0.0, 2.0; length=201))
ILa = [IL_wfb(t, V_over_Γ) for t in τ]
IRa = [IR_wfb(t, V_over_Γ) for t in τ]
Ita = (ILa .+ IRa) ./ 2

@printf("máx |I_L - I_R - dn/dΓt| = %.2e   (conservación de carga)\n",
        maximum(abs.(IL .- IR .- Id)))
@printf("I_L(Γt=2): TDNEGF %.5f | WFB %.5f | I_stat %.5f\n",
        IL[end], ILa[end], IL_stat_wfb(V_over_Γ))

# figura 
plot(τ, ILa;
    xlabel = L"\Gamma t",
    ylabel = L"I(t)\;[\Gamma]",
    label = L"I_L",
    lw = 1.0,
    lc = :red,
    ls = :dash,
    framestyle = :box,
    legend = :bottomright,
    size = (300, 200),
    dpi = 300,
    xlims = (0, 2),
    ylims = (-0.5, 1.0),
    xticks = 0:0.5:2,
    yticks = -0.5:0.5:1.0,
    grid = false,
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    extra_kwargs = Dict(
        :axis => Dict(
            "tick style" => "{line width=1.5pt, color=black}"
        )
    )
)
plot!(τ, Ita; lw = 1.0, lc = :black, ls = :dash, label = L"I")
plot!(τ, IRa; lw = 1.0, lc = :blue,  ls = :dash, label = L"I_R")

plot!(Γt, IL; lw = 1.0, lc = :red,   ls = :solid, label = "")
plot!(Γt, It; lw = 1.0, lc = :black, ls = :solid, label = "")
plot!(Γt, IR; lw = 1.0, lc = :blue,  ls = :solid, label = "")

hline!([IL_stat_wfb(V_over_Γ)]; ls = :dot, lc = :gray, label = L"I_{\mathrm{stat}}")

savefig(joinpath(OUT, "anderson_fig2.svg"))
savefig(joinpath(OUT, "anderson_fig2.png"))
println(joinpath(OUT, "anderson_fig2.png"))
