#!/usr/bin/env julia
# Fig. 3 — AIM no interactuante: corriente de desplazamiento,  Δ = 0,  V = 0,  T/Γ = 0.4

using Pkg
Pkg.activate(@__DIR__)

using TDNEGF, DifferentialEquations, LinearAlgebra
using Printf, DelimitedFiles
using Plots, LaTeXStrings
pgfplotsx()

# ── parámetros en unidades de Γ ───────────────────────────────────────────────
const εc_over_Γ = 20.0
const Δ_over_Γ  = 0.0
const V_over_Γ  = 0.0
const T_over_Γ  = 0.4

const Γ     = 2.0 / εc_over_Γ      # ε_c = 2 con γ_lead = 1   ⇒  Γ = 0.1
const γc2   = Γ / 2                # acoplamiento² por lead:  Σ^r(0) = -i γc²
const Δ_dot = Δ_over_Γ * Γ
const Vbias = V_over_Γ * Γ
const β     = 1.0 / (T_over_Γ * Γ)

const t_end = 2.0 / Γ
const dt    = t_end / 200

const N_λ1, N_λ2 = 49, 20
const Nx, Ny, Nσ, N_orb = 1, 1, 2, 1
const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# ── coeficientes del lead ─────────────────────────────────────────────────────
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

ξ = build_ξ_an(Nx, Ny, Nσ, N_orb; xcol=1, y_coup=1:Ny)
blocks = [SelfEnergyBlock(:left,  p.Nc, N_λ1, N_λ2, ΣL_L, ΣG_L, χ_L, ξ), SelfEnergyBlock(:right, p.Nc, N_λ1, N_λ2, ΣL_R, ΣG_R, χ_R, ξ)]

pb = ExperimentalBlockRHSParams(p.H_ab, blocks, ComplexF64[0.0, 0.0], p)
u0 = zeros(ComplexF64, pb.dims_ρ_ab[1]^2 + pb.aux_layout.total_size)

@printf("Γ = %.4f   Δ = %.4f   V = %.4f   β = %.2f   t_end = %.1f\n",
        Γ, Δ_dot, Vbias, β, t_end)

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
n  = obs.n_i[1, :] ./ 2                     # obs.n_i suma ↑+↓ 

# corrientes 
IL_p = obs.Iα[1, :] ./ Γ
IR_p = -obs.Iα[2, :] ./ Γ
I_tdnegf = IL_p .- IR_p                     # I_disp, directo del EOM
I_trans  = (IL_p .+ IR_p) ./ 2              # corriente de transporte

# Chequeo: la misma cantidad por derivada numérica de n
dn = similar(n)
dn[1]   = (n[2] - n[1]) / (Γt[2] - Γt[1])
dn[end] = (n[end] - n[end-1]) / (Γt[end] - Γt[end-1])
for i in 2:length(n)-1
    dn[i] = (n[i+1] - n[i-1]) / (Γt[i+1] - Γt[i-1])
end
I_deriv = 2 .* dn                           # d(2n)/dΓt

#referencia analítica: corte suave, derivada exacta de n(t)
function I_disp_soft(εc; Δ=Δ_over_Γ, V=V_over_Γ, T=T_over_Γ, h=0.002, N=1000)
    Γa  = 1.0                  
    εcp = -εc
    η   = T

    b = exp((εcp - εc)/T)
    coef = Float64[]; freq = Float64[]
    for μ in (V/2, -V/2)
        m = exp((μ - εc)/T)
        append!(coef, [m/((b-1)*(m-1)), m/((1-b)*(m-b)), m/((1-m)*(b-m))])
        append!(freq, [εc, εcp, μ])
        @assert abs(coef[end-2] + coef[end-1] + coef[end]) < 1e-10
    end
    Σ0 = -im*(Γa/(2π))*sum(coef .* freq)
    ΣLs(s) = abs(s) < 1e-7 ? Σ0 :
             (Γa*T/(2*sinh(π*T*s))) * sum(coef .* exp.(-im .* freq .* s))

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

    sl = [ΣLs(m*h) for m in -N:N]
    Id = zeros(N+1)
    for m in 1:N                               
        w = ones(m+1); w[1] = 0.5; w[end] = 0.5
        integ = h * sum(w[j+1] * sl[N+1 + (j-m)] * conj(d[j+1]) for j in 0:m)
        Id[m+1] = -2*imag(d[m+1] * integ)
    end
    return collect(0:N).*h, Id
end

ts_a, Id_a = I_disp_soft(εc_over_Γ)
I_soft = -2 .* Id_a                             

# ── diagnóstico ───────────────────────────────────────────────────────────────
ja = argmax(I_soft); jt = argmax(I_tdnegf)
@printf("máximo:  TDNEGF %.5f en Γt=%.3f  |  soft %.5f en Γt=%.3f\n",
        I_tdnegf[jt], Γt[jt], I_soft[ja], ts_a[ja])
@printf("regla de suma ∫I_disp dΓt (= 2·n(2)):  TDNEGF %.5f  |  soft %.5f  |  2n(2) = %.5f\n",
        sum(0.5*(I_tdnegf[i]+I_tdnegf[i+1])*(Γt[i+1]-Γt[i]) for i in 1:length(Γt)-1),
        sum(0.5*(I_soft[i]+I_soft[i+1])*(ts_a[i+1]-ts_a[i]) for i in 1:length(ts_a)-1),
        2n[end])
println()
println("  Chequeos internos de TDNEGF:")
@printf("    máx |I_disp - d(2n)/dΓt|      = %.2e   (Ψ/ξ contra ρ)\n",
        maximum(abs.(I_tdnegf .- I_deriv)))
@printf("    máx |I_L - (I + I_disp/2)|    = %.2e   (descomposición del paper)\n",
        maximum(abs.(IL_p .- (I_trans .+ I_tdnegf ./ 2))))
@printf("    máx |I_R - (I - I_disp/2)|    = %.2e\n",
        maximum(abs.(IR_p .- (I_trans .- I_tdnegf ./ 2))))
@printf("    máx |I| (transporte, V=0 ⇒ 0) = %.2e\n", maximum(abs.(I_trans)))
println()

# ── figura ────────────────────────────────────────────────────────────────────
# dashed = analítico (corte suave),  solid = TDNEGF
plot(ts_a, I_soft;
    xlabel = L"\Gamma t",
    ylabel = L"I_{\mathrm{disp}}(t)\;[\Gamma]",
    label = L"\mathrm{soft\ cutoff}",
    lw = 1.0,
    lc = :red,
    ls = :dash,
    framestyle = :box,
    legend = :topright,
    size = (300, 200),
    dpi = 300,
    xlims = (0, 0.75),
    ylims = (0.4, 2.25),
    xticks = 0:0.25:0.75,
    yticks = 0.5:0.5:2.0,
    grid = false,
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    extra_kwargs = Dict(
        :axis => Dict(
            "tick style" => "{line width=1.5pt, color=black}"
        )
    )
)
plot!(Γt, I_tdnegf; lw = 1.0, lc = :red, ls = :solid, label = L"\mathrm{TDNEGF}")

savefig(joinpath(OUT, "anderson_fig3.svg"))
savefig(joinpath(OUT, "anderson_fig3.png"))
println(joinpath(OUT, "anderson_fig3.png"))
