#!/usr/bin/env julia
# Device Rashba 3x3:  leads Rashba (matched) vs leads limpios (mismatch)

using Pkg
Pkg.activate(@__DIR__)

using LinearAlgebra, Printf, DelimitedFiles
using Plots, LaTeXStrings
pgfplotsx()

const sigma0 = ComplexF64[1 0; 0 1]
const sigmax = ComplexF64[0 1; 1 0]
const sigmay = ComplexF64[0 -im; im 0]
const z2 = zeros(ComplexF64, 2, 2)
const z6 = zeros(ComplexF64, 6, 6)

const OUT = joinpath(@__DIR__, "output"); mkpath(OUT)

# ── bloques del device (Rashba) ───────────────────────────────────────────────
function build_blocks(; hbar = 1.0, a = 1.0, alpha = 1.0, m = 1.0, onsite = true)
    t_O  = hbar^2/(2*m*a^2)
    t_SO = alpha/(2*a)
    epsilon0 = onsite ? 4*t_O : 0.0

    h_diag = epsilon0 * sigma0
    h_off  = -t_O * sigma0 - im * t_SO * sigmax

    H_ii = [h_diag   h_off    z2;
            h_off'   h_diag   h_off;
            z2       h_off'   h_diag]

    H_ip1_i = kron(I(3), -t_O * sigma0 - im * t_SO * sigmay)
    H_i_ip1 = adjoint(H_ip1_i)

    return (t_O = t_O, t_SO = t_SO, epsilon0 = epsilon0,
            H_ii = H_ii, H_ip1_i = H_ip1_i, H_i_ip1 = H_i_ip1)
end

# ── bloques del lead limpio (t_SO = 0, mismo ancho, mismo ε₀) ─────────────────
function build_lead_blocks(t_O, epsilon0)
    h_off = ComplexF64.(-t_O * sigma0)
    H_0 = [epsilon0*sigma0  h_off            z2;
           h_off'           epsilon0*sigma0  h_off;
           z2               h_off'           epsilon0*sigma0]
    H_1 = kron(I(3), ComplexF64.(-t_O * sigma0))   # hermítico ⇒ g_L = g_R
    return H_0, H_1
end

# ── decimación iterativa: GF de superficie ───────────────────────────────────
function lead_G(E, eta, steps, H_1, H_0)
    A = H_1
    B = adjoint(H_1)
    C = (E + eta*im) * kron(I(3), sigma0) - H_0
    D = (E + eta*im) * kron(I(3), sigma0) - H_0
    for _ in 1:steps
        Dinv = inv(D)
        A_new = A * Dinv * A
        B_new = B * Dinv * B
        C_new = C - A * Dinv * B
        D_new = D - A * Dinv * B - B * Dinv * A
        A, B, C, D = A_new, B_new, C_new, D_new
        norm(A) + norm(B) < 1e-12 && break
    end
    return inv(C)
end

# ── recursión por rebanadas ──────────────────────────────────────────────────
function green_elementes(E, eta, Sigma_L, Sigma_R, H_ii, H_ip1_i, H_i_ip1)
    EI = (E + eta*im) * kron(I(3), sigma0)

    G_11_f = inv(EI - H_ii - Sigma_L)
    G_22_f = inv(EI - H_ii - H_ip1_i * G_11_f * H_i_ip1)
    G_12_f = G_11_f * H_i_ip1 * G_22_f
    G_33_f = inv(EI - H_ii - Sigma_R - H_ip1_i * G_22_f * H_i_ip1)
    G_13   = G_12_f * H_i_ip1 * G_33_f

    G_33_b = inv(EI - H_ii - Sigma_R)
    G_22_b = inv(EI - H_ii - H_i_ip1 * G_33_b * H_ip1_i)
    G_11   = inv(EI - H_ii - Sigma_L - H_i_ip1 * G_22_b * H_ip1_i)
    G_22   = inv(EI - H_ii - H_ip1_i * G_11_f * H_i_ip1 - H_i_ip1 * G_33_b * H_ip1_i)

    return G_11, G_22, G_33_f, G_13
end

# ── barrido:  leads = :soc  (bloques del device)  |  :clean  (t_SO = 0) ──────
function sweep(E; hbar = 1.0, a = 1.0, alpha = 1.0, m = 1.0, n = 100,
               onsite, leads = :soc)

    p = build_blocks(hbar = hbar, a = a, alpha = alpha, m = m, onsite = onsite)
    H_ii, H_ip1_i, H_i_ip1, t_O = p.H_ii, p.H_ip1_i, p.H_i_ip1, p.t_O

    eta = 1e-5 * t_O
    H0_l, H1_l = build_lead_blocks(t_O, p.epsilon0)

    n_E = length(E)
    T   = zeros(Float64, n_E)
    rho = zeros(Float64, n_E)

    for (k, E_k) in enumerate(E)

        if leads === :soc
            G_lead_L = lead_G(E_k, eta, n, H_ip1_i, H_ii)
            G_lead_R = lead_G(E_k, eta, n, H_i_ip1, H_ii)
            Sigma_L  = H_ip1_i * G_lead_L * H_i_ip1
            Sigma_R  = H_i_ip1 * G_lead_R * H_ip1_i
        else
            g = lead_G(E_k, eta, n, H1_l, H0_l)     # V ∝ 𝟙 ⇒ Σ = t_O² g
            Sigma_L = H1_l * g * adjoint(H1_l)
            Sigma_R = Sigma_L
        end

        Gamma_L = im * (Sigma_L - adjoint(Sigma_L))
        Gamma_R = im * (Sigma_R - adjoint(Sigma_R))

        G_11, G_22, G_33, G_13 = green_elementes(E_k, eta, Sigma_L, Sigma_R,
                                                 H_ii, H_ip1_i, H_i_ip1)

        T[k]   = real(tr(Gamma_L * G_13 * Gamma_R * adjoint(G_13)))
        rho[k] = -imag(tr(G_11 + G_22 + G_33)) / pi
    end

    return T, rho, p
end

# ── ejecución ─────────────────────────────────────────────────────────────────
n_E = 500
E   = range(-3.5, 3.5, length = n_E)

T0, rho0, p0 = sweep(E, alpha = 0.0, onsite = false, leads = :soc)    # sin Rashba
T1, rho1, p1 = sweep(E, alpha = 1.0, onsite = false, leads = :soc)    # Rashba matched
T2, rho2, p2 = sweep(E, alpha = 1.0, onsite = false, leads = :clean)  # mismatch

# ── diagnóstico ───────────────────────────────────────────────────────────────
t_O = p1.t_O
eps_n = [-2t_O*cos(nn*pi/4) for nn in 1:3]
@printf("t_O = %.3f   t_SO = %.3f   L_SO/a = π t_O/t_SO = %.2f sitios\n",
        p1.t_O, p1.t_SO, pi*p1.t_O/p1.t_SO)
@printf("umbrales de subbanda (E/t_O): %s\n",
        join((@sprintf("%+.2f", (e-2t_O)/t_O) for e in sort(eps_n)), ", "))
println("-"^62)
@printf("%-10s  %-11s  %-11s  %-11s\n", "E/t_O", "G(α=0)", "G(α=1,SOC)", "G(α=1,limp)")
println("-"^62)
for Et in [-3.0, -1.5, 0.0, 1.5, 3.0]
    k = argmin(abs.(collect(E) .- Et*t_O))
    @printf("%-10.2f  %-11.5f  %-11.5f  %-11.5f\n",
            E[k]/t_O, 0.5T0[k], 0.5T1[k], 0.5T2[k])
end

# ── figura ────────────────────────────────────────────────────────────────────
plot(0.5*T0, E,
     color = :black, lw = 1.5, linestyle = :dash,
     xlabel = L"G\ [2e^2/h]", ylabel = L"(E-\varepsilon_0)/t_O",
     xlims = (0, 3.25),
     xflip = true, framestyle = :box, size = (200, 250),
     xticks = ([0, 1, 2, 3], [L"0", L"1", L"2", L"3"]),
     yticks = ([-2, -1, 0, 1, 2], [L"-4", L"-2", L"0", L"2", L"4"]),
     legend = false,
     grid = false,
     extra_kwargs = Dict(:subplot => Dict(
         "yticklabel pos"    => "right",
         "ylabel near ticks" => nothing,
         "ylabel style"      => "{rotate=90}",
         "yticklabel style"  => "{rotate=90}",
         "xticklabel style"  => "{rotate=90}",
         "tick style"        => "{line width=1.0pt, color=black}",
     )))

plot!(0.5*T1, E, color = :black, lw = 1.5, linestyle = :solid)
plot!(0.5*T2, E, color = :red,   lw = 1.5, linestyle = :solid)

savefig(joinpath(OUT, "rashba_clean_leads.png"))
savefig(joinpath(OUT, "rashba_clean_leads.svg"))
println(joinpath(OUT, "rashba_clean_leads.png"))
