#!/usr/bin/env julia
# Etapa 2: dinámica. Continúa desde el checkpoint que dejó relax_state.jl —
# textura de espín Y estado electrónico — con amortiguamiento bajo, y enciende el bias.


using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "constriction_common.jl"))

const V_bias     = 0.5
# Tiempos relativos al t_0 heredado del checkpoint. El estado electrónico llega ya equilibrado, así que no hace falta esperar mucho antes del bias.
# El bias sube desde el instante en que arranca esta etapa: de t_0 a t_0+t_rise,
# es decir 100 → 110, igual que en el notebook 02. No hace falta margen previo
# porque el estado electrónico se hereda ya equilibrado desde la relajación.
const t_on_off   = 0.0
const t_rise     = 10.0
const t_duration = 600.0    # de t_0=100 a t=700: tiempo total 700

function run_case(cfg, Rλ, zλ)
    @printf("\n[%s]  config=%s  J_x=%+.3f  J_y=%+.3f\n",
            cfg.name, cfg.config, cfg.J_x, cfg.J_y)
    flush(stdout)

    S0, u_ckpt, t_0 = load_relaxed(cfg.name)
    t_end = t_0 + t_duration
    t_on  = t_0 + t_on_off
    @printf("  continuando desde t_0=%.1f hasta t_end=%.1f (bias en t=%.1f)\n",
            t_0, t_end, t_on)

    p_model, p_blocks, u0, _ = init_electrons(Rλ, zλ)
    sys = init_spins(config = cfg.config, J_x = cfg.J_x, J_y = cfg.J_y, init = S0)

    site_ranges = [get_sub(i, p_model.N_loc) for i in 1:p_model.N_sites]

    # Estado electrónico heredado (el `u0` del notebook) y H_ab reconstruido partir de la textura cargada
    length(u_ckpt) == length(u0) ||
        error("El checkpoint de $(cfg.name) tiene $(length(u_ckpt)) componentes " *
              "y aquí se esperan $(length(u0)). ¿Cambió la geometría?")
    u0 .= u_ckpt
    update_H_e!(p_model, site_ranges, masked_dipoles(sys), j_sd)

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (t_0, t_end), p_blocks)
    intg = init(prob, Vern7(); dt = Δt, save_everystep = false,
                adaptive = true, dense = false)
    llg  = Langevin(Δt; damping = damping_dyn, kT = kT_dyn)

    N_steps = Int(round(t_duration / Δt))
    obs = ObservablesTDNEGF(p_model; N_tmax = N_steps, N_leads = 2)

    started = time()
    for i in 1:N_steps
        obs.idx = i
        DifferentialEquations.step!(intg, Δt, true)
        Sunny.step!(sys, llg)

        dv = pointer_blocks(intg.u, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        ρ  = ρ_eq(E_F, β, p_model.H_ab, N_λ2, Nx, Ny, Nσ, N_orb)

        obs.t[i] = intg.t
        obs_n_i!(dv, p_model, obs)
        obs_σ_i!(dv, p_model, obs)
        obs_Ixα!(dv, p_blocks, obs)
        obs_s_i!(sys.dipoles[:, :, 1, 1], p_model, obs)
        obs_σ_i_eq!(ρ, p_model, obs)

        # realimentación mutua
        update_H_s!(Nx, Ny, sys, obs.σx_i[:, :, i], j_sd)
        update_H_e!(p_model, site_ranges, masked_dipoles(sys), j_sd)

        Δ = smooth_switch(intg.t - t_on, t_rise) * V_bias + 0im
        p_blocks.Δ_blocks[1] = +Δ/2
        p_blocks.Δ_blocks[2] = -Δ/2

        if i % 200 == 0
            @printf("  t=%6.1f/%.0f   I_L=% .4e   elapsed=%.0fs\n",
                    intg.t, t_end, obs.Iα[1, i], time() - started)
            flush(stdout)
        end
    end
    return obs
end

function main()
    sel = isempty(ARGS) ? RUNS : filter(c -> c.name in ARGS, RUNS)
    isempty(sel) && error("Ninguna corrida coincide con $(ARGS). Opciones: " *
                          join((c.name for c in RUNS), ", "))

    # Falla temprano si falta algún estado relajado, en vez de a las 7 horas.
    faltan = [c.name for c in sel if !isfile(relaxed_path(c.name))]
    isempty(faltan) || error("Faltan estados relajados: " * join(faltan, ", ") *
                             ". Corre antes relax_state.jl")

    BLAS.set_num_threads(8)
    Rλ, zλ = load_poles_square(N_λ1, N_λ2)

    print_geometry(KEEP)
    @printf("\nDINÁMICA  ·  α=%.3f  kT=%.4f  V=%.2f (bias en t_0+%.0f, subida %.0f)\n",
            damping_dyn, kT_dyn, V_bias, t_on_off, t_rise)
    @printf("Nc=%d  Ns=%d  N_λ=%d   pasos=%d (Δt=%.2f, duración=%.0f)\n",
            Ny*Nσ*N_orb, Nx*Ny*Nσ*N_orb, N_λ1 + N_λ2,
            Int(round(t_duration/Δt)), Δt, t_duration)
    @printf("Corridas: %s\nHilos de Julia: %d (se usan %d)\n",
            join((c.name for c in sel), ", "), Threads.nthreads(), length(sel))
    Threads.nthreads() < length(sel) &&
        @printf("AVISO: hay %d corridas y solo %d hilos. Relanza con --threads=%d\n",
                length(sel), Threads.nthreads(), length(sel))
    flush(stdout)

    started = time()
    Threads.@threads for i in eachindex(sel)
        cfg = sel[i]
        obs = run_case(cfg, Rλ, zλ)
        save_stage(cfg, obs)
    end
    @printf("\nListo en %.1f s. Salidas en %s\n", time() - started, OUT)
end

main()
