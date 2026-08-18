#!/usr/bin/env julia

# Etapa 1: relajación. 


using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "constriction_common.jl"))

const t_relax = 100.0

function relax_case(cfg, Rλ, zλ)
    @printf("\n[%s]  config=%s  J_x=%+.3f  J_y=%+.3f\n",
            cfg.name, cfg.config, cfg.J_x, cfg.J_y)
    flush(stdout)

    p_model, p_blocks, u0, _ = init_electrons(Rλ, zλ)
    sys = init_spins(config = cfg.config, J_x = cfg.J_x, J_y = cfg.J_y)

    prob = ODEProblem(eom_tdnegf_blocks!, u0, (0.0, t_relax), p_blocks)
    intg = init(prob, Vern7(); dt = Δt, save_everystep = false, adaptive = true, dense = false)

    llg  = Langevin(Δt; damping = damping_relax, kT = kT_relax)

    N_steps = Int(round(t_relax / Δt))
    obs = ObservablesTDNEGF(p_model; N_tmax = N_steps, N_leads = 2)
    site_ranges = [get_sub(i, p_model.N_loc) for i in 1:p_model.N_sites]

    started = time()
    for i in 1:N_steps
        obs.idx = i
        DifferentialEquations.step!(intg, Δt, true)
        Sunny.step!(sys, llg)

        dv = pointer_blocks(intg.u, p_blocks.dims_ρ_ab, p_blocks.aux_layout)
        obs.t[i] = intg.t
        obs_σ_i!(dv, p_model, obs)
        obs_s_i!(sys.dipoles[:, :, 1, 1], p_model, obs)

        update_H_s!(Nx, Ny, sys, obs.σx_i[:, :, i], j_sd)
        update_H_e!(p_model, site_ranges, masked_dipoles(sys), j_sd)

        if i % 200 == 0
            M, Nv = order_parameters(obs, i)
            @printf("  t=%6.1f/%.0f   |M|=%.4f  |N|=%.4f   elapsed=%.0fs\n",
                    intg.t, t_relax, norm(M[:, i]), norm(Nv[:, i]), time() - started)
            flush(stdout)
        end
    end

    M, Nv = order_parameters(obs, N_steps)
    @printf("  final: |M|=%.4f  |N|=%.4f\n", norm(M[:, end]), norm(Nv[:, end]))

    save_relaxed(cfg.name, sys, intg.u, t_relax)
    return nothing
end

function main()
    sel = isempty(ARGS) ? RUNS : filter(c -> c.name in ARGS, RUNS)
    isempty(sel) && error("Ninguna corrida coincide con $(ARGS). Opciones: " *  join((c.name for c in RUNS), ", "))

    BLAS.set_num_threads(8)
    Rλ, zλ = load_poles_square(N_λ1, N_λ2)

    print_geometry(KEEP)
    @printf("\nRELAJACIÓN  ·  α=%.3f  kT=%.4f  sin bias\n", damping_relax, kT_relax)
    @printf("Nc=%d  Ns=%d  N_λ=%d   pasos=%d (Δt=%.2f, t_relax=%.0f)\n",   Ny*Nσ*N_orb, Nx*Ny*Nσ*N_orb, N_λ1 + N_λ2, Int(round(t_relax/Δt)), Δt, t_relax)
    @printf("Corridas: %s\nHilos de Julia: %d (se usan %d)\n", join((c.name for c in sel), ", "), Threads.nthreads(), length(sel))
    Threads.nthreads() < length(sel) &&
        @printf("AVISO: hay %d corridas y solo %d hilos. Relanza con --threads=%d\n", length(sel), Threads.nthreads(), length(sel))
    flush(stdout)

    started = time()
    Threads.@threads for i in eachindex(sel)
        relax_case(sel[i], Rλ, zλ)
    end
    @printf("\nListo en %.1f s. Estados relajados en %s\n", time() - started, OUT)
end

main()
