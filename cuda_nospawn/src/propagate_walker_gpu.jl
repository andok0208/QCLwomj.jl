# --- propagate_walker_gpu.jl ---
function propagate_walker_gpu!(
    itraj, ctx, t1, t2, ::Val{NEQ}
) where {NEQ}
    T = eltype(ctx.d_f)

    u = MVector{NEQ, T}(undef)

    @inbounds for idx in 1:NEQ
        u[idx]  = ctx.d_f[idx, itraj]
    end

    dt = t2 - t1
    current_dmo = ctx.d_dm_on[itraj]

    phase1 = calc_phase_gpu(u, dt, current_dmo, ctx)

    #rk4_step_gpu!(u, dt, current_dmo, ctx)
    symplct4_step_gpu!(u, dt, current_dmo, ctx)

    phase2 = calc_phase_gpu(u, dt, current_dmo, ctx)

    #avg_phase = (phase1 + phase2)/2
    avg_phase = sqrt(phase1 * phase2)
    ctx.d_Rho[itraj] *= avg_phase

    @inbounds for idx in 1:NEQ
        ctx.d_f[idx, itraj] = u[idx]
    end

    return
end

