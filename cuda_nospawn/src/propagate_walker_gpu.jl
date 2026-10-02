# --- propagate_walker_gpu.jl ---
function propagate_walker_gpu!(itraj, ctx, t1, t2, ::Val{NEQ}) where {NEQ}
    T = eltype(ctx.d_f)

    dt = t2 - t1
    current_dmo = ctx.d_dm_on[itraj]

    segment_phase1 = calc_phase_gpu(itraj, dt, current_dmo, ctx)

    #rk4_step_gpu!(itraj, ctx.d_f, dt, current_dmo, ctx, Val(NEQ))
    symplct4_step_gpu!(itraj, ctx.d_f, dt, current_dmo, ctx, Val(NEQ))

    segment_phase2 = calc_phase_gpu(itraj, dt, current_dmo, ctx)

    #avg_phase = (segment_phase1 + segment_phase2) * T(0.5)
    avg_phase = sqrt(segment_phase1 * segment_phase2)
    ctx.d_Rho[itraj] *= avg_phase

    return
end

