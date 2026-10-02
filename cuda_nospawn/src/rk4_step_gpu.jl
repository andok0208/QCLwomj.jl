# --- rk4_step_gpu.jl ---
""" 4th order Runge-Kutta """
@inline function rk4_step_gpu!(itraj, d_f, dt, current_dmo, ctx, ::Val{NEQ}) where {NEQ}
    T = eltype(d_f)

    dt2 = dt / 2
    dt6 = dt / 6

    u = SVector{NEQ, T}(ntuple(idx -> d_f[idx, itraj], Val(NEQ)))

    k1 = derivpx_gpu(u, itraj, current_dmo, ctx)
    u_next = u + dt2 * k1

    k2 = derivpx_gpu(u_next, itraj, current_dmo, ctx)
    u_next = u + dt2 * k2

    k3 = derivpx_gpu(u_next, itraj, current_dmo, ctx)
    u_next = u + dt  * k3

    k4 = derivpx_gpu(u_next, itraj, current_dmo, ctx)
    u_next = u + dt6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @inbounds for idx in 1:NEQ
        d_f[idx, itraj] = u_next[idx]
    end
    return nothing
end

