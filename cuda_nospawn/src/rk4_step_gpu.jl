# --- rk4_step_gpu.jl ---
""" 4th order Runge-Kutta """
@inline function rk4_step_gpu!(u, dt, current_dmo, ctx)
    T = eltype(u)

    dt2 = T(dt / 2)
    dt6 = T(dt / 6)

    u_next = MVector{length(u), T}(undef)

    k1 = derivpx_gpu(u, current_dmo, ctx)
    u_next = u + dt2 * k1

    k2 = derivpx_gpu(u_next, current_dmo, ctx)
    u_next = u + dt2 * k2

    k3 = derivpx_gpu(u_next, current_dmo, ctx)
    u_next = u + dt  * k3

    k4 = derivpx_gpu(u_next, current_dmo, ctx)

    @inbounds for idx in 1:length(u)
        u[idx] = u[idx] + dt6 * (k1[idx] + 2k2[idx] + 2k3[idx] + k4[idx])
    end

    return nothing
end

