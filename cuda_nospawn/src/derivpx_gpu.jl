# --- derivpx_gpu.jl ---
""" derivatives (velocities and forces) """
@inline function derivpx_gpu(u, itraj, current_dmo, ctx)
    T = eltype(u)
    du = MVector{length(u), T}(undef)
    dh_loc = delta_E_gpu(u, ctx)

    # dx/dt
    @inbounds for i in 1:ctx.params.nmod
        k_p = 2i - 1  # index for p
        k_x = 2i      # index for x
        du[k_x] = ctx.d_wf[3, i] * u[k_p]
    end

    # v12 and non-adiabatic phase factor phc
    v12 = zero(T)
    phc = zero(T)
    if ctx.params.cmode
        v12 = ctx.d_lambda[1, 1] * u[2]
        phc = (v12 * ctx.d_lambda[1, 1]) / sqrt(dh_loc^2 + v12^2)
    elseif ctx.params.spinbo
        v12 = ctx.d_E[4]
    end
    ph = dh_loc / sqrt(dh_loc^2 + v12^2)

    # momentum coupling mode
    k_p1 = 1
    k_x1 = 2
    if current_dmo == 1
        du[k_p1] = -ctx.d_wf[3, 1] * u[k_x1] + phc
    elseif current_dmo == 4
        du[k_p1] = -ctx.d_wf[3, 1] * u[k_x1] - phc
    else
        du[k_p1] = -ctx.d_wf[3, 1] * u[k_x1]
    end

    # dp/dt = -dV/dx for tuning modes
    @inbounds for i in (1 + ctx.params.nmodc):ctx.params.nmod
        kak   = (ctx.d_kappa[3, i] - ctx.d_kappa[2, i])/2
        kakp  = (ctx.d_kappa[3, i] + ctx.d_kappa[2, i])/2
        k_p = 2i - 1  # index for p
        k_x = 2i      # index for x
        if current_dmo == 1
            du[k_p] = -(ctx.d_wf[3, i] * u[k_x] + kakp - ph * kak)
        elseif current_dmo == 4
            du[k_p] = -(ctx.d_wf[3, i] * u[k_x] + kakp + ph * kak)
        else
            du[k_p] = -(ctx.d_wf[3, i] * u[k_x] + kakp)
        end
    end

    return SVector(du)
end

