# --- bfactor_gpu.jl ---
""" 
coefficient for linear term of coupling mode 
(velocity vector . nonadiabatic coupling vector) 
"""
@inline function b_factor_gpu(u, ctx)
    T = eltype(u)
    dh_loc = delta_E_gpu(u, ctx)

    val = zero(T)
    @inbounds for i in (1 + ctx.params.nmodc):ctx.params.nmod
        kak = (ctx.d_kappa[3, i] - ctx.d_kappa[2, i])/2
        val += ctx.d_wf[3, i] * kak * u[2i - 1]  # 2i-1 : index for p
    end

    fak = zero(T)
    if ctx.params.cmode
        fak = ctx.d_lambda[1, 1] / (2 * (dh_loc^2 + ctx.d_lambda[1, 1]^2 * u[2]^2))
        val = ctx.d_wf[3, 1] * u[1] * dh_loc - u[2] * val
        bfac = val * fak
    elseif ctx.params.spinbo
        fak = (ctx.d_E[4]/2) / (dh_loc^2 + ctx.d_E[4]^2)  # E[4]=g=2V_12
        bfac = -val * fak
    else
        bfac = zero(T)
    end
    return bfac
end

